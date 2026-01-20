import os
import argparse
import numpy as np
from tqdm import tqdm
import logging
from glob import glob
from pandas import DataFrame, Series
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from utils import setup_seed
from model.adapter import AdaptedCLIP
from model.clip import create_model
from dataset import get_dataset, DOMAINS
from forward_utils import (
    get_adapted_text_embedding,
    calculate_similarity_map,
    metrics_eval,
)
import warnings

warnings.filterwarnings("ignore")

cpu_num = 10
os.environ["OMP_NUM_THREADS"] = str(cpu_num)
torch.set_num_threads(cpu_num)
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def get_predictions(
        model: nn.Module,
        class_text_embeddings: torch.Tensor,
        test_loader: DataLoader,
        device: str,
        img_size: int,
        dataset: str = "MVTec",
):
    masks = []
    labels = []
    preds = []
    preds_image = []
    file_names = []
    for input_data in tqdm(test_loader):
        image = input_data["image"].to(device)
        mask = input_data["mask"].cpu().numpy()
        label = input_data["label"].cpu().numpy()
        file_name = input_data["file_name"]
        class_name = input_data["class_name"]
        assert len(set(class_name)) == 1, "mixed class not supported"
        masks.append(mask)
        labels.append(label)
        file_names.extend(file_name)

        epoch_text_feature = class_text_embeddings
        patch_features, det_feature = model(image)

        pred = det_feature @ epoch_text_feature
        pred = (pred[:, 1] + 1) / 2
        preds_image.append(pred.cpu().numpy())
        patch_preds = []
        for f in patch_features:
            patch_pred = calculate_similarity_map(
                f, epoch_text_feature, img_size, test=True, domain=DOMAINS[dataset]
            )
            patch_preds.append(patch_pred)
        patch_preds = torch.cat(patch_preds, dim=1).sum(1).cpu().numpy()
        preds.append(patch_preds)
    masks = np.concatenate(masks, axis=0)
    labels = np.concatenate(labels, axis=0)
    preds = np.concatenate(preds, axis=0)
    preds_image = np.concatenate(preds_image, axis=0)
    return masks, labels, preds, preds_image, file_names


def main():
    parser = argparse.ArgumentParser(description="Testing")
    parser.add_argument("--model_name", type=str, default="ViT-L-14-336")
    parser.add_argument("--img_size", type=int, default=518)
    parser.add_argument("--relu", action="store_true")
    parser.add_argument("--dataset", type=str, default="MVTec")
    parser.add_argument("--shot", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=111)
    parser.add_argument("--save_path", type=str, default="ckpt/baseline")
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--image_adapt_weight", type=float, default=0.1)
    parser.add_argument("--image_adapt_layers", type=int, nargs='+', default=[0, 1, 2, 3, 4, 5])
    parser.add_argument("--test_epochs", type=int, nargs='+', default=None)
    args = parser.parse_args()

    setup_seed(args.seed)
    os.makedirs(args.save_path, exist_ok=True)
    logger = logging.getLogger(__name__)
    logging.basicConfig(
        filename=os.path.join(args.save_path, "test.log"),
        encoding="utf-8",
        level=logging.INFO,
    )

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")

    clip_model = create_model(
        model_name=args.model_name,
        img_size=args.img_size,
        device=device,
        pretrained="openai",
        require_pretrained=True,
    )
    clip_model.eval()

    # Updated Adapter Init
    model = AdaptedCLIP(
        clip_model=clip_model,
        image_adapt_weight=args.image_adapt_weight,
        image_adapt_layers=args.image_adapt_layers,
        relu=args.relu,
    ).to(device)
    model.eval()

    dataset_ckpt_path = os.path.join(args.save_path, f"{args.dataset}.pth")

    if os.path.exists(dataset_ckpt_path):
        files = [dataset_ckpt_path]
        logger.info(f"Found dataset-specific checkpoint: {dataset_ckpt_path}")
    else:
        files = sorted(glob(args.save_path + "/image_adapter_*.pth"),
                       key=lambda x: int(x.split('_')[-1].split('.')[0]), reverse=False)

        if args.test_epochs is not None:
            filtered_files = []
            for file in files:
                try:
                    epoch_num = int(file.split('_')[-1].split('.')[0])
                    if epoch_num in args.test_epochs:
                        filtered_files.append(file)
                except ValueError:
                    continue
            if len(filtered_files) == 0:
                raise ValueError("No matching checkpoint files found")
            files = filtered_files

    assert len(files) > 0, "image adapter checkpoint not found"

    for file in files:
        checkpoint = torch.load(file, map_location='cuda:0')
        model.image_adapter.load_state_dict(checkpoint["image_adapter"])

        test_epoch = checkpoint.get("epoch", "unknown")
        logger.info(f"load model from epoch {test_epoch}")

        kwargs = {"num_workers": 10, "pin_memory": True} if use_cuda else {}
        image_datasets = get_dataset(
            args.dataset,
            args.img_size,
            None,
            args.shot,
            "test",
            logger=logger,
        )

        with torch.no_grad():
            # Always use raw CLIP model for text since we removed text training
            text_embeddings = get_adapted_text_embedding(
                clip_model, args.dataset, device
            )

        df = DataFrame(
            columns=["class name", "pixel AUC", "pixel AP", "image AUC", "image AP", "sum"]
        )

        for class_name, image_dataset in image_datasets.items():
            image_dataloader = torch.utils.data.DataLoader(
                image_dataset, batch_size=args.batch_size, shuffle=False, **kwargs
            )

            with torch.no_grad():
                class_text_embeddings = text_embeddings[class_name]
                masks, labels, preds, preds_image, file_names = get_predictions(
                    model=model,
                    class_text_embeddings=class_text_embeddings,
                    test_loader=image_dataloader,
                    device=device,
                    img_size=args.img_size,
                    dataset=args.dataset,
                )

            class_result_dict = metrics_eval(
                masks, labels, preds, preds_image, class_name, domain=DOMAINS[args.dataset],
            )
            df.loc[len(df)] = Series(class_result_dict)

        numeric_columns = ["pixel AUC", "pixel AP", "image AUC", "image AP"]
        avg_values = df[numeric_columns].mean()
        avg_row = {"class name": "Average"}
        avg_row.update(avg_values.to_dict())
        avg_row["sum"] = avg_values.sum()
        df.loc[len(df)] = avg_row
        logger.info("final results:\n%s", df.to_string(index=False, justify="center"))


if __name__ == "__main__":
    main()