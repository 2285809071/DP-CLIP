import os
import argparse
import numpy as np
from tqdm import tqdm
import logging
from glob import glob
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset # [Modified] Added ConcatDataset
from torch.optim.lr_scheduler import MultiStepLR
from utils import setup_seed
from model.adapter import AdaptedCLIP
from model.clip import create_model
from dataset import get_dataset
from forward_utils import (
    get_adapted_text_embedding,
    calculate_similarity_map,
    calculate_seg_loss,
)
import warnings

warnings.filterwarnings("ignore")

# Set CPU thread count
cpu_num = 10
os.environ["OMP_NUM_THREADS"] = str(cpu_num)
os.environ["OPENBLAS_NUM_THREADS"] = str(cpu_num)
os.environ["MKL_NUM_THREADS"] = str(cpu_num)
os.environ["VECLIB_MAXIMUM_THREADS"] = str(cpu_num)
os.environ["NUMEXPR_NUM_THREADS"] = str(cpu_num)
torch.set_num_threads(cpu_num)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

global_vn_proto = None  # Global normal prototype
global_va_proto = None  # Global abnormal prototype


def train_image_adapter(
        model: nn.Module,
        text_embeddings: torch.Tensor,
        train_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        device: str,
        start_epoch: int,
        save_path: str,
        image_epoch: int,
        img_size: int,
        logger: logging.Logger,
        decoupling_weight: float = 0.1,
        momentum_alpha: float = 0.999
):
    print(" momentum_alpha:", momentum_alpha)
    print("decoupling_weight:", decoupling_weight)
    if decoupling_weight <= 0:
        print("Note: decoupling_weight <= 0, skipping decoupling loss calculation.")

    global global_vn_proto
    global global_va_proto

    for epoch in range(start_epoch, image_epoch):
        logger.info(f"training image epoch {epoch + 1}:")
        loss_list = []
        decoupling_loss_list = []

        for input_data in tqdm(train_loader):
            image = input_data["image"].to(device)
            mask = input_data["mask"].to(device)
            label = input_data["label"].to(device)
            B, C, H, W = image.shape

            # --- Text feature extraction (using frozen CLIP features) ---
            class_names = input_data["class_name"]
            epoch_text_feature = torch.stack(
                [text_embeddings[class_name] for class_name in class_names], dim=0
            )

            # --- Forward pass ---
            patch_features, det_feature = model(image)
            loss = 0.0

            # --- 1. Classification loss (BCE/CrossEntropy) ---
            det_feature = det_feature.unsqueeze(1)
            cls_preds = torch.matmul(det_feature, epoch_text_feature)[:, 0]
            loss += F.cross_entropy(cls_preds, label)

            # --- 2. Decoupling loss and segmentation loss ---
            total_decoupling_loss = torch.tensor(0.0, device=device)
            flattened_mask = None
            if decoupling_weight > 0:
                total_decoupling_loss.requires_grad_(True)
                # Resize mask to 37x37 (using nearest neighbor interpolation)
                resized_mask = F.interpolate(mask.float(), size=(37, 37), mode='nearest')
                resized_mask = resized_mask.squeeze(1)
                flattened_mask = resized_mask.view(B, -1)

            # Iterate over each feature layer
            for f in patch_features:
                # f shape: Bx1369x768

                # --- 2.1 & 2.2 Decoupling loss calculation (Conditional) ---
                if decoupling_weight > 0:
                    current_layer_decoupling_loss = 0.0
                    all_abnormal_tokens = []
                    all_normal_tokens = []

                    # Batch-level prototype aggregation
                    for b in range(B):
                        sample_mask = flattened_mask[b]
                        abnormal_indices = (sample_mask > 0.5).nonzero(as_tuple=True)[0]
                        normal_indices = (sample_mask <= 0.5).nonzero(as_tuple=True)[0]

                        if len(abnormal_indices) > 0:
                            all_abnormal_tokens.append(f[b, abnormal_indices, :])
                        if len(normal_indices) > 0:
                            all_normal_tokens.append(f[b, normal_indices, :])

                    if len(all_abnormal_tokens) > 0 and len(all_normal_tokens) > 0:
                        batch_abnormal_tokens = torch.cat(all_abnormal_tokens, dim=0)
                        batch_normal_tokens = torch.cat(all_normal_tokens, dim=0)

                        # Mean then Normalize (current batch prototype)
                        va = torch.mean(batch_abnormal_tokens, dim=0)
                        vn = torch.mean(batch_normal_tokens, dim=0)

                        va_hat = va / va.norm(dim=0, keepdim=True)
                        vn_hat = vn / vn.norm(dim=0, keepdim=True)

                        # Decoupling loss calculation
                        if global_vn_proto is None or global_va_proto is None:
                            # Step 1: Initialize on first run
                            cos_sim = torch.dot(va_hat, vn_hat)
                            current_layer_decoupling_loss = cos_sim ** 2

                            # Initialize global prototypes
                            global_vn_proto = vn_hat.detach().clone()
                            global_va_proto = va_hat.detach().clone()
                        else:
                            # Step 2: Calculate mutual decoupling loss
                            global_cos_sim_a = torch.dot(va_hat, global_vn_proto)
                            global_cos_sim_n = torch.dot(vn_hat, global_va_proto)
                            current_layer_decoupling_loss = (global_cos_sim_a ** 2 + global_cos_sim_n ** 2) / 2.0

                        # Accumulate Loss
                        total_decoupling_loss = total_decoupling_loss + current_layer_decoupling_loss

                # --- 2.3 Segmentation loss (Always Run) ---
                patch_preds = calculate_similarity_map(f, epoch_text_feature, img_size)
                seg_loss = calculate_seg_loss(patch_preds, mask)
                loss += seg_loss

            # --- 3. Total loss summary ---
            if decoupling_weight > 0:
                loss += total_decoupling_loss * decoupling_weight

            decoupling_loss_list.append(total_decoupling_loss.item())

            # --- 4. Optimization ---
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # --- 5. Momentum update global prototypes (Conditional) ---
            if decoupling_weight > 0 and 'va_hat' in locals() and global_vn_proto is not None:
                with torch.no_grad():
                    global_vn_proto.data = momentum_alpha * global_vn_proto.data + (
                            1 - momentum_alpha) * vn_hat.detach()
                    global_va_proto.data = momentum_alpha * global_va_proto.data + (
                            1 - momentum_alpha) * va_hat.detach()

                    # Re-normalize
                    global_vn_proto.data = global_vn_proto.data / global_vn_proto.data.norm(dim=0, keepdim=True)
                    global_va_proto.data = global_va_proto.data / global_va_proto.data.norm(dim=0, keepdim=True)

            loss_list.append(loss.item())
            scheduler.step()

        avg_loss = np.mean(loss_list)
        avg_decoupling_loss = np.mean(decoupling_loss_list)
        logger.info(f"loss: {avg_loss}, decoupling_loss: {avg_decoupling_loss}")

        # Save checkpoint
        model_dict = {
            "epoch": epoch + 1,
            "image_adapter": model.image_adapter.state_dict(),
            "image_optimizer": optimizer.state_dict(),
        }
        torch.save(model_dict, os.path.join(save_path, "image_adapter.pth"))
        if (epoch + 1) % 1 == 0:
            ckp_path = os.path.join(save_path, f"image_adapter_{epoch + 1}.pth")
            torch.save(model_dict, ckp_path)

    return model


def main():
    parser = argparse.ArgumentParser(description="Training")
    # model
    parser.add_argument(
        "--model_name",
        type=str,
        default="ViT-L-14-336",
        help="clip model to use (default: ViT-L-14-336)",
    )
    parser.add_argument("--img_size", type=int, default=518)
    # [Removal] Removed --surgery_until_layer as it was for clip_surgery
    parser.add_argument("--relu", action="store_true", help="use relu after projection")

    # training
    parser.add_argument("--cuda_device", type=str, default="0", help="CUDA device id(s)")
    
    # [Modified] Changed to nargs='+' to accept list of datasets
    parser.add_argument("--dataset", type=str, default=["VisA"], nargs='+', 
                        help="Datasets for training (e.g. VisA mvtec)")
    
    parser.add_argument(
        "--training_mode",
        type=str,
        default="full_shot",
        choices=["few_shot", "full_shot"],
    )
    parser.add_argument("--shot", type=int, default=32, help="number of shots (0 means full shot)")
    parser.add_argument("--image_batch_size", type=int, default=3)
    parser.add_argument("--image_epoch", type=int, default=20, help="epochs for stage2")
    parser.add_argument("--image_lr", type=float, default=0.0005, help="learning rate for stage2")
    parser.add_argument(
        "--criterion", type=str, default=["dice_loss", "focal_loss"], nargs="+"
    )
    # exp
    parser.add_argument("--seed", type=int, default=111)
    parser.add_argument("--save_path", type=str, default="ckpt/baseline")
    # hyper-parameters
    parser.add_argument("--image_adapt_weight", type=float, default=0.1)
    parser.add_argument("--image_adapt_layers", type=int, nargs='+', default=[0, 1, 2, 3, 4, 5])
    parser.add_argument("--decoupling_weight", type=float, default=0.1)
    parser.add_argument("--momentum_alpha", type=float, default=0.999)
    args = parser.parse_args()

    # ========================================================
    setup_seed(args.seed)
    # check save_path and setting logger
    os.makedirs(args.save_path, exist_ok=True)
    logger = logging.getLogger(__name__)
    logging.basicConfig(
        filename=os.path.join(args.save_path, "train.log"),
        encoding="utf-8",
        level=logging.INFO,
    )
    logger.info("args: %s", vars(args))

    # set device
    use_cuda = torch.cuda.is_available()
    device = torch.device(f"cuda:{args.cuda_device}" if use_cuda else "cpu")

    # ========================================================
    # load model
    # set up model for training (This is the one actually used)
    clip_model = create_model(
        model_name=args.model_name,
        img_size=args.img_size,
        device=device,
        pretrained="openai",
        require_pretrained=True,
    )
    clip_model.eval()

    model = AdaptedCLIP(
        clip_model=clip_model,
        image_adapt_weight=args.image_adapt_weight,
        image_adapt_layers=args.image_adapt_layers,
        relu=args.relu,
    ).to(device)
    model.eval()

    # set optimizer
    image_optimizer = torch.optim.Adam(
        model.image_adapter.parameters(),
        lr=args.image_lr,
        betas=(0.5, 0.999),
    )
    image_scheduler = MultiStepLR(image_optimizer, milestones=[16000, 32000], gamma=0.5)

    # ========================================================
    # load checkpoints if exists
    file = glob(args.save_path + "/image_adapter.pth")
    if len(file) > 0:
        checkpoint = torch.load(file[0])
        image_start_epoch = checkpoint["epoch"]
        model.image_adapter.load_state_dict(checkpoint["image_adapter"])
        image_optimizer.load_state_dict(checkpoint["image_optimizer"])
    else:
        image_start_epoch = 0

    # ========================================================
    # load dataset
    if args.training_mode == "full_shot":
        args.shot = -1
    kwargs = {"num_workers": 8, "pin_memory": True} if use_cuda else {}

    logger.info("loading image adaptation dataset ...")
    
    dataset_list = []
    
    datasets_to_load = args.dataset if isinstance(args.dataset, list) else [args.dataset]
    
    for d_name in datasets_to_load:
        logger.info(f"Loading dataset: {d_name}")
        ds_result = get_dataset(
            d_name,
            args.img_size,
            args.training_mode,
            args.shot,
            "train",
            logger,
        )
        # get_dataset returns (text_ds, image_ds), we usually need the image_ds (index 1) for training loop
        dataset_list.append(ds_result[1])

    if len(dataset_list) > 1:
        image_dataset = ConcatDataset(dataset_list)
        logger.info(f"Combined {len(dataset_list)} datasets.")
    else:
        image_dataset = dataset_list[0]

    image_dataloader = torch.utils.data.DataLoader(
        image_dataset, batch_size=args.image_batch_size, shuffle=True, **kwargs
    )

    # ========================================================
    # training
    torch.cuda.empty_cache()

    combined_text_embeddings = {}
    
    with torch.no_grad():
        for d_name in datasets_to_load:
            print(f"Extracting text embeddings for: {d_name}")
            embeddings = get_adapted_text_embedding(
                clip_model, d_name, device
            )
            combined_text_embeddings.update(embeddings)


    model = train_image_adapter(
        model=model,
        text_embeddings=combined_text_embeddings, 
        image_epoch=args.image_epoch,
        train_loader=image_dataloader,
        optimizer=image_optimizer,
        scheduler=image_scheduler,
        device=device,
        start_epoch=image_start_epoch,
        save_path=args.save_path,
        img_size=args.img_size,
        logger=logger,
        decoupling_weight=args.decoupling_weight,
        momentum_alpha=args.momentum_alpha,
    )


if __name__ == "__main__":
    main()
