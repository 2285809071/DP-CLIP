import numpy as np
import cv2
import os
import torch
import torch.nn as nn
from torch.nn import functional as F
from tqdm import tqdm
from kornia.filters import gaussian_blur2d
import ipdb
from dataset.constants import CLASS_NAMES, REAL_NAMES, PROMPTS
from model.tokenizer import tokenize
import pandas as pd
from dataset.constants import DATA_PATH
from utils import cos_sim
from numpy import ndarray
from skimage import measure
from sklearn.metrics import auc
from statistics import mean
from sklearn.metrics import roc_auc_score, precision_recall_curve, average_precision_score
import seaborn as sns
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
# ================================================================================================
# The following code is used to get criterion for training
def return_best_thr(y_true, y_score):
    precs, recs, thrs = precision_recall_curve(y_true, y_score)

    f1s = 2 * precs * recs / (precs + recs + 1e-7)
    f1s = f1s[:-1]
    thrs = thrs[~np.isnan(f1s)]
    f1s = f1s[~np.isnan(f1s)]
    best_thr = thrs[np.argmax(f1s)]
    return best_thr


def f1_score_max(y_true, y_score):
    precs, recs, thrs = precision_recall_curve(y_true, y_score)

    f1s = 2 * precs * recs / (precs + recs + 1e-7)
    f1s = f1s[:-1]
    return f1s.max()

def compute_pro(masks: ndarray, amaps: ndarray, num_th: int = 200) -> None:
    """Compute the area under the curve of per-region overlaping (PRO) and 0 to 0.3 FPR"""

    assert isinstance(amaps, ndarray), "type(amaps) must be ndarray"
    assert isinstance(masks, ndarray), "type(masks) must be ndarray"
    assert amaps.ndim == 3, "amaps.ndim must be 3 (num_test_data, h, w)"
    assert masks.ndim == 3, "masks.ndim must be 3 (num_test_data, h, w)"
    assert amaps.shape == masks.shape, "amaps.shape and masks.shape must be same"
    assert set(masks.flatten()) == {0, 1}, "set(masks.flatten()) must be {0, 1}"
    assert isinstance(num_th, int), "type(num_th) must be int"

    df = pd.DataFrame([], columns=["pro", "fpr", "threshold"])
    binary_amaps = np.zeros_like(amaps, dtype=np.bool_)

    min_th = amaps.min()
    max_th = amaps.max()
    delta = (max_th - min_th) / num_th

    for th in np.arange(min_th, max_th, delta):
        binary_amaps[amaps <= th] = 0
        binary_amaps[amaps > th] = 1

        pros = []
        for binary_amap, mask in zip(binary_amaps, masks):
            for region in measure.regionprops(measure.label(mask)):
                axes0_ids = region.coords[:, 0]
                axes1_ids = region.coords[:, 1]
                tp_pixels = binary_amap[axes0_ids, axes1_ids].sum()
                pros.append(tp_pixels / region.area)

        inverse_masks = 1 - masks
        fp_pixels = np.logical_and(inverse_masks, binary_amaps).sum()
        fpr = fp_pixels / inverse_masks.sum()

        new_row = pd.DataFrame([{"pro": mean(pros), "fpr": fpr, "threshold": th}])
        df = pd.concat([df, new_row], ignore_index=True)

    # Normalize FPR from 0 ~ 1 to 0 ~ 0.3
    df = df[df["fpr"] < 0.3]
    df["fpr"] = df["fpr"] / df["fpr"].max()

    pro_auc = auc(df["fpr"], df["pro"])
    return pro_auc

class FocalLoss(nn.Module):
    def __init__(
        self,
        apply_nonlin=None,
        alpha=None,
        gamma=2,
        balance_index=0,
        smooth=1e-5,
        size_average=True,
    ):
        super(FocalLoss, self).__init__()
        self.apply_nonlin = apply_nonlin
        self.alpha = alpha
        self.gamma = gamma
        self.balance_index = balance_index
        self.smooth = smooth
        self.size_average = size_average

        if self.smooth is not None:
            if self.smooth < 0 or self.smooth > 1.0:
                raise ValueError("smooth value should be in [0,1]")

    def forward(self, logit, target):
        if self.apply_nonlin is not None:
            logit = self.apply_nonlin(logit)
        num_class = logit.shape[1]

        if logit.dim() > 2:
            logit = logit.view(logit.size(0), logit.size(1), -1)
            logit = logit.permute(0, 2, 1).contiguous()
            logit = logit.view(-1, logit.size(-1))
        target = torch.squeeze(target, 1)
        target = target.view(-1, 1)
        alpha = self.alpha

        if alpha is None:
            alpha = torch.ones(num_class, 1)
        elif isinstance(alpha, (list, np.ndarray)):
            assert len(alpha) == num_class
            alpha = torch.FloatTensor(alpha).view(num_class, 1)
            alpha = alpha / alpha.sum()
        elif isinstance(alpha, float):
            alpha = torch.ones(num_class, 1)
            alpha = alpha * (1 - self.alpha)
            alpha[self.balance_index] = self.alpha
        else:
            raise TypeError("Not support alpha type")

        if alpha.device != logit.device:
            alpha = alpha.to(logit.device)

        idx = target.cpu().long()

        one_hot_key = torch.FloatTensor(target.size(0), num_class).zero_()
        one_hot_key = one_hot_key.scatter_(1, idx, 1)
        if one_hot_key.device != logit.device:
            one_hot_key = one_hot_key.to(logit.device)

        if self.smooth:
            one_hot_key = torch.clamp(
                one_hot_key, self.smooth / (num_class - 1), 1.0 - self.smooth
            )
        pt = (one_hot_key * logit).sum(1) + self.smooth
        logpt = pt.log()

        gamma = self.gamma

        alpha = alpha[idx]
        alpha = torch.squeeze(alpha)
        loss = -1 * alpha * torch.pow((1 - pt), gamma) * logpt

        if self.size_average:
            loss = loss.mean()
        return loss


class BinaryDiceLoss(nn.Module):
    def __init__(self):
        super(BinaryDiceLoss, self).__init__()

    def forward(self, input, targets):
        N = targets.size()[0]
        smooth = 1
        input_flat = input.view(N, -1)
        targets_flat = targets.view(N, -1)
        intersection = input_flat * targets_flat
        N_dice_eff = (2 * intersection.sum(1) + smooth) / (
            input_flat.sum(1) + targets_flat.sum(1) + smooth
        )
        loss = 1 - N_dice_eff.sum() / N
        return loss


# ================================================================================================
# The following code is used to get adapted text embeddings
prompt = PROMPTS
prompt_normal = prompt["prompt_normal"]
prompt_abnormal = prompt["prompt_abnormal"]
prompt_state = [prompt_normal, prompt_abnormal]
prompt_templates = prompt["prompt_templates"]


def get_adapted_single_class_text_embedding(model, dataset_name, class_name, device):
    if class_name == "object":
        real_name = class_name
    else:
        assert class_name in CLASS_NAMES[dataset_name], (
            f"class_name {class_name} not found; available class_names: {CLASS_NAMES[dataset_name]}"
        )
        real_name = REAL_NAMES[dataset_name][class_name]
    text_features = []
    for i in range(len(prompt_state)):
        prompted_state = [state.format(real_name) for state in prompt_state[i]]
        prompted_sentence = []
        for s in prompted_state:
            for template in prompt_templates:
                prompted_sentence.append(template.format(s))
        prompted_sentence = tokenize(prompted_sentence).to(device)
        class_embeddings = model.encode_text(prompted_sentence)
        class_embeddings = class_embeddings / class_embeddings.norm(
            dim=-1, keepdim=True
        )
        class_embedding = class_embeddings.mean(dim=0)
        class_embedding = class_embedding / class_embedding.norm()
        text_features.append(class_embedding)
    text_features = torch.stack(text_features, dim=1).to(device)
    return text_features


def get_adapted_single_sentence_text_embedding(model, dataset_name, class_name, device):
    assert class_name in CLASS_NAMES[dataset_name], (
        f"class_name {class_name} not found; available class_names: {CLASS_NAMES[dataset_name]}"
    )
    real_name = REAL_NAMES[dataset_name][class_name]
    text_features = []
    for i in range(len(prompt_state)):
        prompted_state = [state.format(real_name) for state in prompt_state[i]]
        prompted_sentence = []
        for s in prompted_state:
            for template in prompt_templates:
                prompted_sentence.append(template.format(s))
        prompted_sentence = tokenize(prompted_sentence).to(device)
        class_embeddings = model.encode_text(prompted_sentence)
        class_embeddings = F.normalize(class_embeddings, dim=-1)
        text_features.append(class_embeddings)
    text_features = torch.cat(text_features, dim=0).to(device)
    return text_features


def get_adapted_text_embedding(model, dataset_name, device):
    ret_dict = {}
    for class_name in CLASS_NAMES[dataset_name]:
        text_features = get_adapted_single_class_text_embedding(
            model, dataset_name, class_name, device
        )
        ret_dict[class_name] = text_features
    return ret_dict


# ================================================================================================
def calculate_similarity_map(
    patch_features, epoch_text_feature, img_size, test=False, domain="Medical"
):
    patch_anomaly_scores = 100.0 * torch.matmul(patch_features, epoch_text_feature)
    B, L, C = patch_anomaly_scores.shape
    H = int(np.sqrt(L))
    patch_pred = patch_anomaly_scores.permute(0, 2, 1).view(B, C, H, H)
    if test:
        assert C == 2
        sigma = 1 if domain == "Industrial" else 1.5
        kernel_size = 7 if domain == "Industrial" else 9
        patch_pred = (patch_pred[:, 1] + 1 - patch_pred[:, 0]) / 2
        patch_pred = gaussian_blur2d(
            patch_pred.unsqueeze(1), (kernel_size, kernel_size), (sigma, sigma)
        )
    patch_preds = F.interpolate(
        patch_pred, size=img_size, mode="bilinear", align_corners=True
    )
    if not test and C > 1:
        patch_preds = torch.softmax(patch_preds, dim=1)
    return patch_preds


focal_loss = FocalLoss()
dice_loss = BinaryDiceLoss()


def calculate_seg_loss(patch_preds, mask):
    loss = focal_loss(patch_preds, mask)
    loss += dice_loss(patch_preds[:, 0, :, :], 1 - mask)
    loss += dice_loss(patch_preds[:, 1, :, :], mask)
    return loss


# ================================================================================================
def metrics_eval_only_image(
    pixel_label: np.ndarray,
    image_label: np.ndarray,
    pixel_preds: np.ndarray,
    image_preds: np.ndarray,
    class_names: str,
    domain: str,
):
    if pixel_preds.max() != 1:
        pixel_preds = (pixel_preds - pixel_preds.min()) / (
            pixel_preds.max() - pixel_preds.min()
        )
    pmax_pred = pixel_preds.max(axis=(1, 2))
 
    image_preds = pmax_pred


    # image level auc & ap
    if image_label.max() != image_label.min():
        image_label = image_label.flatten()
        agg_image_preds = image_preds.flatten()
        agg_image_auc = roc_auc_score(image_label, agg_image_preds)
        agg_image_ap = average_precision_score(image_label, agg_image_preds)
    else:
        agg_image_auc = 0
        agg_image_ap = 0
    # ================================================================================================
    result = {
        "class name": class_names,
        "image AUC": round(agg_image_auc, 4) * 100,
        "image AP": round(agg_image_ap, 4) * 100,
    }
    return result



def metrics_eval_image(
    pixel_label: np.ndarray,
    image_label: np.ndarray,
    pixel_preds: np.ndarray,
    image_preds: np.ndarray,
    class_names: str,
    domain: str,
):
    if pixel_preds.max() != 1:
        pixel_preds = (pixel_preds - pixel_preds.min()) / (
            pixel_preds.max() - pixel_preds.min()
        )
    if image_preds.max() != 1:
        image_preds = (image_preds - image_preds.min()) / (
            image_preds.max() - image_preds.min()
        )

    pmax_pred = pixel_preds.max(axis=(1, 2))
    image_preds = pmax_pred * 0.5 + image_preds * 0.5


    # image level auc & ap
    if image_label.max() != image_label.min():
        image_label = image_label.flatten()
        agg_image_preds = image_preds.flatten()
        agg_image_auc = roc_auc_score(image_label, agg_image_preds)
        agg_image_ap = average_precision_score(image_label, agg_image_preds)
    else:
        agg_image_auc = 0
        agg_image_ap = 0
    # ================================================================================================
    result = {
        "class name": class_names,
        "image AUC": round(agg_image_auc, 4) * 100,
        "image AP": round(agg_image_ap, 4) * 100,
    }
    return result

def metrics_eval(
    pixel_label: np.ndarray,
    image_label: np.ndarray,
    pixel_preds: np.ndarray,
    image_preds: np.ndarray,
    class_names: str,
    domain: str,
):
    if pixel_preds.max() != 1:
        pixel_preds = (pixel_preds - pixel_preds.min()) / (
            pixel_preds.max() - pixel_preds.min()
        )
    if image_preds.max() != 1:
        image_preds = (image_preds - image_preds.min()) / (
            image_preds.max() - image_preds.min()
        )

    pmax_pred = pixel_preds.max(axis=(1, 2))
    if domain != "Medical":
        image_preds = pmax_pred * 0.5 + image_preds * 0.5
    else:
        image_preds = pmax_pred
    # ================================================================================================
    # pixel level auc & ap
    pixel_label = pixel_label.flatten()
    pixel_preds = pixel_preds.flatten()

    zero_pixel_auc = roc_auc_score(pixel_label, pixel_preds)
    zero_pixel_ap = average_precision_score(pixel_label, pixel_preds)
    # ================================================================================================
    # image level auc & ap
    if image_label.max() != image_label.min():
        image_label = image_label.flatten()
        agg_image_preds = image_preds.flatten()
        agg_image_auc = roc_auc_score(image_label, agg_image_preds)
        agg_image_ap = average_precision_score(image_label, agg_image_preds)
    else:
        agg_image_auc = 0
        agg_image_ap = 0
    # ================================================================================================
    result = {
        "class name": class_names,
        "pixel AUC": round(zero_pixel_auc, 4) * 100,
        "pixel AP": round(zero_pixel_ap, 4) * 100,
        "image AUC": round(agg_image_auc, 4) * 100,
        "image AP": round(agg_image_ap, 4) * 100,
    }
    return result


def metrics_eval_all(
    pixel_label: np.ndarray,
    image_label: np.ndarray,
    pixel_preds: np.ndarray,
    image_preds: np.ndarray,
    class_names: str,
    domain: str,
):
    if pixel_preds.max() != 1:
        pixel_preds = (pixel_preds - pixel_preds.min()) / (
            pixel_preds.max() - pixel_preds.min()
        )
    if image_preds.max() != 1:
        image_preds = (image_preds - image_preds.min()) / (
            image_preds.max() - image_preds.min()
        )
        
    pmax_pred = pixel_preds.max(axis=(1, 2))
    if domain != "Medical":
        image_preds = pmax_pred * 0.5 + image_preds * 0.5
    else:
        image_preds = pmax_pred
    # ================================================================================================
    # pixel level auc & ap
    pixel_aupro = compute_pro(pixel_label.squeeze(), pixel_preds.squeeze())
    
    pixel_label = pixel_label.flatten()
    pixel_preds = pixel_preds.flatten()
    
    pixel_f1_max = f1_score_max(pixel_label, pixel_preds)
    zero_pixel_auc = roc_auc_score(pixel_label, pixel_preds)
    zero_pixel_ap = average_precision_score(pixel_label, pixel_preds)
    # ================================================================================================
    # image level auc & ap
    if image_label.max() != image_label.min():
        image_label = image_label.flatten()
        agg_image_preds = image_preds.flatten()
        agg_image_auc = roc_auc_score(image_label, agg_image_preds)
        agg_image_ap = average_precision_score(image_label, agg_image_preds)
        image_f1_max = f1_score_max(image_label, image_preds)
    else:
        agg_image_auc = 0
        agg_image_ap = 0
        image_f1_max =0
    # ================================================================================================
    result = {
        "class name": class_names,
        "pixel AUC": round(zero_pixel_auc, 4) * 100,
        "pixel AP": round(zero_pixel_ap, 4) * 100,
        "pixel F1 max": round(pixel_f1_max, 4)*100,   
        "pixel AUPRO": round(pixel_aupro, 4)*100,
        "image AUC": round(agg_image_auc, 4) * 100,
        "image AP": round(agg_image_ap, 4) * 100,
        "image F1 max": round(image_f1_max, 4)*100,   
    }
    return result


def normalize(pred, max_value=None, min_value=None):
    if max_value is None or min_value is None:
        return (pred - pred.min()) / (pred.max() - pred.min())
    else:
        return (pred - min_value) / (max_value - min_value)


def apply_ad_scoremap(image, scoremap, alpha=0.5):
    np_image = np.asarray(image, dtype=float)
    scoremap = (scoremap * 255).astype(np.uint8)
    scoremap = cv2.applyColorMap(scoremap, cv2.COLORMAP_JET)
    scoremap = cv2.cvtColor(scoremap, cv2.COLOR_BGR2RGB)
    return (alpha * np_image + (1 - alpha) * scoremap).astype(np.uint8)


import numpy as np
import cv2
import os
import torch
import torch.nn as nn
from torch.nn import functional as F
from tqdm import tqdm
from kornia.filters import gaussian_blur2d
import ipdb
from dataset.constants import CLASS_NAMES, REAL_NAMES, PROMPTS
from model.tokenizer import tokenize
import pandas as pd
from dataset.constants import DATA_PATH
from utils import cos_sim
from numpy import ndarray
from skimage import measure
from sklearn.metrics import auc
from statistics import mean
from sklearn.metrics import roc_auc_score, precision_recall_curve, average_precision_score

# ... (Previous helper functions like return_best_thr, compute_pro, losses, text embeddings, etc. remain unchanged) ...

def normalize(pred, max_value=None, min_value=None):
    if max_value is None or min_value is None:
        return (pred - pred.min()) / (pred.max() - pred.min())
    else:
        return (pred - min_value) / (max_value - min_value)


def apply_ad_scoremap(image, scoremap, alpha=0.5):
    np_image = np.asarray(image, dtype=float)
    scoremap = (scoremap * 255).astype(np.uint8)
    scoremap = cv2.applyColorMap(scoremap, cv2.COLORMAP_JET)
    scoremap = cv2.cvtColor(scoremap, cv2.COLOR_BGR2RGB)
    return (alpha * np_image + (1 - alpha) * scoremap).astype(np.uint8)


def visualization(results, obj_list, img_size, save_path):
    for obj in obj_list:
        paths = []
        pr_px = []
        gt_px = [] # container for masks
        cls_names = []
        
        # Filter data for the current object class
        for idxes in range(len(results['cls_names'])):
            if results['cls_names'][idxes] == obj:
                paths.append(results['img_path'][idxes])
                pr_px.append(results['pr_px'][idxes])
                # Check if GT exists in results
                if 'gt_px' in results and len(results['gt_px']) > idxes:
                    gt_px.append(results['gt_px'][idxes])
                cls_names.append(results['cls_names'][idxes])
        
        if not pr_px:
            continue
            
        pr_px = normalize(np.array(pr_px))
        
        # Iterate over filtered images
        for idx in range(len(paths)):
            path = paths[idx]
            cls = path.split('/')[-2]
            filename = path.split('/')[-1]
            fname_no_ext = os.path.splitext(filename)[0]
            
            # Read image
            vis_original = cv2.imread(path)
            if vis_original is None:
                print(f"Warning: Could not read image {path}")
                continue
            
            # Resize and convert to RGB for consistency (heatmap generation needs RGB)
            vis_original = cv2.resize(vis_original, (img_size, img_size))
            vis_rgb = cv2.cvtColor(vis_original, cv2.COLOR_BGR2RGB)
            
            # 1. Prepare Heatmap (Prediction)
            mask = normalize(pr_px[idx])
            vis_heatmap = apply_ad_scoremap(vis_rgb, mask) 
            vis_heatmap = cv2.cvtColor(vis_heatmap, cv2.COLOR_RGB2BGR) # Back to BGR for saving
            
            # =======================================================
            # 2. Prepare Ground Truth (Overlay Red Blocks on Original)
            # =======================================================
            vis_gt = vis_original.copy() # Start with the original BGR image
            
            if gt_px:
                gt_mask = gt_px[idx] # (H, W) or (1, H, W)
                if gt_mask.ndim == 3:
                    gt_mask = gt_mask.squeeze()
                
                # Resize GT to match image size
                gt_mask = cv2.resize(
                    gt_mask.astype(float), 
                    (img_size, img_size), 
                    interpolation=cv2.INTER_NEAREST
                )
                
                # Define anomaly regions (where mask > 0)
                anomaly_indices = gt_mask > 0
                
                if anomaly_indices.any():
                    # Define Red color in BGR
                    red_color = np.array([0, 0, 255]) 
                    alpha = 1 # Transparency factor (0.5 means 50% original, 50% red)
                    
                    # Apply Alpha Blending only at anomaly regions
                    # Formula: Output = Original * (1 - alpha) + Color * alpha
                    
                    # Extract the region of interest (ROI) from original image
                    roi = vis_gt[anomaly_indices]
                    
                    # Perform blending
                    blended_roi = (roi.astype(float) * (1 - alpha) + red_color * alpha).astype(np.uint8)
                    
                    # Put blended pixels back into the image
                    vis_gt[anomaly_indices] = blended_roi
            
            # =======================================================
            
            # Ensure save directory exists
            save_vis = os.path.join(save_path, 'imgs', cls_names[idx], cls)
            os.makedirs(save_vis, exist_ok=True)
            
            # Save all three files
            # 1. Original Image
            cv2.imwrite(os.path.join(save_vis, f"{fname_no_ext}_org.jpg"), vis_original)
            
            # 2. Ground Truth (Now Overlayed)
            cv2.imwrite(os.path.join(save_vis, f"{fname_no_ext}_gt.jpg"), vis_gt)
            
            # 3. Heatmap Visualization
            cv2.imwrite(os.path.join(save_vis, f"{fname_no_ext}_vis.jpg"), vis_heatmap)
            
            
def plot_tsne(features, labels, class_names, save_path, title="T-SNE Visualization"):
    """
    features: (N, D) numpy array
    labels: (N,) numpy array (0 for Normal, 1 for Anomaly)
    class_names: list of strings corresponding to the source dataset/class for each point
    save_path: str, path to save the image
    """
    print(f"Generating T-SNE plot to {save_path}...")
    
    # -----------------------------------------------------------
    # FIX: 动态计算 perplexity 以防止 ValueError
    n_samples = features.shape[0]
    # perplexity 必须小于 n_samples。通常 perplexity 在 5 到 50 之间。
    # 这里我们设置一个逻辑：如果样本少，就用样本数减 1，否则用默认的 30
    target_perplexity = 30
    if n_samples <= target_perplexity:
        perplexity_val = max(1, n_samples - 1)  # 保证至少为 1
        print(f"Warning: n_samples ({n_samples}) <= 30. Adjusting perplexity to {perplexity_val}.")
    else:
        perplexity_val = target_perplexity
    # -----------------------------------------------------------

    # Perform T-SNE
    # MODIFIED: 使用动态计算的 perplexity_val，且移除了 n_iter
    tsne = TSNE(n_components=2, verbose=0, perplexity=perplexity_val, random_state=42)
    tsne_results = tsne.fit_transform(features)

    # Prepare DataFrame for Seaborn
    df_subset = pd.DataFrame()
    df_subset['tsne-2d-one'] = tsne_results[:,0]
    df_subset['tsne-2d-two'] = tsne_results[:,1]
    
    # Convert binary labels to strings
    label_str = []
    for l in labels:
        label_str.append('Normal' if l == 0 else 'Anomaly')
    df_subset['Condition'] = label_str
    df_subset['Class'] = class_names

    plt.figure(figsize=(10, 8))
    
    # Plot Logic
    if len(set(class_names)) > 1:
        # Multiple classes (Medical scenario)
        sns.scatterplot(
            x="tsne-2d-one", y="tsne-2d-two",
            hue="Class",
            style="Condition",
            palette=sns.color_palette("hls", len(set(class_names))),
            data=df_subset,
            legend="full",
            alpha=0.8,
            s=100
        )
    else:
        # Single class (MVTec scenario)
        sns.scatterplot(
            x="tsne-2d-one", y="tsne-2d-two",
            hue="Condition",
            palette={"Normal": "tab:blue", "Anomaly": "tab:red"},
            data=df_subset,
            legend="full",
            alpha=0.8,
            s=100
        )

    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()