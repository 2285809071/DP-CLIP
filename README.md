# DP-CLIP
DP-CLIP Implementation
## Abstract
Vision-Language Models have demonstrated immense potential in the field of Zero-Shot Anomaly Detection (ZSAD). However, their inherent semantic bias leads to a limited ability to distinguish between anomalous and normal features. To address this challenge, we propose DP-CLIP, a novel framework that leverages Feature Decoupling and Channel Purification to learn domain-agnostic discriminative representations. To explicitly separate normal and anomalous patterns, we introduce a momentum-guided feature decoupling mechanism. By utilizing the momentum mechanism to aggregate global prototypes, image features are forced to decouple from opposing prototypes in the latent space, thereby enlarging the distributional discrepancy between normal and abnormal instances. Furthermore, to enhance feature distinctiveness, we propose improved adapters and projection layers. These components effectively suppress redundant channel activations and refine feature representations while fine-tuning the original features, enabling the model to focus on anomaly-sensitive cues. Extensive experiments across multiple industrial and medical datasets demonstrate that DP-CLIP significantly outperforms existing methods, achieving superior generalization performance and detection accuracy.

## Quick Start 
### 1. Installation  
```bash
conda create -n aaclip python=3.10 -y  
conda activate aaclip  
pip install -r requirements.txt  
```
### 2. Datasets
The datasets can be downloaded from [MVTec-AD](https://www.mvtec.com/company/research/datasets/mvtec-ad/), [VisA](https://github.com/amazon-science/spot-diff), [MPDD](https://github.com/stepanje/MPDD), [BrainMRI, LiverCT, Retinafrom](https://drive.google.com/drive/folders/1La5H_3tqWioPmGN04DM1vdl3rbcBez62?usp=sharing) from [BMAD](https://github.com/DorisBao/BMAD), [CVC-ColonDB, CVC-ClinicDB, Kvasir, CVC-300](https://figshare.com/articles/figure/Polyp_DataSet_zip/21221579) from Polyp Dataset.

Put all the datasets under ``./data`` and use jsonl files in ``./dataset/metadata/``. You can use your own dataset and generate personalized jsonl files with below format:
```json
{"image_path": "xxxx/xxxx/xxx.png", 
 "label": 1.0 (for anomaly) # or 0.0 (for normal), 
 "class_name": "xxxx", 
 "mask_path": "xxxx/xxxx/xxx.png"}
```
The way of creating corresponding jsonl file differs, depending on the file structure of the original dataset. The basic logic is recording the path of every image file, mask file, anomaly label ``label``, and category name, then putting them together under a jsonl file.

(Optional) the base data directory can be edited in ``./dataset/constants``. If you want to reproduce the results with few-shot training, you can generate corresponding jsonl files and put them in ``./dataset/metadata/{$dataset}`` with ``{$shot}-shot.jsonl`` as the file name. For few-shot training, we use ``$shot`` samples from each category to train the model.

> Notice: Since the anomaly scenarios in VisA are closer to real situations, the default hyper-parameters are set according to the results trained on VisA. More analysis and discussion will be available.

### 3. Training & Evaluation
```bash
# evaluation
python test.py --save_path $save_path --dataset $dataset
```
Model definition is in ``./model/``. We thank [```open_clip```](https://github.com/mlfoundations/open_clip.git) for being open-source. To run the code, one has to download the weight of OpenCLIP ViT-L-14-336px and put it under ```./model/```.
