# TikTokX: Multi-Modal Self-Supervised Learning for Recommendation (MMSSL)

This is an implementation of the potential tiktok recommendation algorithm with 

Tiktok is an advanced multimedia recommender system that fuses the generative modality-aware collaborative self-augmentation and contrastive cross-modality dependency encoding to achieve superior performance compared to existing state-of-the-art multi-model recommenders.

## Installation

```pip install tiktokx```

---

## Usage

To start training and inference:

```bash
cd tiktokx
python main.py --dataset {DATASET}
```
Supported datasets include `Amazon-Baby`, `Amazon-Sports`, `Tiktok`, and `Allrecipes`.

----

## Datasets

Dataset specifications are tabulated below:

| Dataset      | Modality | Embed Dim | User  | Item  | Interactions | Sparsity |
|--------------|:--------:|:---------:|:-----:|:-----:|:------------:|:--------:|
| Amazon       | V  T     | 4096 1024 | 35598 | 18357 | 256308       | 99.961%  |
| Tiktok       | V  A  T  | 128  128  768 | 9319  | 6710  | 59541        | 99.904%  |
| Allrecipes   | V  T     | 2048 20   | 19805 | 10067 | 58922        | 99.970%  |

Datasets can be accessed from [Google Drive](https://drive.google.com/drive/folders/1AB1RsnU-ETmubJgWLpJrXd8TjaK_eTp0?usp=share_link). Note: The official website for the `Tiktok` dataset is no longer available. However, we've processed and made available various versions of the [Tiktok dataset](https://drive.google.com/drive/folders/1hLvoS7F0R_K0HBixuS_OVXw_WbBxnshF?usp=share_link). Kindly cite our work if you utilize our preprocessed Tiktok dataset.

