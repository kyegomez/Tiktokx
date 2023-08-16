[![Multi-Modality](agorabanner.png)](https://discord.gg/qUtxnK2NMf)


# TikTokX: Multi-Modal Recommentation Algorithm

[![GitHub issues](https://img.shields.io/github/issues/kyegomez/tiktokx)](https://github.com/kyegomez/tiktokx/issues) 
[![GitHub forks](https://img.shields.io/github/forks/kyegomez/tiktokx)](https://github.com/kyegomez/tiktokx/network) 
[![GitHub stars](https://img.shields.io/github/stars/kyegomez/tiktokx)](https://github.com/kyegomez/tiktokx/stargazers) [![GitHub license](https://img.shields.io/github/license/kyegomez/tiktokx)](https://github.com/kyegomez/tiktokx/blob/master/LICENSE)
[![Share on Twitter](https://img.shields.io/twitter/url/https/twitter.com/cloudposse.svg?style=social&label=Share%20%40kyegomez/tiktokx)](https://twitter.com/intent/tweet?text=Excited%20to%20introduce%20tiktokx,%20the%20all-new%20robotics%20model%20with%20the%20potential%20to%20revolutionize%20automation.%20Join%20us%20on%20this%20journey%20towards%20a%20smarter%20future.%20%23RT1%20%23Robotics&url=https%3A%2F%2Fgithub.com%2Fkyegomez%2Ftiktokx)
[![Share on Facebook](https://img.shields.io/badge/Share-%20facebook-blue)](https://www.facebook.com/sharer/sharer.php?u=https%3A%2F%2Fgithub.com%2Fkyegomez%2Ftiktokx)
[![Share on LinkedIn](https://img.shields.io/badge/Share-%20linkedin-blue)](https://www.linkedin.com/shareArticle?mini=true&url=https%3A%2F%2Fgithub.com%2Fkyegomez%2Ftiktokx&title=Introducing%20tiktokx%2C%20the%20All-New%20Robotics%20Model&summary=tiktokx%20is%20the%20next-generation%20robotics%20model%20that%20promises%20to%20transform%20industries%20with%20its%20intelligence%20and%20efficiency.%20Join%20us%20to%20be%20a%20part%20of%20this%20revolutionary%20journey%20%23RT1%20%23Robotics&source=)
![Discord](https://img.shields.io/discord/999382051935506503)
[![Share on Reddit](https://img.shields.io/badge/-Share%20on%20Reddit-orange)](https://www.reddit.com/submit?url=https%3A%2F%2Fgithub.com%2Fkyegomez%2Ftiktokx&title=Exciting%20Times%20Ahead%20with%20tiktokx%2C%20the%20All-New%20Robotics%20Model%20%23RT1%20%23Robotics) [![Share on Hacker News](https://img.shields.io/badge/-Share%20on%20Hacker%20News-orange)](https://news.ycombinator.com/submitlink?u=https%3A%2F%2Fgithub.com%2Fkyegomez%2Ftiktokx&t=Exciting%20Times%20Ahead%20with%20tiktokx%2C%20the%20All-New%20Robotics%20Model%20%23RT1%20%23Robotics)
[![Share on Pinterest](https://img.shields.io/badge/-Share%20on%20Pinterest-red)](https://pinterest.com/pin/create/button/?url=https%3A%2F%2Fgithub.com%2Fkyegomez%2Ftiktokx&media=https%3A%2F%2Fexample.com%2Fimage.jpg&description=tiktokx%2C%20the%20Revolutionary%20Robotics%20Model%20that%20will%20Change%20the%20Way%20We%20Work%20%23RT1%20%23Robotics)
[![Share on WhatsApp](https://img.shields.io/badge/-Share%20on%20WhatsApp-green)](https://api.whatsapp.com/send?text=I%20just%20discovered%20tiktokx,%20the%20all-new%20robotics%20model%20that%20promises%20to%20revolutionize%20automation.%20Join%20me%20on%20this%20exciting%20journey%20towards%20a%20smarter%20future.%20%23RT1%20%23Robotics%0A%0Ahttps%3A%2F%2Fgithub.com%2Fkyegomez%2Ftiktokx)




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

