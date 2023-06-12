# LoViT
This work is proposed for online surgical phase recognition task, and the full code is not available now because of the patent and review process. But the extracted features and trained spatial feature extractor weight on Cholec80 and AutoLaparo are vailable now.

The core code of getting transition map is also available.

## The trained weight
The trained temporally-rich spatial feature extractor weight and the extracted features on Cholec80 and AutoLapro are available at:[OneDrive Link](https://emckclac-my.sharepoint.com/:f:/g/personal/k21073807_kcl_ac_uk/EpShcwpjssRGomJdcEhfZ68B6bNAt_WAVKfOrtrUfI-Bgw?e=YPJNpf).

## How to use the trained weight
```
def extracted_spatial_feature(video, TIMM):
  '''
  video shape: B, C, len, h, w #Note that the features are extracted separately
  '''
  feats = TIMM(video_gpu) #B, 768, len, 1, 1
  
TIMM = torch.load('Trained_VIT_Cholec80.pth') # Load weight. We saved the structure of the model in the weight file
```
## How to use the extracted features directly
```
with open(f"DATA/Cholec80/{video_indx}.pkl", 'rb') as f:
  feature= torch.tensor(pickle.load(f)) # 768, len, 1, 1
```
## Cite this work
```
@article{liu2023lovit,
  title={LoViT: Long Video Transformer for Surgical Phase Recognition},
  author={Liu, Yang and Boels, Maxence and Garcia-Peraza-Herrera, Luis C and Vercauteren, Tom and Dasgupta, Prokar and Granados, Alejandro and Ourselin, Sebastien},
  journal={arXiv preprint arXiv:2305.08989},
  year={2023}
}
```
