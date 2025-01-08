# Transformers in Artify


## Introduction
Transformers are a crucial part of Artify's architecture, especially for applying complex style transfer techniques. This document explains how transformers are used, their role in the project, and why they are essential.


## What Are Transformers?
Transformers are state-of-the-art deep learning models initially designed for NLP tasks but have been successfully adapted for other domains like image processing. In Artify, transformers enable efficient style transfer through high-quality feature extraction and style blending.


### Key Features of Transformers
1. **Attention Mechanisms**: Focuses on the most relevant parts of the content and style images.
2. **Parallelism**: Allows faster processing compared to sequential models.
3. **Feature Extraction**: Extracts meaningful representations from both the content and style images.


## Why Transformers in Artify?
Artify relies on the **VGG-19 model**, which acts as a feature extractor, but transformers handle more advanced applications like:
- **Dynamic weight adjustment** during training.
- **Future extensions** for combining multiple styles seamlessly.


## Implementation in Artify

1. **Using Pretrained Models**
   - Artify uses `torchvision`'s **VGG-19 model**, pretrained on the ImageNet dataset, for feature extraction.
   - Layers such as `conv4_2` and `conv1_1` are used to compute the **content** and **style losses** respectively.

   ```python
   from torchvision.models import vgg19

   def _load_pretrained_model(self):
       model = vgg19(pretrained=True).features.eval()
       for param in model.parameters():
           param.requires_grad = False
       return model.to(self.device)
   ```

2. **Feature Extraction**
   - Features are extracted from specific layers to calculate the **content loss** and **style loss**.
   - Artify uses a Gram matrix to compute correlations between features for style transfer.

   ```python
   def _extract_features(self, tensor):
       layers = {
           "0": "conv1_1",
           "5": "conv2_1",
           "10": "conv3_1",
           "19": "conv4_1",
           "21": "conv4_2",
           "28": "conv5_1",
       }
       features = {}
       x = tensor
       for name, layer in self.model._modules.items():
           x = layer(x)
           if name in layers:
               features[layers[name]] = x.clone()
       return features
   ```

3. **Dynamic Loss Weight Adjustment**
   - Transformers ensure that the **style weight** is dynamically scaled during training.
   - This ensures better style representation and faster convergence.

   ```python
   scaled_style_weight = style_weight * (1 + 0.5 * (i / iterations))
   total_loss = scaled_style_weight * style_loss + content_weight * content_loss + tv_weight * tv_loss
   ```
