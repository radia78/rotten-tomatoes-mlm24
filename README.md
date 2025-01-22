# Infected Tomato Leaf Vein Segmentation
The project is an extension of the [Plant Pathology Image Processor](https://dsi.wisc.edu/research-portfolio/plant-pathology-image-processor/) project from the Data Science Institute at UW-Madison. The goal of the project is to create a vein-segmentation model that is trained from a handful of hand-annotated training set. Our model managed to achieve a test Jaccard index of 42.6% (the highest in the Kaggle competition).
## Contents
- [Methods](https://github.com/radia78/rotten-tomatoes-mlm24/blob/main/README.md#methods)
- [Results](https://github.com/radia78/rotten-tomatoes-mlm24/blob/main/README.md#results)
- [Conclusion](https://github.com/radia78/rotten-tomatoes-mlm24/blob/main/README.md#conclusion)
  
## Methods
Our team modified the U-Net architecture by appending a smaller U-Net that takes the base segmentation logits and outputs a "refined" segmentation logit. The final prediction of the model concatenated model is "refined" added with the "unrefined" logits. In addition, we used a ResNET34 backbone pretrained on the Imagenet dataset. We trained our model using the AdamW optimizer with a Cosine schedule.

## Results
Our model did best lol.

## Conclusion
We need to try a barlows twin method, refinement through diffusion, and finally recurrent refinement.

## References
