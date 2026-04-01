# magnetic-tile-defect-segmentation
This repository contains an implementation for deep learning based semantic segmentation of defects from images of magnetic tile surfaces. Semantic segmentation is the task of classifying each pixel of an image to a specific class. In this project pixels are classified to two classes, defect and background, making the task binary segmentation. The dataset contains images of five different defect types, which are blowhole, crack, break, fray, and uneven. It is available for example in [here](https://www.kaggle.com/datasets/alex000kim/magnetic-tile-surface-defects). This dataset was first used by Huang et al. \[1\].

# Setup
In this project the implemented neural network is based on the U-Net, which is originally developed for biomedical image segmentation \[2\]. In my implementation the dimensions stay the same when going from one convolutional layer to the other, so only max pooling layers reduce dimensions. Additionally, the sizes of convolutional layers and input dimensions are smaller compared to the original U-Net. More details in the comments in 'unet.py'. <br />
All the defect categories except 'uneven' were used for training and testing of the model. The 'uneven' category was left out as I believe it doesn't represent a damaged metal surface in the same way as the other categories. It is more of a manufacturing defect. Additionally, 80 images were sampled randomly from 'free' category and used in training and validation.

# Results

# References
\[1\] Huang, Y., Qiu, C., Guo, Y., Wang, X., & Yuan, K. (2018). Surface Defect Saliency of Magnetic Tile. 2018 IEEE 14th International Conference on Automation Science and Engineering (CASE). doi:10.1109/coase.2018.8560423
\[2\] Ronneberger, O., Fischer, P., Brox, T. (2015). U-Net: Convolutional Networks for Biomedical Image Segmentation. Proceedings of the International Conference on Medical image computing and computer-assisted intervention, pp. 234–241.
