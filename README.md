# Intel Image Classification using CNN Models

This repository contains the implementation of three Convolutional Neural Network (CNN) models - AlexNet, MobileNetV2, and ResNet50 - for classifying landscape images into six categories: mountain, street, glacier, buildings, sea, and forest. The project utilizes the Intel Image Classification dataset from Kaggle.

## Table of Contents
1. [Introduction](#introduction)
2. [Dataset Description](#dataset-description)
3. [Data Preprocessing](#data-preprocessing)
4. [Model Implementations](#model-implementations)
    - [AlexNet](#alexnet)
    - [MobileNetV2](#mobilenetv2)
    - [ResNet50](#resnet50)
5. [Results and Analysis](#results-and-analysis)
6. [Conclusion](#conclusion)

## Introduction

This project focuses on image classification using deep learning techniques. Three different CNN architectures are implemented and compared to classify landscape images into six distinct categories. The models are built using TensorFlow and trained on the Intel Image Classification dataset.

## Dataset Description

The dataset consists of approximately 24,000 landscape images divided into:
- Training set: ~14,000 images
- Test set: ~3,000 images
- Prediction set: ~7,000 unlabeled images

Images are categorized into six classes:
1. Mountain
2. Street
3. Glacier
4. Buildings
5. Sea
6. Forest

All images are in JPEG format and resized to 150x150 pixels for model input.

## Data Preprocessing

### Data Loading and Preparation
- Images loaded using OpenCV
- Converted from BGR to RGB color space
- Resized to 150x150 pixels
- Normalized to float32 format

### Data Sampling
- Stratified sampling applied to maintain class distribution
- 30% of data sampled from each class
- Data shuffled using sklearn's shuffle function with random_state=25

## Model Implementations

### AlexNet
- Custom implementation of AlexNet architecture
- Five convolutional layers with max pooling
- Three fully connected layers with 4096 neurons each
- Dropout layers for regularization
- Softmax output layer for six-class classification

### MobileNetV2
- Pre-trained MobileNetV2 as base model
- Global Average Pooling and Dropout layers added
- Custom dense layers for six-class classification
- Transfer learning approach with frozen base model weights

### ResNet50
- Pre-trained ResNet50 as base model
- Similar architecture modifications as MobileNetV2
- Global Average Pooling and Dropout layers
- Custom dense layers for six-class classification

## Results and Analysis

### Performance Metrics
- Models evaluated on test set accuracy
- Confusion matrices generated for detailed analysis

### Key Findings
1. **AlexNet**
- Lower accuracy performance
- Significant confusion between classes, especially sea (class 4)
- Misclassification patterns indicate difficulty in distinguishing similar classes

2. **MobileNetV2**
- Best performing model with up to 90% test accuracy
- Excellent performance on class 5 (sea) with 100% accuracy
- Some confusion between:
   - Mountain (class 1) and Glacier (class 3)
   - Buildings (class 4) and Street (class 2)

3. **ResNet50**
- Underfitting observed with only 39% training and 43% test accuracy
- Requires more epochs for proper training
- Significant misclassification issues, particularly with sea and building classes

## Conclusion

The project successfully implemented and compared three CNN architectures for landscape image classification. MobileNetV2 demonstrated the best performance, achieving up to 90% accuracy on the test set. The results highlight the importance of transfer learning and proper model tuning. Future improvements could focus on:
- Increasing training epochs for ResNet50
- Experimenting with different data augmentation techniques
- Fine-tuning pre-trained model layers
- Exploring additional model architectures

## Acknowledgments

This project was supervised by Dr. Hoàng Hữu Trung at the University of Economics, Hue University. Special thanks to the Kaggle community for providing the Intel Image Classification dataset.

## License

[MIT License](LICENSE)
