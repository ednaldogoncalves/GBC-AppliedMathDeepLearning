# Mid-Term Project: Image Classification Using CNN

## Overview
In this exercise, you will learn to use image dataset for classification.

## Objectives
- Build image classification CNN using python on the ImageNet datasets.
- CNNs work for the image classification task
- Using TensorFlow’s Keras library to demonstrate image classification using CNNs

## Dataset

There are various datasets that you can leverage for applying convolutional neural networks. Here are three popular datasets:

- MNIST
- CIFAR-10
- ImageNet

## Libraries

The algorithm described in Python 3.11.6 will be implemented, using the Keras library (on top of the TensorFlow libraries, which works as a backend). Allows the use of a GPU, if present, to accelerate the computation, which is recommended in the case of this type of networks, since their training is computationally heavy.

## Download the Imagenette dataset

Once you have downloaded the dataset, you will notice that it has two folders – “train” and “val”. These contain the training and validation set, respectively. Inside each folder, there are separate folders for each class.

`!wget https://s3.amazonaws.com/fast-ai-imageclas/imagenette2.tgz`

`!tar -xf imagenette2.tgz`

It was not added to GitHub due to the size of the directory.

## The Project

The file: [image_classification_cnn_imagenet_dataset.ipynb](https://github.com/ednaldogoncalves/GBC-AppliedMathDeepLearning/blob/main/mid_term_project/Image_Classification_CNN_ImageNet_Dataset.ipynb)

## The Models

The **.h5** models generated after execution were also not included in GitHub due to the size of the directory, but they can be downloaded from the links below:

- [cnn_model.h5](https://drive.google.com/file/d/1kqNxrpReP-LI75G9mq7ePZPz-4mUr_ON/view?usp=sharing)
- [fcnn_inception_model.h5](https://drive.google.com/file/d/1k4KPyySb8_tjYW8yNSozmuLzLkcpBPW9/view?usp=sharing)
- [fcnn_efficientNet_model.h5](https://drive.google.com/file/d/1kJ9F5JuRki_DV6Vdc9J0662iHs9HcvLL/view?usp=sharing)
- [fcnn_restnet_model.h5](https://drive.google.com/file/d/1kGM97e87u_f_y4PjFeND8xCTbO_Gh9fI/view?usp=sharing)
- [fcnn_vgg_model.h5](https://drive.google.com/file/d/1kaf_c6msmuIz1TbZTOZNgNZRNdFdV7uM/view?usp=sharing)
