# Pre_train vgg16 model, a keras implement
Currently，transfer learning, fine-tuning, and mul-task learning are popular in deep learning research and we may see them in many start-of-the-art networks.  

A pre-train model like vgg16 and ResNet, is ususally desired if we use above methods.  

So, in this experiment, I will use a pre-train vgg16 model to predict a cat for practice.


To learn more about vgg16, you can refer to    
《Very Deep Convolutional Networks for Large-Scale Image Recognition》： https://arxiv.org/abs/1409.1556

## Experiment preparation
1、Experiment environment: Ubuntu14.04, tensorflow 1.0.0, keras1.2.2  

2、Download the vgg16_weights and move it to working directory  
(URL:https://docs.google.com/uc?id=0Bz7KyqmuGsilT0J5dmRCM0ROVHc&export=download)  

3、Download a picture that contains a dog, move to working directory and rename it as 'dog.jpg'

## Notation
The pre_train weight we download is based on theano,not tensorflow.  

So i firstly set the image dimension ordering using ```K.set_image_dim_ordering('th')``` in ```vgg16.py``` as that setting is different between tensorflow and theano.

## Vgg structure overview
The vgg16 structure: See column C

![](https://github.com/j-c-zhang/VGG16/blob/master/picture/Vgg.jpg)  


