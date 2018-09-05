# vgg_cifar100
a typical implementation of classification task

This is a typical implementation of vgg16 on cifar100 including data preprocessing based on tf.layers api
(architecture: VGG convolutions + 512 dense + 0.5 dropout)
with more regulariers and better models we may reach higher

the model could be trained from scratch to reach arround 60% test accuracy

data loader should be later extracted to be writen in yield format to allow data loading and training at the same time
