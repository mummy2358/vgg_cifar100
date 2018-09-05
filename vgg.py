import tensorflow as tf
import numpy as np

class vgg_creator:
  def __init__(self,model_name="vgg"):
    with tf.variable_scope(model_name,reuse=tf.AUTO_REUSE) as scope:
      self.model_name=model_name
      self.hwc=(32,32,3)
      self.class_num=100
      self.x=tf.placeholder(tf.float32,shape=[None,self.hwc[0],self.hwc[1],self.hwc[2]])
      self.y=tf.placeholder(tf.float32,shape=[None,self.class_num])
      self.dr_rate=tf.placeholder_with_default(0.5,shape=[])
      self.logits=self.forward(self.x,self.dr_rate)
      self.loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.logits,labels=self.y))
      self.prob=tf.nn.softmax(self.logits,axis=-1)
  
  def forward(self,inputs,dr_rate):
    # forward graph of vgg (fc:512->100)
    conv1_1=tf.layers.conv2d(inputs,filters=64,kernel_size=3,padding="same")
    conv1_1=tf.layers.batch_normalization(conv1_1,training=True)
    conv1_1=tf.nn.relu(conv1_1)
    conv1_1=tf.layers.dropout(conv1_1,rate=dr_rate)
    
    conv1_2=tf.layers.conv2d(conv1_1,filters=64,kernel_size=3,padding="same")
    conv1_2=tf.layers.batch_normalization(conv1_2,training=True)
    conv1_2=tf.nn.relu(conv1_2)
    pool1=tf.layers.max_pooling2d(conv1_2,pool_size=2,strides=2)
    pool1=tf.layers.dropout(pool1,rate=dr_rate)
    
    
    conv2_1=tf.layers.conv2d(pool1,filters=128,kernel_size=3,padding="same")
    conv2_1=tf.layers.batch_normalization(conv2_1,training=True)
    conv2_1=tf.nn.relu(conv2_1)
    conv2_1=tf.layers.dropout(conv2_1,rate=dr_rate)
    
    conv2_2=tf.layers.conv2d(conv2_1,filters=128,kernel_size=3,padding="same")
    conv2_2=tf.layers.batch_normalization(conv2_2,training=True)
    conv2_2=tf.nn.relu(conv2_2)
    pool2=tf.layers.max_pooling2d(conv2_2,pool_size=2,strides=2)
    pool2=tf.layers.dropout(pool2,rate=dr_rate)
    
    
    conv3_1=tf.layers.conv2d(pool2,filters=256,kernel_size=3,padding="same")
    conv3_1=tf.layers.batch_normalization(conv3_1,training=True)
    conv3_1=tf.nn.relu(conv3_1)
    conv3_1=tf.layers.dropout(conv3_1,rate=dr_rate)
    
    conv3_2=tf.layers.conv2d(conv3_1,filters=256,kernel_size=3,padding="same")
    conv3_2=tf.layers.batch_normalization(conv3_2,training=True)
    conv3_2=tf.nn.relu(conv3_2)
    conv3_2=tf.layers.dropout(conv3_2,rate=dr_rate)
    
    conv3_3=tf.layers.conv2d(conv3_2,filters=256,kernel_size=3,padding="same")
    conv3_3=tf.layers.batch_normalization(conv3_3,training=True)
    conv3_3=tf.nn.relu(conv3_3)
    pool3=tf.layers.max_pooling2d(conv3_3,pool_size=2,strides=2)
    pool3=tf.layers.dropout(pool3,rate=dr_rate)
    
      
    conv4_1=tf.layers.conv2d(pool3,filters=512,kernel_size=3,padding="same")
    conv4_1=tf.layers.batch_normalization(conv4_1,training=True)
    conv4_1=tf.nn.relu(conv4_1)
    conv4_1=tf.layers.dropout(conv4_1,rate=dr_rate)
    
    conv4_2=tf.layers.conv2d(conv4_1,filters=512,kernel_size=3,padding="same")
    conv4_2=tf.layers.batch_normalization(conv4_2,training=True)
    conv4_2=tf.nn.relu(conv4_2)
    conv4_2=tf.layers.dropout(conv4_2,rate=dr_rate)
    
    conv4_3=tf.layers.conv2d(conv4_2,filters=512,kernel_size=3,padding="same")
    conv4_3=tf.layers.batch_normalization(conv4_3,training=True)
    conv4_3=tf.nn.relu(conv4_3)
    pool4=tf.layers.max_pooling2d(conv4_3,pool_size=2,strides=2)
    pool4=tf.layers.dropout(pool4,rate=dr_rate)
    
    
    conv5_1=tf.layers.conv2d(pool4,filters=512,kernel_size=3,padding="same")
    conv5_1=tf.layers.batch_normalization(conv5_1,training=True)
    conv5_1=tf.nn.relu(conv5_1)
    conv5_1=tf.layers.dropout(conv5_1,rate=dr_rate)
    
    conv5_2=tf.layers.conv2d(conv5_1,filters=512,kernel_size=3,padding="same")
    conv5_2=tf.layers.batch_normalization(conv5_2,training=True)
    conv5_2=tf.nn.relu(conv5_2)
    conv5_2=tf.layers.dropout(conv5_2,rate=dr_rate)
    
    conv5_3=tf.layers.conv2d(conv5_2,filters=512,kernel_size=3,padding="same")
    conv5_3=tf.layers.batch_normalization(conv5_3,training=True)
    conv5_3=tf.nn.relu(conv5_3)
    pool5=tf.layers.max_pooling2d(conv5_3,pool_size=2,strides=2)
    pool5=tf.layers.dropout(pool5,rate=dr_rate)
    
    before_fc=tf.reshape(pool5,shape=[-1,int(pool5.get_shape()[1]*pool5.get_shape()[2]*pool5.get_shape()[3])])
    fc1=tf.layers.dense(before_fc,units=512)
    fc1=tf.layers.dropout(fc1,rate=dr_rate)
    logits=tf.layers.dense(fc1,units=self.class_num)
    
    return logits





