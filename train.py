import vgg
import numpy as np
import os
import tensorflow as tf

def image_process(imgs):
  # imgs in [batch,3072]
  res=np.reshape(imgs,[-1,3,32,32])
  res=np.swapaxes(res,1,2)
  res=np.swapaxes(res,2,3)
  return res

def load_data(file_name):
  import pickle
  with open(file_name,'rb') as fo:
    data_file=pickle.load(fo,encoding="bytes")
  inputs=data_file[b'data']
  labels=data_file[b'fine_labels']
  
  inputs=image_process(inputs)
  labels_onehot=[]
  eye100=np.eye(100)
  for label in labels:
    labels_onehot.append(eye100[label])
  return [inputs,labels_onehot]


def acc(logits,labels):
  acc=0
  pred=np.argmax(logits,axis=-1)
  labels_sparse=np.argmax(labels,axis=-1)
  for i in range(np.shape(logits)[0]):
    if pred[i]==labels_sparse[i]:
      acc+=1
  return acc/np.shape(logits)[0]


def train(model_obj,restore_iter=0,model_dir="model_dir",save_iter=10,test_iter=10,max_epoch=10000,batch_size=100):
  # abandon final incomplete batch in an epoch
  # extract data loader later
  # save model to "./model_dir"
  
  # prepare data
  train_data=load_data("./train")
  test_data=load_data("./test")
  #
  train_num=np.shape(train_data[0])[0]
  train_index=np.arange(train_num)
  
  optimizer=tf.train.AdamOptimizer(1e-3)
  with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
    train_step=optimizer.minimize(model_obj.loss)
  
  correct=tf.reduce_sum(tf.cast(tf.equal(tf.argmax(model_obj.logits,axis=-1),tf.argmax(model_obj.y,axis=-1)),dtype=tf.float32))
  
  init=tf.global_variables_initializer()
  saver=tf.train.Saver(tf.global_variables(),max_to_keep=1)
  
  sess=tf.Session()
  sess.run(init)
  if not os.path.exists(model_dir):
    os.system("mkdir ./"+model_dir)
  if restore_iter!=0:
    saver.restore(sess,"./"+model_dir+"/model_epoch_"+str(restore_iter)+".ckpt")
  
  print("initialized..")
  for e in range(max_epoch):
    np.random.shuffle(train_index)
    epoch_loss=0
    for b in range(train_num//batch_size):
      start=int(b*batch_size)
      end=int(batch_size+b*batch_size)
      train_batch_x=np.array(train_data[0])[train_index[start:end]]
      train_batch_y=np.array(train_data[1])[train_index[start:end]]
      [_op,batch_loss]=sess.run([train_step,model_obj.loss],feed_dict={model_obj.x:train_batch_x,model_obj.y:train_batch_y,model_obj.dr_rate:0.5})
      epoch_loss+=batch_loss
    epoch_loss=epoch_loss/(train_num//batch_size)
    print("epoch "+str(e+1)+" loss:"+str(epoch_loss))
    if (e+1)%test_iter==0:
      print("testing")
      train_acc=0
      test_acc=0
      for b in range(np.shape(test_data[0])[0]//100):
        test_acc+=sess.run(correct,feed_dict={model_obj.x:test_data[0][b*100:(b+1)*100],model_obj.y:test_data[1][b*100:(b+1)*100],model_obj.dr_rate:0.0})
      if np.shape(test_data[0])[0]%100!=0:
        test_acc+=sess.run(correct,feed_dict={model_obj.x:test_data[0][np.shape(test_data[0])[0]//100*100:],model_obj.y:test_data[1][np.shape(test_data[0])[0]//100*100:],model_obj.dr_rate:0.0})
      test_acc=test_acc/np.shape(test_data[0])[0]
      for b in range(np.shape(train_data[0])[0]//100):
        train_acc+=sess.run(correct,feed_dict={model_obj.x:train_data[0][b*100:(b+1)*100],model_obj.y:train_data[1][b*100:(b+1)*100],model_obj.dr_rate:0.0})
      if np.shape(train_data[0])[0]%100!=0:
        train_acc+=sess.run(correct,feed_dict={model_obj.x:train_data[0][np.shape(train_data[0])[0]//100*100:],model_obj.y:train_data[1][np.shape(train_data[0])[0]//100*100:],model_obj.dr_rate:0.0})
      train_acc=train_acc/np.shape(train_data[0])[0]
      print("test acc: "+str(test_acc)+"  train acc: "+str(train_acc))
    if (e+1)%save_iter==0:
      print("saving")
      saver.save(sess,model_dir+"/"+"model_epoch_"+str(e+1)+".ckpt")
      
if __name__=="__main__":
  vgg16=vgg.vgg_creator()
  train(vgg16,save_iter=1,test_iter=1)

