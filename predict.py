#coding=utf-8
from PIL import Image, ImageFilter
import tensorflow as tf
import matplotlib.pyplot as plt 
import numpy as np

from glob import glob

WIDTH = 224
HEIGHT = 224
CHANNEL = 3

init_op = tf.initialize_all_variables()
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.Session(config = config)
sess.run(init_op)
## load the graph and restore the params
saver = tf.train.import_meta_graph('model.ckpt-1000.meta')
#saver.restore(sess,tf.train.latest_checkpoint('./'))
saver.restore(sess, "./model.ckpt-8000")#这里使用了之前保存的模型参数
print ("Model restored.")

## get the tensor and operation
graph = tf.get_default_graph()
x=graph.get_operation_by_name('x').outputs[0]
#x=graph.get_tensor_by_name('x:0')
keep_prob=graph.get_tensor_by_name('keep_prob:0')
y=tf.get_collection("y")[0]
    

def imageprepare(file_name):
    """ 
    This function returns the pixel values.
    The input is a png file location.
    """
    im = Image.open(file_name)

#    plt.imshow(im)
#    plt.show()

    tv = np.asarray(im)
    tv = np.reshape(im, [WIDTH, HEIGHT, CHANNEL])
    tv = np.true_divide(tv, 255) - 0.5
    return tv 

def predictGender(file_name):
#    with tf.Session(config=config) as sess:
#        sess.run(init_op)
    
    
        result=imageprepare(file_name)
    
        prediction=tf.argmax(y,1)
        predint=prediction.eval(feed_dict={x: [result], keep_prob: 1}, session=sess)
    
        print('recognize result:%d' % predint[0])
        return predint[0]
