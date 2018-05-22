import tensorflow as tf
import numpy as np
import matplotlib.image as mpimg
import random
import pickle

total_epochs = 150
batch_size = 100
learning_rate = 0.001
image_size = 32

init = tf.random_normal_initializer(mean=0, stddev=0.15)

def conv_image(raw):
    to_float = np.array(raw,dtype=float)/255.0
    images = to_float.reshape([-1,3,image_size,image_size])
    images = images.transpose([0,2,3,1])
    return images

def extract_data_from_pickle(filename):
    with open(filename,mode='rb') as file:
            data = pickle.load(file,encoding='bytes')
    return conv_image(data[b'data']),data[b'labels']

def get_data():
    l_x,l_y=extract_data_from_pickle('./cifar-10-batches-py/data_batch_1')
    filenames= ['./cifar-10-batches-py/data_batch_{}'.format(i) for i in range(2,6)]
    for filename in filenames:
        data_x,data_y = extract_data_from_pickle(filename)
        l_x=np.concatenate((l_x,data_x),axis=0)
        l_y=np.concatenate((l_y,data_y),axis=0)
    return l_x,l_y

def encoder(x, reuse = False ):
    with tf.variable_scope(name_or_scope = "encoder") as scope:
        return output

def decoder(z, reuse = False):
    with tf.variable_scope(name_or_scope="decoder", reuse=reuse) as scope:
        return output

g = tf.Graph()

with g.as_default():
    X = tf.placeholder(tf.float32, [None, image_size,image_size,3])
    enc = encoder(X)
    dec = decoder(enc)
    
    loss = tf.reduce_mean( tf.square(X-dec) )

    t_vars = tf.trainable_variables()
    e_vars = [var for var in t_vars if "encoder" in var.name]
    d_vars = [var for var in t_vars if "decoder" in var.name]
    optimizer = tf.train.AdamOptimizer(learning_rate)
    train = optimizer.minimize(loss, var_list = e_vars+d_vars)



with tf.Session(graph = g) as sess:
    data_x,data_y = get_data()
'''
    sess.run(tf.global_variables_initializer())
    total_batchs = int(50000 / batch_size)
    for epoch in range(total_epochs):
        for batch in range(total_batchs):
            batch_x,batch_y = mnist.train.next_batch(batch_size)
            sess.run(train, feed_dict = {X : batch_x })
            loss_r = sess.run(loss, feed_dict = {X : batch_x})

        if epoch % 2 == 0:
            print("======= Epoch : ", epoch , " =======")
            print("loss: " , loss_r)
            sample_z = random_z()
            gen = sess.run(dec,feed_dict = {X: batch_x})
            img = gen[0].reshape([28,28])
            mpimg.imsave('./epoch/epoch'+str(epoch)+'.png',img,format='png')
'''
