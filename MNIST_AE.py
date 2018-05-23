import tensorflow as tf
import numpy as np
import matplotlib.image as mpimg
import random

from tensorflow.examples.tutorials.mnist import input_data as mnist_input
mnist = mnist_input.read_data_sets("MNIST_data/")

total_epochs = 150
batch_size = 100
learning_rate = 0.001
random_size = 100
image_size = 28*28
z_dim = 20

init = tf.random_normal_initializer(mean=0, stddev=0.15)

def encoder(x, reuse = False ):
    l = [image_size, 50, 30, z_dim]
    with tf.variable_scope(name_or_scope = "mnist_encoder") as scope:
        out1 = tf.layers.dense(x, l[1], activation=tf.nn.relu)
        out2 = tf.layers.dense(out1, l[2], activation=tf.nn.relu)
        output = tf.layers.dense(out2, l[3], activation=tf.nn.sigmoid)
        return output

def decoder(z, reuse = False):
    l = [z_dim, 30, 50, image_size]
    with tf.variable_scope(name_or_scope="mnist_decoder", reuse=reuse) as scope:
        out1 = tf.layers.dense(z, l[1], activation=tf.nn.relu)
        out2 = tf.layers.dense(out1, l[2], activation=tf.nn.relu)
        output = tf.layers.dense(out2, l[3], activation=tf.nn.sigmoid)               
        return output

def random_z():
    return np.random.normal(size=[1,z_dim])

g = tf.Graph()

with g.as_default():
    X = tf.placeholder(tf.float32, [None, image_size])
    Z = tf.placeholder(tf.float32, [1,z_dim])
    enc = encoder(X)
    dec = decoder(enc)
    dec_gen = decoder(Z,True)
    
    loss = tf.reduce_mean( tf.square(X-dec) )

    t_vars = tf.trainable_variables()
    e_vars = [var for var in t_vars if "encoder" in var.name]
    d_vars = [var for var in t_vars if "decoder" in var.name]
    optimizer = tf.train.AdamOptimizer(learning_rate)
    train = optimizer.minimize(loss, var_list = e_vars+d_vars)



with tf.Session(graph = g) as sess:
    sess.run(tf.global_variables_initializer())
    total_batchs = int(55000 / batch_size)

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
