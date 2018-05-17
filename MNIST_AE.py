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

init = tf.random_normal_initializer(mean=0, stddev=0.15)

def encoder(x, reuse = False ):
    with tf.variable_scope(name_or_scope = "encoder") as scope:
        W1 = tf.get_variable(name="W1",shape = [784,50],initializer = init)
        b1 = tf.get_variable(name="b1",shape = [50],initializer = init)
        r1 = tf.matmul(x,W1)+b1
        out1 = tf.nn.relu(r1)
        
        W2 = tf.get_variable(name="W2",shape = [50,30],initializer = init)
        b2 = tf.get_variable(name="b2",shape = [30],initializer = init)
        r2 = tf.matmul(out1,W2)+b2
        out2 = tf.nn.relu(r2)
        
        W3 = tf.get_variable(name="W3",shape = [30,20],initializer = init)
        b3 = tf.get_variable(name="b3",shape = [20],initializer = init)
        r3 = tf.matmul(out2,W3)+b3
        output = tf.nn.sigmoid(r3)
        return output

def decoder(z, reuse = False):
    with tf.variable_scope(name_or_scope="decoder", reuse=reuse) as scope:
        W1 = tf.get_variable(name="W1",shape = [20,30],initializer = init)
        b1 = tf.get_variable(name="b1",shape = [30],initializer = init)
        r1 = tf.matmul(z,W1)+b1
        out1 = tf.nn.relu(r1)

        W2 =tf.get_variable(name="W2",shape = [30,50],initializer = init)
        b2 = tf.get_variable(name="b2",shape = [50],initializer = init)
        r2 = tf.matmul(out1,W2)+b2
        out2 = tf.nn.relu(r2)
        
        W3 =tf.get_variable(name="W3",shape = [50,784],initializer = init)
        b3 = tf.get_variable(name="b3",shape = [784],initializer = init)
        r3 = tf.matmul(out2,W3)+b3
        output = tf.nn.sigmoid(r3)
        return output

def random_z():
    return np.random.normal(size=[1,20])

g = tf.Graph()

with g.as_default():
    X = tf.placeholder(tf.float32, [None, 784])
    Z = tf.placeholder(tf.float32, [1,20])
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
