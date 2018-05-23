import tensorflow as tf
import numpy as np
import matplotlib.image as mpimg
import random
import pickle
import random

total_epochs = 150
batch_size = 100
learning_rate = 0.001
image_size = 32
z_dim = 300

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
    l = [16, 32, 64, z_dim] 
    with tf.variable_scope(name_or_scope = "cifar_encoder") as scope:
        out1 = tf.layers.conv2d(x, l[0], 3, strides=2, padding='SAME', name='conv0') # 32*32*3 -> 16*16*16
        out1 = tf.nn.leaky_relu(out1)
        out2 = tf.layers.conv2d(out1, l[1], 3, strides=2, padding='SAME', name='conv1') # 16*16*16 -> 8*8*32
        out2 = tf.nn.leaky_relu(out2)
        out3 = tf.layers.conv2d(out2, l[2], 3, strides=2, padding='SAME', name='conv2') # 8*8*32 -> 4*4*64
        out3 = tf.nn.leaky_relu(out3)
        out3 = tf.layers.flatten(out3) # 4*4*64 -> 1024
        output = tf.layers.dense(out3, l[3], name='fc') # 1024 -> 20
        return output

def decoder(z, reuse = False):
    l = [64, 32, 16, 3]
    with tf.variable_scope(name_or_scope="cifar_decoder", reuse=reuse) as scope:
        z = tf.layers.dense(z, l[0]*4*4, name='fc') # 20 -> 1024
        z = tf.reshape(z, [-1, 4, 4, l[0]]) # 1024 -> 4*4*64 
        out1 = tf.layers.conv2d_transpose(z, l[1], 3, strides=2, padding='SAME', name='conv_trans0') # 4*4*64 -> 8*8*32
        out1 = tf.nn.relu(out1)
        out2 = tf.layers.conv2d_transpose(out1, l[2], 3, strides=2, padding='SAME', name='conv_trans1') # 8*8*32 -> 16*16*16
        out2 = tf.nn.relu(out2)
        out3 = tf.layers.conv2d_transpose(out2, l[3], 3, strides=2, padding='SAME', name='conv_trans2') # 16*16*16 -> 32*32*3
        output = tf.nn.sigmoid(out3)
        return output

def random_z():
    return np.random.normal(size=[1,z_dim])

g = tf.Graph()

with g.as_default():
    X = tf.placeholder(tf.float32, [None, image_size,image_size,3])
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
    data_x,data_y = get_data()
    sess.run(tf.global_variables_initializer())
    total_batchs = int(50000 / batch_size)
    for epoch in range(total_epochs):
        for batch in range(total_batchs):
            batch_x = data_x[batch * batch_size : (batch+1) * batch_size]
            #batch_x,batch_y = mnist.train.next_batch(batch_size)
            sess.run(train, feed_dict = {X : batch_x })
            loss_r = sess.run(loss, feed_dict = {X : batch_x})

        if epoch % 2 == 0:
            print("======= Epoch : ", epoch , " =======")
            print("loss: " , loss_r)
            sample_z = random_z()
            gen = sess.run(dec,feed_dict = {X: batch_x})
            idx = random.randint(0,batch_size-1)
            img = gen[idx].reshape([32,32,3])
            mpimg.imsave('./epoch/epoch'+str(epoch)+'.png',img,format='png')
            mpimg.imsave('./epoch/epoch'+str(epoch)+'_original.png',batch_x[idx],format='png')
