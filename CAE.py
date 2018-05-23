import tensorflow as tf
import numpy as np
import matplotlib.image as mpimg
import random
import pickle

#from MNIST_AE import encoder as mnise_encoder
#from MNISE_AE import decoder as mnise_decoder
#from CIFAR_AE import encoder as cifar_encoder
#from CIFAR_AE import decoder as cifar_decoder
#from CIFAR_AE import get_data


from tensorflow.examples.tutorials.mnist import input_data as mnist_input
mnist = mnist_input.read_data_sets("MNIST_data/", one_hot=False)

total_epochs = 150
batch_size = 100
learning_rate = 0.001
random_size = 100
mnist_image_size = 28*28
cifar_image_size = 32
z_dim = 400

init = tf.random_normal_initializer(mean=0, stddev=0.15)

def conv_image(raw):
    to_float = np.array(raw,dtype=float)/255.0
    images = to_float.reshape([-1,3,cifar_image_size,cifar_image_size])
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

def mnist_encoder(x, reuse = False ):
    l = [mnist_image_size, 50, 30, z_dim]
    with tf.variable_scope(name_or_scope = "mnist_encoder") as scope:
        out1 = tf.layers.dense(x, l[1], activation=tf.nn.relu)
        out2 = tf.layers.dense(out1, l[2], activation=tf.nn.relu)
        output = tf.layers.dense(out2, l[3], activation=tf.nn.sigmoid)
        return output

def mnist_decoder(z, reuse = False):
    l = [z_dim, 30, 50, mnist_image_size]
    with tf.variable_scope(name_or_scope="mnist_decoder", reuse=reuse) as scope:
        out1 = tf.layers.dense(z, l[1], activation=tf.nn.relu)
        out2 = tf.layers.dense(out1, l[2], activation=tf.nn.relu)
        output = tf.layers.dense(out2, l[3], activation=tf.nn.sigmoid)               
        return output

def cifar_encoder(x, reuse = False ):
    l = [16, 32, 64, z_dim] 
    with tf.variable_scope(name_or_scope = "cifar_encoder") as scope:
        out1 = tf.layers.conv2d(x, l[0], 3, strides=2, padding='SAME', name='conv0') # 32*32*3 -> 16*16*16
        out1 = tf.nn.leaky_relu(out1)
        out2 = tf.layers.conv2d(out1, l[1], 3, strides=2, padding='SAME', name='conv1') # 16*16*16 -> 8*8*32
        out2 = tf.nn.leaky_relu(out2)
        out3 = tf.layers.conv2d(out2, l[2], 3, strides=2, padding='SAME', name='conv2') # 8*8*32 -> 4*4*64
        out3 = tf.nn.leaky_relu(out3)
        out3 = tf.layers.flatten(out3) # 4*4*64 -> 1024
        output = tf.layers.dense(out3, l[3],activation=tf.nn.sigmoid, name='fc') # 1024 -> 20
        return output

def cifar_decoder(z, reuse = False):
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
    mnist_X = tf.placeholder(tf.float32, [None, mnist_image_size])
    mnist_Y = tf.placeholder(tf.float32, [None])
    cifar_X = tf.placeholder(tf.float32, [None, cifar_image_size, cifar_image_size, 3])
    cifar_Y = tf.placeholder(tf.float32, [None])
    Z = tf.placeholder(tf.float32, [1,z_dim])
    
    mnist_enc = mnist_encoder(mnist_X)
    mnist_dec = mnist_decoder(mnist_enc)

    cifar_enc = cifar_encoder(cifar_X)
    cifar_dec = cifar_decoder(cifar_enc)
    
    mnist_dec_gen = mnist_decoder(Z,True)
    cifar_dec_gen = cifar_decoder(Z,True)
    
    mnist_loss = tf.reduce_mean( tf.square(mnist_X - mnist_dec) )
    cifar_loss = tf.reduce_mean( tf.square(cifar_X - cifar_dec) )
    is_equal = tf.cast(tf.equal(mnist_Y,cifar_Y),tf.float32)
    dist = tf.reduce_mean(tf.square(mnist_enc-cifar_enc))
    z_loss =tf.reduce_mean(tf.multiply(is_equal,dist)*9-tf.multiply((1-is_equal),dist))
    loss = mnist_loss + cifar_loss + z_loss*0.03

    #t_vars = tf.trainable_variables()
    #m_vars = [var for var in t_vars if "mnist" in var.name]
    #c_vars = [var for var in t_vars if "cifar" in var.name]
    optimizer = tf.train.AdamOptimizer(learning_rate)
    #train = optimizer.minimize(loss, var_list = e_vars+d_vars)
    train = optimizer.minimize(loss)

with tf.Session(graph = g) as sess:
    cifar_data_x, cifar_data_y = get_data()
    print(cifar_data_x.shape)
    print(cifar_data_y.shape)
    sess.run(tf.global_variables_initializer())
    total_batchs = int(50000 / batch_size)
   
    for epoch in range(total_epochs):
        for batch in range(total_batchs):
            cifar_batch_x = cifar_data_x[batch * batch_size : (batch+1) * batch_size]
            cifar_batch_y = cifar_data_y[batch * batch_size : (batch+1) * batch_size]
            mnist_batch_x, mnist_batch_y = mnist.train.next_batch(batch_size)

            feed = {
                mnist_X: mnist_batch_x,
                mnist_Y: mnist_batch_y,
                cifar_X: cifar_batch_x,
                cifar_Y: cifar_batch_y,
            }
            sess.run(train, feed_dict = feed)
            ml, cl, zl, loss_r = sess.run([mnist_loss, cifar_loss, z_loss, loss], feed_dict = feed)
            
        if epoch % 2 == 0:
            print("======= Epoch : ", epoch , " =======")
            print("loss: " , loss_r)
            print("mnist_loss: " , ml)
            print("cifar_loss: " , cl)
            print("z_loss: ", zl)

            mnist_gen = sess.run(mnist_dec,feed_dict = {mnist_X: mnist_batch_x})
            mnist_img = mnist_gen[0].reshape([28,28])
            mpimg.imsave('./epoch/epoch'+str(epoch)+'_mnist.png',mnist_img,format='png')

            cifar_gen = sess.run(cifar_dec,feed_dict = {cifar_X: cifar_batch_x})
            cifar_img = cifar_gen[0].reshape([32,32,3])
            mpimg.imsave('./epoch/epoch'+str(epoch)+'_cifar.png',cifar_img,format='png')



