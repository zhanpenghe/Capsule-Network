import numpy as np
import tensorflow as tf


def squash(vector, epsilon=10e-5):

    vector += epsilon
    norm = tf.reduce_sum(tf.square(vector), -2, keepdims=True)
    scalar_factor = norm / (1 + norm) / tf.sqrt(norm)
    squashed = scalar_factor * vector
    return (squashed)


# Convolution->Reshape
def conv2d_caps(input_layer, nb_filters, kernel, capsule_size, strides=2):

    conv = tf.layers.conv2d(
        inputs=input_layer, 
        filters=nb_filters*capsule_size, 
        kernel_size=kernel, 
        strides=strides, 
        padding='valid'
    )
    shape = conv.get_shape().as_list()
    capsules = tf.reshape(conv, shape=[-1, np.prod(shape[1:3]) * nb_filters, capsule_size, 1])
    return squash(capsules)

def dynamic_routing(u_hat, b_ij, nb_capsules, prev_nb_capsules, iterations=5):
    """
        The dynamic routing algorithm from paper: https://arxiv.org/pdf/1710.09829.pdf
    """
    for i in range(iterations):
        with tf.variable_scope('routing'+str(i)):
            
            c_ij = tf.nn.softmax(b_ij, axis=2)
            s_j =tf.reduce_sum(tf.multiply(c_ij, u_hat), axis=1, keepdims=True)
            v_j = squash(s_j)

            if i < iterations-1:
                b_ij = b_ij + tf.reduce_sum(
                    tf.matmul(u_hat, tf.tile(v_j, [1, prev_nb_capsules, 1, 1, 1]), transpose_a=True),
                    axis=0,
                    keepdims=True
                )

    return tf.squeeze(v_j, axis=1)


def dense_capsule(input_layer, capsule_size, nb_capsules, iterations=5):
    """
        Take the output from a layer of capsules and perform the following computations:
        ......
    """
    prev_shape = input_layer.get_shape().as_list()
    init = tf.random_normal_initializer(stddev=0.01, seed=0)
    w_shape = [prev_shape[1], nb_capsules, capsule_size, prev_shape[2]]
    w_ij = tf.get_variable('weight', shape=w_shape, dtype=tf.float32, initializer=init)


    # Expand dimension to allow the dot product
    # Dimension change from [None, prev_nb_capsules, prev_capsule_size, 1]
    # to [None, prev_nb_capsules, prev_capsule_size, nb_capsules, 1]
    expanded_input_layer = tf.expand_dims(input=input_layer, axis=2)
    
    # Make nb_capsule of copies of previous capsules to proform multiplication
    expanded_input_layer = tf.tile(expanded_input_layer, [1, 1, nb_capsules, 1, 1])

    u_hat = tf.einsum('abdc,iabcf->iabdf', w_ij, expanded_input_layer)
    b_ij = tf.zeros(shape=[prev_shape[1], nb_capsules, 1, 1], dtype=np.float32)

    return dynamic_routing(u_hat, b_ij, nb_capsules, prev_shape[1], iterations)

def test():
    # Some testing..
    input_layer = tf.placeholder(tf.float32, shape=[None, 28, 28, 1])
    caps1 = conv2d_caps(input_layer, 64, [3, 3], 8)
    print(caps1.shape)
    caps2 = dense_capsule(caps1, 4, 100)
    print(caps2.shape)

if __name__ == '__main__':
    test()