import numpy as np
import keras.backend as K
from keras.engine.topology import Layer
from keras.layers import Conv2D, Reshape
from keras.activations import softmax

def squash(vector, epsilon=K.epsilon()):

    vector += epsilon
    norm = K.sum(K.square(vector), -1, keepdims=True)
    scalar_factor = norm / (1 + norm) / K.sqrt(norm)
    squashed = scalar_factor * vector
    return (squashed)


def conv2d_caps(input_layer, nb_filters, kernel_size, capsule_size, strides=2):

    conv = Conv2D(
        filters=nb_filters*capsule_size, 
        kernel_size=kernel_size, 
        strides=strides,
        padding='valid'
    )(input_layer)

    conv_shape = conv.shape
    nb_capsules= int(conv_shape[1]*conv_shape[2]*nb_filters)

    capsules = Reshape(target_shape=(nb_capsules, capsule_size, 1))(conv)

    return squash(capsules)


class DenseCapsule(Layer):

    """
        A fully connected capsule layer which is similar to
        the dense layer but replace the neurons to capsules 
    """

    def __init__(self, capsule_size, nb_capsules,  kernel_initializer='glorot_uniform', iterations=5, **kwargs):
        super(DenseCapsule, self).__init__(**kwargs)
        self.nb_capsules = nb_capsules
        self.iterations = iterations
        self.capsule_size = capsule_size
        self.initializer = kernel_initializer

    def build(self, input_shape):
        self.prev_shape = input_shape
        self.w_ij = self.add_weight(
            name='w_ij',
            shape=(input_shape[1], self.nb_capsules, self.capsule_size, input_shape[2]),
            initializer=self.initializer
        )
        self.built = True

    def batch_dot(self, X, w, axis):
        return K.map_fn(lambda x: K.batch_dot(x, w, axis), elems=X)

    def _dynamic_routing(self, u_hat, b_ij):
        
        for i in range(self.iterations):

            c_ij = softmax(b_ij, axis=1)
            s_j = K.sum(K.batch_dot(u_hat, c_ij, [3, 4]), axis=1, keepdims=True)
            v_j = squash(s_j)

            if i < self.iterations-1:
                b_ij = b_ij + K.batch_dot(K.tile(v_j,  [1, self.prev_shape[1], 1, 1, 1]), u_hat, [3, 4])

        return K.squeeze(v_j, axis=1)

    def call(self, inputs):
        
        expanded_input = K.expand_dims(inputs, 2)
        expanded_input = K.tile(expanded_input, [1, 1, self.nb_capsules, 1, 1])

        u_hat = self.batch_dot(expanded_input, self.w_ij, [2, 3])
        
        b_ij = K.zeros(shape=[K.shape(u_hat)[0], self.prev_shape[1], self.nb_capsules, 1, 1], dtype=np.float32) 
        
        return self._dynamic_routing(u_hat, b_ij)

    def compute_output_shape(self, input_shape):
        return tuple([None, self.nb_capsules, self.capsule_size, 1])

def test():
    from keras.layers import Input
    input_layer = Input(shape=(28, 28, 1))
    caps1 = conv2d_caps(input_layer, 64, (3, 3), 8)
    print(caps1.shape)
    caps2 = DenseCapsule(4, 100)(caps1)
    print(caps2.shape)

if __name__ == '__main__':
    test()