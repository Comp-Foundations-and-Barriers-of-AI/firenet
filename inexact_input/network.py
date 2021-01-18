import tensorflow as tf
from tensorflow.keras import Model
import numpy as np



class LISTA(Model):
    """
The LISTA architecture

Arguments
---------
m (int): Input dimension
N (int): Output dimension
nbr_layers (int): Number of layers
theta_min, theta_max (float): Initialize all soft threshoulding parameters with
                              random numbers from the uniform distribution on 
                              the interval [theta_min, theta_max]. 
dtype (tf.dtypes) : Precision.
    """


    def __init__(self,m, N, A, nbr_layers, theta_min=0, theta_max=1.0, dtype=tf.float64):
        super(LISTA, self).__init__()

        self.m = m
        self.N = N
        self.A = tf.constant(np.swapaxes(A, 0,1), dtype=dtype)
        self.norm_A = float(np.linalg.norm(A,2))*10
        self.nbr_layers = nbr_layers
        self.dtype1=dtype

        self.dense_layers = []
        self.theta = [];
        
        w_init = tf.random_normal_initializer();
        theta_init = tf.random_uniform_initializer(minval=theta_min, maxval=theta_max)

        for i in range(nbr_layers):
            self.dense_layers.append(tf.Variable(
                                     initial_value=w_init(shape=(m,N), dtype=self.dtype1)/self.norm_A, #tf.eye(num_rows=m,num_columns=N, 
                                     trainable=True,
                                     name=f'W_{i}'
                                     )
                                    )

            self.theta.append(tf.Variable(
                                     initial_value=theta_init(shape=(1,), 
                                     dtype=dtype),
                                     trainable=True,
                                     name=f'theta_{i}',
                                     constraint=tf.nn.relu
                                     )
                                    )
    def __call__(self, y, training=False):
        """Evaluate the network"""
        xk = tf.zeros([y.shape[0], self.N], dtype=self.dtype1)
        for i in range(self.nbr_layers):
            a = y - tf.linalg.matmul(xk/self.norm_A, self.A)
            a = xk + tf.linalg.matmul(a ,self.dense_layers[i]);
            xk = tf.nn.relu(a-self.theta[i]) - tf.nn.relu(-a-self.theta[i]);

        return xk
