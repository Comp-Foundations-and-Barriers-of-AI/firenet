import tensorflow as tf

from tensorflow.keras.layers import Conv2D, MaxPool2D, Conv2DTranspose, ReLU, Concatenate, BatchNormalization, Dropout
from tensorflow.keras import Model, Sequential
from tensorflow.keras.activations import relu

class UNet(Model):
    """ U-net model
    
    Arguments
    ---------
    in_channels (int): Number of channel in the input tensor.
    out_channels (int): Number of channel in the output tensor.
    init_features (int): Number of channel in the convolution. This number is doubeled after each max-pooling
    use_bias (bool): Use bias in the convolutiional layers.
    kernel_size (int, tuple): Size of the kernel in the convolutional layers
    """

    def __init__(self, in_channels=2, out_channels=1, init_features=32, use_bias=True, kernel_size=(3,3)):
        super(UNet, self).__init__()

        self.kernel_size = (3,3)
        self.use_bias = use_bias

        features = init_features
        self.encoder1 = UNet._block(features, name="enc1", use_bias=use_bias, kernel_size=kernel_size)
        self.pool1 = MaxPool2D(pool_size=(2,2), strides=2, padding='same')
        self.encoder2 = UNet._block(features * 2, name="enc2", use_bias=use_bias, kernel_size=kernel_size)
        self.pool2 = MaxPool2D(pool_size=(2,2), strides=2, padding='same')
        self.encoder3 = UNet._block(features * 4, name="enc3", use_bias=use_bias, kernel_size=kernel_size)
        self.pool3 = MaxPool2D(pool_size=(2,2), strides=2, padding='same')
        self.encoder4 = UNet._block(features * 8, name="enc4", use_bias=use_bias, kernel_size=kernel_size)
        self.pool4 = MaxPool2D(pool_size=(2,2), strides=2, padding='same')

        self.bottleneck = UNet._block(features * 16, name="bottleneck", use_bias=use_bias, kernel_size=kernel_size)

        self.upconv4 = Conv2DTranspose(
            features * 8, kernel_size=2, strides=2, padding='same'
        )
        self.decoder4 = UNet._block(features * 8, name="dec4", use_bias=use_bias, kernel_size=kernel_size)
        self.upconv3 = Conv2DTranspose(
            features * 4, kernel_size=2, strides=2, padding='same'
        )
        self.decoder3 = UNet._block(features * 4, name="dec3", use_bias=use_bias, kernel_size=kernel_size)
        self.upconv2 = Conv2DTranspose(
            features * 2, kernel_size=2, strides=2, padding='same'
        )
        self.decoder2 = UNet._block(features * 2, name="dec2", use_bias=use_bias, kernel_size=kernel_size)
        self.upconv1 = Conv2DTranspose(
            features, kernel_size=2, strides=2, padding='same'
        )
        self.decoder1 = UNet._block(features, name="dec1", use_bias=use_bias, kernel_size=kernel_size)

        self.conv = Conv2D(
            out_channels, kernel_size=1, padding='same'
        )

        self.concat1 = Concatenate(axis=-1)
        self.concat2 = Concatenate(axis=-1)
        self.concat3 = Concatenate(axis=-1)
        self.concat4 = Concatenate(axis=-1)


    def __call__(self, x, training=False):
        enc1 = self.encoder1(x, training=training)
        enc2 = self.encoder2(self.pool1(enc1), training=training)
        enc3 = self.encoder3(self.pool2(enc2), training=training)
        enc4 = self.encoder4(self.pool3(enc3), training=training)

        bottleneck = self.bottleneck(self.pool4(enc4))

        dec4 = self.upconv4(bottleneck)
        dec4 = self.concat4([dec4, enc4])
        dec4 = self.decoder4(dec4, training=training)
        dec3 = self.upconv3(dec4)
        dec3 = self.concat3([dec3, enc3]) 
        dec3 = self.decoder3(dec3, training=training)
        dec2 = self.upconv2(dec3)
        dec2 = self.concat2([dec2, enc2]) 
        dec2 = self.decoder2(dec2, training=training)
        dec1 = self.upconv1(dec2)
        dec1 = self.concat1([dec1, enc1]) 
        dec1 = self.decoder1(dec1, training=training)
        return self.conv(dec1)

    @staticmethod
    def _block(features, name, use_bias=True, kernel_size=(3,3)):
        return Sequential(
                        [Conv2D(
                            filters=features,
                            kernel_size=kernel_size,
                            padding='same',
                            use_bias=use_bias,
                        ),
                    BatchNormalization(),
                    ReLU(),
                    Conv2D(
                            filters=features,
                            kernel_size=kernel_size,
                            padding="same",
                            use_bias=use_bias,
                        ),
                    BatchNormalization(),
                    ReLU(),
                ]
            )











