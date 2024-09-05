import tensorflow as tf
from tensorflow.keras import layers, initializers, optimizers, callbacks


# Advanced Radial Basis Function Layer
class RadialBasisFunction(layers.Layer):
    def __init__(self, grid_min, grid_max, num_grids, **kwargs):
        super(RadialBasisFunction, self).__init__(**kwargs)
        self.grid = tf.cast(tf.linspace(grid_min, grid_max, num_grids), dtype=tf.float32)
        self.denominator = tf.cast((grid_max - grid_min) / num_grids, dtype=tf.float32)
        self.grid = tf.expand_dims(self.grid, axis=0)  # Broadcast grid

    def call(self, x):
        return tf.exp(-tf.square((x[..., None] - self.grid) / self.denominator))


# Squeeze-and-Excite Block
class SqueezeExciteBlock(layers.Layer):
    def __init__(self, reduction_ratio=16):
        super(SqueezeExciteBlock, self).__init__()
        self.reduction_ratio = reduction_ratio

    def build(self, input_shape):
        self.global_avg_pool = layers.GlobalAveragePooling1D()  # Adjusted for 1D inputs
        self.fc1 = layers.Dense(input_shape[-1] // self.reduction_ratio, activation='relu',
                                kernel_initializer=initializers.he_normal())
        self.fc2 = layers.Dense(input_shape[-1], activation='sigmoid',
                                kernel_initializer=initializers.he_normal())

    def call(self, x):
        se = self.global_avg_pool(x)
        se = self.fc1(se)
        se = self.fc2(se)
        se = tf.expand_dims(se, axis=1)
        return x * se


# Self-Attention with Squeeze-and-Excitation block
class SelfAttentionWithSE(layers.Layer):
    def __init__(self, num_heads, key_dim, use_se_block=False):
        super(SelfAttentionWithSE, self).__init__()
        self.multi_head_attention = layers.MultiHeadAttention(num_heads=num_heads, key_dim=key_dim)
        self.use_se_block = use_se_block
        if self.use_se_block:
            self.se_block = SqueezeExciteBlock()

    def call(self, x):
        attention_output = self.multi_head_attention(x, x)
        if self.use_se_block:
            attention_output = self.se_block(attention_output)
        return attention_output


# Enhanced FastKAN Layer with Residual Connections and Batch Normalization
class EnhancedFastKANLayer(layers.Layer):
    def __init__(self, input_dim, output_dim, grid_min, grid_max, num_grids, use_base_update, base_activation,
                 spline_weight_init_scale, use_attention=False, num_heads=4, key_dim=32, use_se_block=False,
                 dropout_rate=0.0):
        super(EnhancedFastKANLayer, self).__init__()
        self.norm = layers.BatchNormalization()  # Using Batch Normalization instead of LayerNormalization
        self.rbf = RadialBasisFunction(grid_min, grid_max, num_grids)
        self.spline_linear = layers.Dense(output_dim, kernel_initializer=initializers.he_normal())
        self.use_base_update = use_base_update
        self.use_attention = use_attention
        self.dropout_rate = dropout_rate

        if use_base_update:
            self.base_activation = layers.Activation(base_activation)
            self.base_linear = layers.Dense(output_dim, kernel_initializer=initializers.he_normal())

        if use_attention:
            self.attention = SelfAttentionWithSE(num_heads=num_heads, key_dim=key_dim, use_se_block=use_se_block)

        if dropout_rate > 0.0:
            self.dropout = layers.Dropout(dropout_rate)

        # Residual connection initialization
        if input_dim != output_dim:
            self.residual_connection = layers.Dense(output_dim, kernel_initializer=initializers.he_normal())
        else:
            self.residual_connection = lambda x: x

    def call(self, x):
        x_norm = self.norm(x)

        if self.use_attention:
            x_norm = self.attention(x_norm)

        spline_basis = self.rbf(x_norm)
        spline_basis_flat = tf.reshape(spline_basis, [tf.shape(spline_basis)[0], -1])
        ret = self.spline_linear(spline_basis_flat)

        if self.use_base_update:
            base = self.base_linear(self.base_activation(x))
            ret = ret + base

        if self.dropout_rate > 0.0:
            ret = self.dropout(ret)

        # Apply residual connection
        ret += self.residual_connection(x)

        return ret


# Enhanced FastKAN Model with Learning Rate Scheduler and Early Stopping
class Attent_FastKAN(tf.keras.Model):
    def __init__(self, layers_hidden, grid_min=-1, grid_max=1, num_grids=10, use_base_update=False,
                 base_activation='relu', spline_weight_init_scale=1, use_attention=False, num_heads=4, key_dim=32,
                 use_se_block=False, dropout_rate=0.0):
        super(Attent_FastKAN, self).__init__()
        self.layers_list = []
        for in_dim, out_dim in zip(layers_hidden[:-1], layers_hidden[1:]):
            self.layers_list.append(
                EnhancedFastKANLayer(in_dim, out_dim, grid_min, grid_max, num_grids, use_base_update, base_activation,
                                     spline_weight_init_scale, use_attention, num_heads, key_dim, use_se_block,
                                     dropout_rate))

    def call(self, inputs):
        x = inputs
        for layer in self.layers_list:
            x = layer(x)
        return x