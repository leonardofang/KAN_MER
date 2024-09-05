import tensorflow as tf
from tensorflow.keras import layers


# based on the model FastKAN
# @article{li2024kolmogorovarnold,
#       title={Kolmogorov-Arnold Networks are Radial Basis Function Networks},
#       author={Ziyao Li},
#       year={2024},
#       eprint={2405.06721},
#       archivePrefix={arXiv},
#       primaryClass={cs.LG}
# }


class RadialBasisFunction(layers.Layer):
    def __init__(self, grid_min, grid_max, num_grids, **kwargs):
        super(RadialBasisFunction, self).__init__(**kwargs)

        self.grid = tf.cast(
            tf.linspace(grid_min, grid_max, num_grids),
            dtype=tf.float32,
        )
        self.denominator = tf.cast(
            (grid_max - grid_min) / num_grids,
            dtype=tf.float32,
        )

    def call(self, x):
        return tf.exp(-((x[..., None] - self.grid) / self.denominator) ** 2)


class FastKANLayer(layers.Layer):
    def __init__(self, input_dim, output_dim, grid_min, grid_max, num_grids, use_base_update, base_activation, spline_weight_init_scale):
        super(FastKANLayer, self).__init__()
        self.norm = layers.LayerNormalization(axis=-1)
        self.rbf = RadialBasisFunction(grid_min, grid_max, num_grids)
        self.spline_linear = layers.Dense(output_dim)
        self.use_base_update = use_base_update
        if use_base_update:
            self.base_activation = base_activation
            self.base_linear = layers.Dense(output_dim)
    def call(self, x):
        x_norm = self.norm(x)
        spline_basis = self.rbf(x_norm)
        spline_basis_flat = tf.reshape(spline_basis, [tf.shape(spline_basis)[0], -1])
        ret = self.spline_linear(spline_basis_flat)
        if self.use_base_update:
            base = self.base_linear(self.base_activation(x))
            ret = ret + base
        return ret


class FastKAN(tf.keras.Model):
    def __init__(self, layers_hidden, grid_min=-1, grid_max=1, num_grids=10, use_base_update=False, base_activation='relu', spline_weight_init_scale=1):
        super(FastKAN, self).__init__()
        self.layers_list = []
        for in_dim, out_dim in zip(layers_hidden[:-1], layers_hidden[1:]):
            self.layers_list.append(FastKANLayer(in_dim, out_dim, grid_min, grid_max, num_grids, use_base_update, base_activation, spline_weight_init_scale))

    def call(self, inputs):
        x = inputs
        for layer in self.layers_list:
            x = layer(x)
        return x