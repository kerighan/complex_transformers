from tensorflow.keras import layers, activations, initializers
import tensorflow as tf
import numpy as np


def clone_initializer(initializer):
    if not isinstance(initializer, initializers.Initializer):
        return initializer
    config = initializer.get_config()
    return initializer.__class__.from_config(config)


class ComplexEncoder(layers.Layer):
    def __init__(
        self, intermediate_dim, num_heads, dropout=0,
        attention_width=None, operation="real", use_temperature=False,
        causal_mask=False,
        **kwargs
    ):
        super(ComplexEncoder, self).__init__(**kwargs)
        self.intermediate_dim = intermediate_dim
        self.dropout = dropout
        self.num_heads = num_heads
        self.operation = operation
        self.use_temperature = use_temperature
        self.attention_width = attention_width
        self.supports_masking = True
        self.causal_mask = causal_mask

    def build(self, input_shape):
        hidden_dim = input_shape[-1]
        head_dim = int(hidden_dim // self.num_heads)

        self.attention = MultiHeadComplexAttention(
            self.num_heads, key_dim=head_dim,
            attention_width=self.attention_width,
            operation=self.operation,
            use_temperature=self.use_temperature,
            causal_mask=self.causal_mask)
        self.attention_dropout = layers.Dropout(rate=self.dropout)

        self.feedforward_intermediate = layers.Dense(
            self.intermediate_dim, activation="relu",
            kernel_initializer=clone_initializer("glorot_uniform"))
        self.feedforward_output = layers.Dense(
            hidden_dim, kernel_initializer=clone_initializer("glorot_uniform"))
        self.feedforward_dropout = layers.Dropout(rate=self.dropout)

        self.layer_norm_1 = layers.LayerNormalization(epsilon=1e-5)
        self.layer_norm_2 = layers.LayerNormalization(epsilon=1e-5)

    def call(self, inputs, mask=None):
        attention_output = self.attention(inputs, mask=mask)
        attention_output = self.attention_dropout(attention_output)
        layer_norm_output = self.layer_norm_1(inputs + attention_output)

        ff_out_1 = self.feedforward_intermediate(layer_norm_output)
        ff_out_2 = self.feedforward_output(ff_out_1)
        ff_out_2 = self.feedforward_dropout(ff_out_2)

        layer_norm_2_output = self.layer_norm_2(
            ff_out_2 + layer_norm_output)
        return layer_norm_2_output

    def compute_mask(self, _, mask=None):
        return mask


class ComplexDecoder(layers.Layer):
    def __init__(
        self, intermediate_dim, num_heads, dropout=0,
        attention_width=None, operation="real", use_temperature=False,
        **kwargs
    ):
        super(ComplexDecoder, self).__init__(**kwargs)
        self.intermediate_dim = intermediate_dim
        self.dropout = dropout
        self.attention_width = attention_width
        self.num_heads = num_heads
        self.operation = operation
        self.use_temperature = use_temperature
        self.supports_masking = True

    def build(self, input_shape):
        hidden_dim = input_shape[-1]
        head_dim = int(hidden_dim // self.num_heads)

        self.attention = MultiHeadComplexAttention(
            self.num_heads, key_dim=head_dim, causal_mask=True,
            operation=self.operation, use_temperature=self.use_temperature,
            attention_width=self.attention_width)
        self.attention_dropout = layers.Dropout(rate=self.dropout)

        self.cross_attention = MultiHeadComplexAttention(
            self.num_heads, key_dim=head_dim, causal_mask=True,
            operation=self.operation, use_temperature=self.use_temperature,
            attention_width=self.attention_width)
        self.cross_attention_dropout = layers.Dropout(rate=self.dropout)

        self.feedforward_intermediate = layers.Dense(
            self.intermediate_dim, activation="relu",
            kernel_initializer=clone_initializer("glorot_uniform"))
        self.feedforward_output = layers.Dense(
            hidden_dim, activation=None,
            kernel_initializer=clone_initializer("glorot_uniform"))
        self.feedforward_dropout = layers.Dropout(rate=self.dropout)

        self.layer_norm_1 = layers.LayerNormalization(epsilon=1e-5)
        self.layer_norm_2 = layers.LayerNormalization(epsilon=1e-5)
        self.layer_norm_3 = layers.LayerNormalization(epsilon=1e-5)

        self.summarizer = ComplexAttentivePooling(
            self.intermediate_dim, self.num_heads)

    def call(
        self,
        inputs,
        encoder_inputs=None,
        mask=None
    ):
        if encoder_inputs is None:
            encoder_inputs = inputs

        attention_output = self.attention(inputs, mask=mask)
        attention_output = self.attention_dropout(attention_output)
        layer_norm_output = self.layer_norm_1(inputs + attention_output)

        cross_attention_output = self.cross_attention(
            inputs, value=encoder_inputs, mask=mask)
        cross_attention_output = self.cross_attention_dropout(
            cross_attention_output)
        layer_norm_2_output = self.layer_norm_2(
            cross_attention_output + layer_norm_output)

        ff_out_1 = self.feedforward_intermediate(layer_norm_2_output)
        ff_out_2 = self.feedforward_output(ff_out_1)
        ff_out_2 = self.feedforward_dropout(ff_out_2)

        layer_norm_3_output = self.layer_norm_3(
            ff_out_2 + layer_norm_2_output)
        return layer_norm_3_output

    def compute_mask(self, _, mask=None):
        return mask


class ComplexAttention(layers.Layer):
    def __init__(
        self,
        key_dim=None,
        value_dim=None,
        attention_width=None,
        use_bias=True,
        causal_mask=False,
        dropout=0.0,
        operation="real",
        use_temperature=False,
        use_position=False,
        **kwargs
    ):
        super(ComplexAttention, self).__init__(**kwargs)
        self.key_dim = key_dim
        self.value_dim = key_dim if value_dim is None else value_dim
        self.attention_width = attention_width
        self.use_bias = use_bias
        self.causal_mask = causal_mask
        self.dropout = dropout
        self.supports_masking = True
        self.operation = get_operation(operation)
        self.use_temperature = use_temperature
        self.use_position = use_position

    def build(self, input_shape):
        self.query = ComplexDense(self.key_dim, use_bias=self.use_bias)
        self.key = ComplexDense(self.key_dim, use_bias=self.use_bias)
        self.value = layers.Dense(
            self.value_dim, use_bias=self.use_bias,
            kernel_initializer=clone_initializer("glorot_uniform"))
        self.dropout = layers.Dropout(rate=self.dropout)

        self.temperature = self.add_weight(
            "temperature", shape=(1,),
            initializer="ones", trainable=self.use_temperature)

        seq_len = input_shape[1]
        self.seq_len = seq_len
        if self.causal_mask:
            self._causal_mask = tf.constant(
                np.tril(np.ones((self.seq_len, self.seq_len)))[None, :, :],
                dtype=tf.float32)

        if self.attention_width is not None:
            ind = tf.range(self.seq_len, dtype=tf.int32)
            self.attention_mask = tf.cast(
                tf.math.abs(ind[:, None] - ind) <= self.attention_width,
                dtype=tf.float32)

        if self.use_position:
            self.attention_interference = AttentionInterference(self.key_dim)

    def call(
        self, query, value=None, key=None,
        attention_interference=None,
        mask=None, training=None
    ):
        if value is None:
            value = query
        if key is None:
            key = value

        query = self.query(query)
        value = self.value(value)
        key = self.key(key)

        # apply scalar multiply
        query = tf.multiply(query, 1.0 / float(self.key_dim)**.5)

        # apply attention interference
        if self.use_position:
            if attention_interference is None:
                attention_interference = self.attention_interference(query)
            key += attention_interference[0]
            query += attention_interference[1]

        # compute attention scores
        interference = tf.matmul(query, key, transpose_b=True)
        interference /= self.key_dim**.5

        # attention weights are the real part of the dot product
        attention_weights = self.operation(interference)

        if mask is not None:
            mask = tf.expand_dims(tf.cast(mask, tf.float32), axis=-1)
            attention_weights -= 1e9 * (
                (1.0 - mask) * (1.0 - tf.keras.backend.permute_dimensions(
                    mask, (0, 2, 1))))
        if self.attention_width is not None:
            attention_weights = attention_weights * self.attention_mask
            attention_weights -= 1e9 * (1. - self.attention_mask)
        if self.causal_mask:
            attention_weights = attention_weights * self._causal_mask
            attention_weights -= 1e9 * (1. - self._causal_mask)

        attention_probs = tf.nn.softmax(
            attention_weights * self.temperature, axis=-1)
        attention_probs = self.dropout(attention_probs, training=training)

        # apply attention to the value
        output = tf.matmul(attention_probs, value)
        return output

    def compute_mask(self, _, mask=None):
        return mask


class ComplexAttention(layers.Layer):
    def __init__(
        self,
        key_dim=None,
        value_dim=None,
        attention_width=None,
        use_bias=True,
        causal_mask=False,
        dropout=0.0,
        operation="real",
        use_temperature=False,
        use_position=False,
        **kwargs
    ):
        super(ComplexAttention, self).__init__(**kwargs)
        self.key_dim = key_dim
        self.value_dim = key_dim if value_dim is None else value_dim
        self.attention_width = attention_width
        self.use_bias = use_bias
        self.causal_mask = causal_mask
        self.dropout = dropout
        self.supports_masking = True
        self.operation = get_operation(operation)
        self.use_temperature = use_temperature
        self.use_position = use_position

    def build(self, input_shape):
        self.query = ComplexDense(self.key_dim, use_bias=self.use_bias)
        self.key = ComplexDense(self.key_dim, use_bias=self.use_bias)
        self.value = layers.Dense(
            self.value_dim, use_bias=self.use_bias,
            kernel_initializer=clone_initializer("glorot_uniform"))
        self.dropout = layers.Dropout(rate=self.dropout)

        self.temperature = self.add_weight(
            "temperature", shape=(1,),
            initializer="ones", trainable=self.use_temperature)

        seq_len = input_shape[1]
        self.seq_len = seq_len
        if self.causal_mask:
            self._causal_mask = tf.constant(
                np.tril(np.ones((self.seq_len, self.seq_len)))[None, :, :],
                dtype=tf.float32)

        if self.attention_width is not None:
            ind = tf.range(self.seq_len, dtype=tf.int32)
            self.attention_mask = tf.cast(
                tf.math.abs(ind[:, None] - ind) <= self.attention_width,
                dtype=tf.float32)

        if self.use_position:
            self.attention_interference = AttentionInterference(self.key_dim)

    def call(
        self, query, value=None, key=None,
        attention_interference=None,
        mask=None, training=None
    ):
        if value is None:
            value = query
        if key is None:
            key = value

        query = self.query(query)
        value = self.value(value)
        key = self.key(key)

        # apply scalar multiply
        query = tf.multiply(query, 1.0 / float(self.key_dim)**.5)

        # apply attention interference
        if self.use_position:
            if attention_interference is None:
                attention_interference = self.attention_interference(query)
            key += attention_interference[0]
            query += attention_interference[1]

        # compute attention scores
        interference = tf.matmul(query, key, transpose_b=True)
        interference /= self.key_dim**.5

        # attention weights are the real part of the dot product
        attention_weights = self.operation(interference)

        if mask is not None:
            mask = tf.expand_dims(tf.cast(mask, tf.float32), axis=-1)
            attention_weights -= 1e9 * (
                (1.0 - mask) * (1.0 - tf.keras.backend.permute_dimensions(
                    mask, (0, 2, 1))))
        if self.attention_width is not None:
            attention_weights = attention_weights * self.attention_mask
            attention_weights -= 1e9 * (1. - self.attention_mask)
        if self.causal_mask:
            attention_weights = attention_weights * self._causal_mask
            attention_weights -= 1e9 * (1. - self._causal_mask)

        attention_probs = tf.nn.softmax(
            attention_weights * self.temperature, axis=-1)
        attention_probs = self.dropout(attention_probs, training=training)

        # apply attention to the value
        output = tf.matmul(attention_probs, value)
        return output

    def compute_mask(self, _, mask=None):
        return mask


class MultiHeadComplexAttention(layers.Layer):
    def __init__(
        self,
        num_heads,
        key_dim,
        value_dim=None,
        attention_width=None,
        use_bias=True,
        causal_mask=False,
        operation="real",
        use_temperature=False,
        use_position=True,
        **kwargs
    ):
        super(MultiHeadComplexAttention, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.value_dim = value_dim if value_dim is not None else key_dim
        self.attention_width = attention_width
        self.use_bias = use_bias
        self.causal_mask = causal_mask
        self.operation = operation
        self.use_temperature = use_temperature
        self.use_position = use_position
        self.supports_masking = True

    def build(self, input_shape):
        hidden_dim = input_shape[-1]
        self.dense = layers.Dense(
            hidden_dim, use_bias=self.use_bias,
            kernel_initializer=clone_initializer("glorot_uniform"))

        self.attentions = [
            ComplexAttention(key_dim=self.key_dim,
                             value_dim=self.value_dim,
                             attention_width=self.attention_width,
                             use_bias=self.use_bias,
                             causal_mask=self.causal_mask,
                             operation=self.operation,
                             use_temperature=self.use_temperature,
                             use_position=self.use_position,
                             name=f"attention_{i}")
            for i in range(self.num_heads)]
        for attention in self.attentions:
            self.trainable_weights.extend(attention.trainable_weights)

    def call(
        self, query, value=None, key=None,
        attention_interference=None,
        mask=None
    ):
        res = []
        for i in range(self.num_heads):
            attention_out = self.attentions[i](
                query, value, key, attention_interference, mask)
            res.append(attention_out)
        res = tf.concat(res, axis=-1)
        output = self.dense(res)
        return output

    def compute_mask(self, _, mask=None):
        return mask


class ComplexDense(layers.Layer):
    def __init__(
        self, units, activation=None,
        use_bias=True, use_polar=False,
        **kwargs
    ):
        super(ComplexDense, self).__init__(**kwargs)
        self.units = units
        self.use_polar = use_polar
        self.use_bias = use_bias
        self.activation = activation
        self.supports_masking = True

    def build(self, _):
        self.dense_1 = layers.Dense(
            self.units, use_bias=self.use_bias,
            activation=self.activation,
            kernel_initializer=clone_initializer("glorot_uniform"))
        self.dense_2 = layers.Dense(
            self.units, use_bias=self.use_bias,
            activation=self.activation,
            kernel_initializer=clone_initializer("glorot_uniform"))

    def call(self, inputs):
        if self.use_polar:
            modulus = tf.math.abs(self.dense_1(inputs))
            argument = np.pi * self.dense_2(inputs)
            complex_matrix = tf.complex(modulus * tf.cos(argument),
                                        modulus * tf.sin(argument))
        else:
            real_part = self.dense_1(inputs)
            imag_part = self.dense_2(inputs)
            complex_matrix = tf.complex(real_part, imag_part)
        return complex_matrix

    def compute_mask(self, _, mask=None):
        return mask


class AttentionInterference(layers.Layer):
    def __init__(
        self, units, activation=None, **kwargs
    ):
        super().__init__(**kwargs)
        self.activation = activations.get(activation)
        self.units = units
        self.supports_masking = True

    def build(self, input_shape):
        seq_len = input_shape[1]
        self.seq_len = seq_len
        self.dim = input_shape[-1]

        shape = (seq_len, self.units)
        self.query_real = self.add_weight(
            "query_real", shape=shape,
            initializer=clone_initializer("glorot_uniform"))
        self.query_imag = self.add_weight(
            "query_imag", shape=shape,
            initializer=clone_initializer("glorot_uniform"))
        self.key_real = self.add_weight(
            "key_real", shape=shape,
            initializer=clone_initializer("glorot_uniform"))
        self.key_imag = self.add_weight(
            "key_imag", shape=shape,
            initializer=clone_initializer("glorot_uniform"))

    def call(self, _, mask=None):
        key = tf.complex(self.key_real, self.key_imag)[None, :, :]
        query = tf.complex(self.query_real, self.query_imag)[None, :, :]
        key = self.activation(key)
        query = self.activation(query)
        return key, query

    def compute_mask(self, _, mask=None):
        return mask


class PositionalEmbedding(layers.Layer):
    def __init__(
        self, n_features, output_dim, mask_zero=False,
        positional_activation=None, **kwargs
    ):
        super(PositionalEmbedding, self).__init__(**kwargs)
        self.mask_zero = mask_zero
        self.n_features = n_features
        self.output_dim = output_dim
        self.supports_masking = True
        self.positional_activation = activations.get(positional_activation)

    def build(self, input_shape):
        seq_len = input_shape[1]
        self.embedding = tf.keras.layers.Embedding(
            input_dim=self.n_features,
            output_dim=self.output_dim,)
        self.positional_embedding = tf.keras.layers.Embedding(
            input_dim=seq_len,
            output_dim=self.output_dim,)

    def call(self, inputs):
        mask = None
        if self.mask_zero:
            mask = tf.expand_dims(tf.math.not_equal(inputs, 0), axis=-1)

        input_length = tf.shape(inputs)[1]

        position_indices = tf.range(input_length, dtype=tf.int32)[None, :]
        position_embeddings = self.positional_activation(
            self.positional_embedding(position_indices))

        x = position_embeddings + self.embedding(inputs)
        if self.mask_zero and mask is not None:
            x *= tf.cast(mask, tf.float32)
        return x

    def compute_mask(self, inputs, mask=None):
        if self.mask_zero:
            return tf.not_equal(inputs, 0)
        else:
            return None

    def get_config(self):
        config = super().get_config()
        config.update({
            "n_features": self.n_features,
            "output_dim": self.output_dim,
            "mask_zero": self.mask_zero,
            "positional_activation": activations.serialize(
                self.positional_activation)
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    @staticmethod
    def get_custom_objects():
        return {"PositionalEmbedding": PositionalEmbedding}


class PositionalEncoding(layers.Layer):
    def __init__(self, positional_activation=None, **kwargs):
        super(PositionalEncoding, self).__init__(**kwargs)
        self.positional_activation = activations.get(positional_activation)

    def build(self, input_shape):
        seq_len, dim = input_shape[1], input_shape[2]
        self.embedding = tf.keras.layers.Embedding(
            input_dim=seq_len, output_dim=dim)

    def call(self, inputs, mask=None):
        input_length = tf.shape(inputs)[1]
        position_indices = tf.range(
            input_length, dtype=tf.int32)[tf.newaxis, :]
        position_embeddings = self.embedding(position_indices)
        if mask is not None:
            print(mask)
        return position_embeddings + inputs

    def get_config(self):
        config = super().get_config()
        config.update({
            "positional_activation": activations.serialize(
                self.positional_activation)
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    @staticmethod
    def get_custom_objects():
        return {"PositionalEncoding": PositionalEncoding}


class ComplexAttentivePooling(layers.Layer):
    def __init__(
        self, num_heads, key_dim, value_dim=None, output_dim=None,
        operation="real",
        **kwargs
    ):
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.value_dim = value_dim if value_dim is not None else key_dim
        self.operation = get_operation(operation)
        self.output_dim = output_dim

    def build(self, input_shape):
        if self.output_dim is None:
            hidden_dim = input_shape[-1]
        else:
            hidden_dim = self.output_dim

        self.key_real = self.add_weight(
            name="key_real",
            shape=(self.num_heads, self.key_dim,),
            initializer=clone_initializer("glorot_uniform"))
        self.key_imag = self.add_weight(
            name="key_imag",
            shape=(self.num_heads, self.key_dim,),
            initializer=clone_initializer("glorot_uniform"))
        self.temperature = self.add_weight(
            name="temperature", shape=self.num_heads, initializer="ones")

        # self.query_dense = ComplexDense(self.key_dim, name="query_dense")
        self.query_dense = [
            ComplexDense(self.key_dim, name=f"query_dense_{i}")
            for i in range(self.num_heads)
        ]
        self.value_dense = [
            layers.Dense(
                self.value_dim,
                kernel_initializer=clone_initializer("glorot_uniform"),
                name=f"value_dense_{i}")
            for i in range(self.num_heads)
        ]
        self.feedforward = layers.Dense(
            hidden_dim, kernel_initializer=clone_initializer("glorot_uniform"),
            name="feedforward")
        self.layer_norm = layers.LayerNormalization(epsilon=1e-6)
        self.layer_norm_2 = layers.LayerNormalization(epsilon=1e-6)

    def call(self, inputs, mask=None):
        if mask is not None:
            mask = tf.expand_dims(tf.cast(mask, tf.float32), -1)

        res = []
        for i in range(self.num_heads):
            query = self.query_dense[i](inputs)
            key = tf.complex(self.key_real[i], self.key_imag[i])
            value = self.value_dense[i](inputs)

            query = tf.math.l2_normalize(query, axis=-1)
            key = tf.math.l2_normalize(key, axis=-1)

            score = tf.einsum("bij,j->bi", query, key)[:, :, None]
            score = self.operation(score) * mask

            # score = self.layer_norm_2(score)
            # score = tf.sinh(score * self.temperature[i]) * mask
            # score /= tf.reduce_sum(score, axis=1, keepdims=True) + 1e-5

            pooled_value = tf.reduce_sum(score * value, axis=1, keepdims=False)
            res.append(pooled_value)
        res = tf.concat(res, axis=-1)
        value = self.feedforward(res)
        # value = self.layer_norm(value)
        return value

    def get_config(self):
        config = super().get_config()
        config.update({
            "num_heads": self.num_heads,
            "key_dim": self.key_dim,
            "value_dim": self.value_dim,
            "output_dim": self.output_dim,
            "operation": self.operation
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    @staticmethod
    def get_custom_objects():
        return {"ComplexAttentivePooling": ComplexAttentivePooling}


def accuracy(y_true, y_pred):
    # Flatten the prediction and true labels
    y_true = tf.cast(tf.reshape(y_true, [-1]), tf.int64)
    y_pred = tf.reshape(tf.argmax(y_pred, axis=-1), [-1])

    # Create a mask of valid observations
    mask = tf.not_equal(y_true, 0)

    # Compute the accuracy
    acc = tf.cast(tf.equal(y_true[mask], y_pred[mask]), tf.float32)
    acc = tf.reduce_mean(acc)
    return acc


def get_operation(operation):
    if operation == "real":
        return tf.math.real
    elif operation == "imag":
        return tf.math.imag
    elif operation == "abs":
        return tf.math.abs
    elif callable(operation):
        return operation

    raise ValueError("Unknown operation: {}".format(operation))
