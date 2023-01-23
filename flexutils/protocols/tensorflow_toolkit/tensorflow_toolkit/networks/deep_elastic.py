# **************************************************************************
# *
# * Authors:  David Herreros Calero (dherreros@cnb.csic.es)
# *
# * Unidad de  Bioinformatica of Centro Nacional de Biotecnologia , CSIC
# *
# * This program is free software; you can redistribute it and/or modify
# * it under the terms of the GNU General Public License as published by
# * the Free Software Foundation; either version 2 of the License, or
# * (at your option) any later version.
# *
# * This program is distributed in the hope that it will be useful,
# * but WITHOUT ANY WARRANTY; without even the implied warranty of
# * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# * GNU General Public License for more details.
# *
# * You should have received a copy of the GNU General Public License
# * along with this program; if not, write to the Free Software
# * Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA
# * 02111-1307  USA
# *
# *  All comments concerning this program package may be sent to the
# *  e-mail address 'scipion@cnb.csic.es'
# *
# **************************************************************************


import numpy as np

import tensorflow as tf
from tensorflow.keras import layers, losses
from tensorflow.python.ops.linalg.linalg_impl import diag_part
from tensorflow.python.ops.math_ops import to_float
from tensorflow.python.ops.array_ops import shape_internal


# --------------------------------------------
# VAE Autoencoder
# --------------------------------------------

class Encoder(tf.keras.Model):
  def __init__(self, latent_dim, input_dim, sigma=1., train_log_var=True):
    super(Encoder, self).__init__()
    self.latent_dim = latent_dim

    encoder_inputs = tf.keras.Input(shape=(input_dim))
    x = layers.Dense(10 * latent_dim, activation='relu')(encoder_inputs)
    x = layers.Dense(5 * latent_dim, activation='relu')(x)
    x = layers.Dense(2 * latent_dim, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    x = layers.BatchNormalization()(x)
    z_mean = layers.Dense(latent_dim, activation=None, name="z_mean")(x)
    if train_log_var:
        z_log_var = layers.Dense(latent_dim, activation=None,
                                 kernel_initializer=tf.keras.initializers.TruncatedNormal(mean=0., stddev=sigma),
                                 name="z_log_var")(x)
        z_sampled = Sampling_Train_Var()([z_mean, z_log_var])
    else:
        z_log_var = layers.Dense(latent_dim, activation=None, name="z_log_var")(x)
        z_sampled = Sampling_Fix_Var(sigma=sigma)([z_mean, z_log_var])
    self.encoder = tf.keras.Model(encoder_inputs, [z_mean, z_log_var, z_sampled], name="encoder")

  def call(self, x):
    z_mean, z_log_var, z_sampled = self.encoder(x)
    return z_mean, z_log_var, z_sampled


class Decoder(tf.keras.Model):
  def __init__(self, latent_dim, final_dim):
    super(Decoder, self).__init__()
    self.latent_dim = latent_dim
    self.decoder = tf.keras.Sequential([
        layers.Dense(2 * latent_dim, activation='relu'),
        layers.Dense(5 * latent_dim, activation='relu'),
        layers.Dense(10 * latent_dim, activation='relu'),
        layers.Dropout(0.3),
        layers.BatchNormalization(),
        layers.Dense(final_dim, activation=None),
    ])

  def call(self, x):
    decoded = self.decoder(x)
    return decoded

class VAExplode(tf.keras.Model):
    def __init__(self, latent_dim, data_dim, kl_lambda=0., loss_lambda=0., dist_sigma=1., epsilon=1e-5,
                 train_log_var=True, **kwargs):
        super(VAExplode, self).__init__(**kwargs)
        self.encoder = Encoder(latent_dim, data_dim, dist_sigma, train_log_var)
        self.decoder = Decoder(latent_dim, data_dim)
        self.dist_sigma = dist_sigma
        self.train_log_var = train_log_var
        self.kl_lambda = kl_lambda if self.train_log_var else 0.0
        self.loss_lambda = loss_lambda
        self.epsilon = epsilon
        self.total_loss_tracker = tf.keras.metrics.Mean(name="total_loss")
        self.decoder_loss_tracker = tf.keras.metrics.Mean(
            name="decoder_loss"
        )
        self.encoder_loss_tracker = tf.keras.metrics.Mean(
            name="encoder_loss"
        )
        self.z_mean_loss_tracker = tf.keras.metrics.Mean(
            name="z_mean_loss"
        )
        self.kl_loss_tracker = tf.keras.metrics.Mean(name="kl_loss")

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.decoder_loss_tracker,
            self.kl_loss_tracker,
        ]

    def train_step(self, data):
        with tf.GradientTape() as tape:
            z_mean, z_log_var, encoded = self.encoder(data)
            decoded = self.decoder(encoded)

            z_mean_loss = encoder_loss(data, z_mean, self.epsilon)
            e_loss_ed, e_loss_cd = encoder_loss(data, encoded, self.epsilon)
            d_loss_ed, d_loss_cd = decoder_loss(data, decoded, self.epsilon)

            e_loss = e_loss_ed + self.loss_lambda * e_loss_cd
            d_loss = d_loss_ed + self.loss_lambda * d_loss_cd

            kl_loss = 0.0
            if self.kl_lambda > 0.0:
                kl_loss = np.log(self.dist_sigma) - 0.5 * z_log_var + ((tf.exp(z_log_var) + tf.square(z_mean))
                                                                       / (2 * np.square(self.dist_sigma))) - 0.5
                kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))

            total_loss = z_mean_loss + e_loss + d_loss + self.kl_lambda * kl_loss

        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.encoder_loss_tracker.update_state(e_loss)
        self.decoder_loss_tracker.update_state(d_loss)
        self.z_mean_loss_tracker.update_state(z_mean_loss)
        self.kl_loss_tracker.update_state(self.kl_lambda * kl_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "z_mean_loss": self.z_mean_loss_tracker.result(),
            "encoder_loss": self.encoder_loss_tracker.result(),
            "decoder_loss": self.decoder_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }

    def call(self, input_features):
        encoded = self.encoder(input_features)
        decoded = self.decoder(encoded)
        return decoded, encoded

class Sampling_Train_Var(tf.keras.layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

class Sampling_Fix_Var(tf.keras.layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""
    def __init__(self, sigma=1.):
        super(Sampling_Fix_Var, self).__init__()
        self.sigma = sigma

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.cast(self.sigma, dtype=tf.float32) * epsilon

def encoder_loss(y_true, y_pred, epsilon=1e-5):
  dist_mat_pred = _pairwise_distances(y_pred)
  dist_mat_true = _pairwise_distances(y_true)
  cos_mat_pred = _pairwise_cosine_distances(y_pred)
  cos_mat_true = _pairwise_cosine_distances(y_true)

  # mask_epsilon_distance = tf.multiply(tf.ones_like(dist_mat_pred), epsilon)
  mask_epsilon_distance = tf.fill(shape_internal(dist_mat_pred), tf.cast(epsilon, dtype=tf.float32))
  div_dist_mat_true = tf.multiply(dist_mat_true,
                                  tf.cast(tf.greater(dist_mat_true, mask_epsilon_distance), dtype=tf.float32))
  div_cos_mat_true = tf.multiply(cos_mat_true,
                                 tf.cast(tf.greater(cos_mat_true, mask_epsilon_distance), dtype=tf.float32))

  # Sammon mapping
  upper_mat_pred = tf.linalg.band_part(dist_mat_pred, 0, -1) - tf.linalg.band_part(dist_mat_pred, 0, 0)
  div_upper_mat_true = tf.linalg.band_part(div_dist_mat_true, 0, -1) - tf.linalg.band_part(div_dist_mat_true, 0, 0)

  aux_1 = tf.reduce_sum(tf.math.divide_no_nan(tf.square(div_upper_mat_true - upper_mat_pred), div_upper_mat_true))
  aux_2 = 1. / tf.reduce_sum(div_upper_mat_true)
  loss_1 = aux_2 * aux_1

  # Sammon mapping (cosine pairwise distances)
  upper_mat_pred = tf.linalg.band_part(cos_mat_pred, 0, -1) - tf.linalg.band_part(cos_mat_pred, 0, 0)
  div_upper_mat_true = tf.linalg.band_part(div_cos_mat_true, 0, -1) - tf.linalg.band_part(div_cos_mat_true, 0, 0)

  aux_1 = tf.reduce_sum(tf.math.divide_no_nan(tf.square(div_upper_mat_true - upper_mat_pred), div_upper_mat_true))
  aux_2 = 1. / tf.reduce_sum(div_upper_mat_true)
  loss_2 = aux_2 * aux_1

  return loss_1, loss_2

def decoder_loss(y_true, y_pred, epsilon=1e-5):
  dist_mat_pred = _pairwise_distances(y_pred)
  dist_mat_true = _pairwise_distances(y_true)
  cos_mat_pred = _pairwise_cosine_distances(y_pred)
  cos_mat_true = _pairwise_cosine_distances(y_true)

  # mask_epsilon_distance = tf.multiply(tf.ones_like(dist_mat_pred), epsilon)
  mask_epsilon_distance = tf.fill(shape_internal(dist_mat_pred), tf.cast(epsilon, dtype=tf.float32))
  div_dist_mat_true = tf.multiply(dist_mat_true,
                                  tf.cast(tf.greater(dist_mat_true, mask_epsilon_distance), dtype=tf.float32))
  div_cos_mat_true = tf.multiply(cos_mat_true,
                                 tf.cast(tf.greater(cos_mat_true, mask_epsilon_distance), dtype=tf.float32))

  # Sammon mapping (pairwise distances)
  upper_mat_pred = tf.linalg.band_part(dist_mat_pred, 0, -1) - tf.linalg.band_part(dist_mat_pred, 0, 0)
  div_upper_mat_true = tf.linalg.band_part(div_dist_mat_true, 0, -1) - tf.linalg.band_part(div_dist_mat_true, 0, 0)

  aux_1 = tf.reduce_sum(tf.math.divide_no_nan(tf.square(div_upper_mat_true - upper_mat_pred), div_upper_mat_true))
  aux_2 = 1. / tf.reduce_sum(div_upper_mat_true)
  loss_1 = aux_2 * aux_1

  # Sammon mapping (cosine pairwise distances)
  upper_mat_pred = tf.linalg.band_part(cos_mat_pred, 0, -1) - tf.linalg.band_part(cos_mat_pred, 0, 0)
  div_upper_mat_true = tf.linalg.band_part(div_cos_mat_true, 0, -1) - tf.linalg.band_part(div_cos_mat_true, 0, 0)

  aux_1 = tf.reduce_sum(tf.math.divide_no_nan(tf.square(div_upper_mat_true - upper_mat_pred), div_upper_mat_true))
  aux_2 = 1. / tf.reduce_sum(div_upper_mat_true)
  loss_2 = aux_2 * aux_1

  return loss_1, loss_2

# --------------------------------------------


# --------------------------------------------
# Utilities
# --------------------------------------------

def cluster_explosion(data, cluster, power):
    centers = cluster.cluster_centers_
    labels = cluster.labels_
    unique_labels = np.unique(labels)
    for idx in range(len(unique_labels)):
        ind = np.where(labels == unique_labels[idx])
        data[ind, :] += centers[idx, :] * power
    return data

def _pairwise_distances(embeddings, squared=False):
  """Compute the 2D matrix of distances between all the embeddings.
  Args:
      embeddings: tensor of shape (batch_size, embed_dim)
      squared: Boolean. If true, output is the pairwise squared euclidean distance matrix.
               If false, output is the pairwise euclidean distance matrix.
  Returns:
      pairwise_distances: tensor of shape (batch_size, batch_size)
  """
  # Get the dot product between all embeddings
  # shape (batch_size, batch_size)
  dot_product = tf.matmul(embeddings, tf.transpose(embeddings))

  # Get squared L2 norm for each embedding. We can just take the diagonal of `dot_product`.
  # This also provides more numerical stability (the diagonal of the result will be exactly 0).
  # shape (batch_size,)
  square_norm = diag_part(dot_product)

  # Compute the pairwise distance matrix as we have:
  # ||a - b||^2 = ||a||^2  - 2 <a, b> + ||b||^2
  # shape (batch_size, batch_size)
  distances = tf.expand_dims(square_norm, 1) - 2.0 * dot_product + tf.expand_dims(square_norm, 0)

  # Because of computation errors, some distances might be negative so we put everything >= 0.0
  distances = tf.maximum(distances, 0.0)

  if not squared:
    # Because the gradient of sqrt is infinite when distances == 0.0 (ex: on the diagonal)
    # we need to add a small epsilon where distances == 0.0
    mask = to_float(tf.equal(distances, 0.0))
    distances = distances + mask * 1e-16

    distances = tf.sqrt(distances)

    # Correct the epsilon added: set the distances on the mask to be exactly 0.0
    distances = distances * (1.0 - mask)

  return distances

def _pairwise_cosine_distances(a):
    # x shape is n_a * dim
    # y shape is n_b * dim
    # results shape is n_a * n_b

    normalize_a = tf.nn.l2_normalize(a, 1)
    # normalize_b = tf.nn.l2_normalize(b, 1)
    # distance = 1 - tf.matmul(normalize_a, normalize_b, transpose_b=True)
    distance = 1 - tf.matmul(normalize_a, normalize_a, transpose_b=True)
    return tf.abs(distance)

# --------------------------------------------