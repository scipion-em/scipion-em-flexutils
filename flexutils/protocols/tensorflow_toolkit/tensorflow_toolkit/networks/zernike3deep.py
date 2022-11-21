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


import tensorflow as tf
from tensorflow.python.keras import Input
from tensorflow.python.keras.models import Model
from tensorflow.keras import layers

from tensorflow_toolkit.utils import euler_matrix_row, computeCTF


class Encoder(Model):
  def __init__(self, latent_dim, input_dim):
    super(Encoder, self).__init__()
    self.latent_dim = latent_dim

    encoder_inputs = Input(shape=(input_dim, input_dim, 1))
    x = layers.Flatten()(encoder_inputs)
    for _ in range(12):
        x = layers.Dense(1024, activation='relu')(x)

    z_space_x = layers.Dense(latent_dim, activation="linear", name="z_space_x")(x)
    z_space_y = layers.Dense(latent_dim, activation="linear", name="z_space_y")(x)
    z_space_z = layers.Dense(latent_dim, activation="linear", name="z_space_z")(x)
    self.encoder = Model(encoder_inputs, [z_space_x, z_space_y, z_space_z], name="encoder")

  def call(self, x):
    z_space_x, z_space_y, z_space_z = self.encoder(x)
    return z_space_x, z_space_y, z_space_z


class Decoder(Model):
  def __init__(self, latent_dim, generator):
    super(Decoder, self).__init__()
    self.generator = generator

    decoder_inputs_x = Input(shape=(latent_dim,))
    decoder_inputs_y = Input(shape=(latent_dim,))
    decoder_inputs_z = Input(shape=(latent_dim,))

    # Compute deformation field
    d_x = layers.Lambda(self.generator.computeDeformationField, trainable=False)(decoder_inputs_x)
    d_y = layers.Lambda(self.generator.computeDeformationField, trainable=False)(decoder_inputs_y)
    d_z = layers.Lambda(self.generator.computeDeformationField, trainable=False)(decoder_inputs_z)

    # Apply deformation field
    c_x = layers.Lambda(lambda inp: self.generator.applyDeformationField(inp, 0), trainable=False)(d_x)
    c_y = layers.Lambda(lambda inp: self.generator.applyDeformationField(inp, 1), trainable=False)(d_y)
    c_z = layers.Lambda(lambda inp: self.generator.applyDeformationField(inp, 2), trainable=False)(d_z)

    # Apply alignment
    c_r_x = layers.Lambda(lambda inp: self.generator.applyAlignment(inp, 0), trainable=False)([c_x, c_y, c_z])
    c_r_y = layers.Lambda(lambda inp: self.generator.applyAlignment(inp, 1), trainable=False)([c_x, c_y, c_z])

    # Apply shifts
    c_r_s_x = layers.Lambda(lambda inp: self.generator.applyShifts(inp, 0), trainable=False)(c_r_x)
    c_r_s_y = layers.Lambda(lambda inp: self.generator.applyShifts(inp, 1), trainable=False)(c_r_y)

    # Scatter image and bypass gradient
    scatter_images = layers.Lambda(self.generator.scatterImgByPass, trainable=False)([c_r_s_x, c_r_s_y])

    # Gaussian filter image
    decoded = layers.Lambda(self.generator.gaussianFilterImage)(scatter_images)

    # CTF filter image
    decoded_ctf = layers.Lambda(self.generator.ctfFilterImage)(decoded)

    self.decoder = Model([decoder_inputs_x, decoder_inputs_y, decoder_inputs_z], decoded_ctf, name="decoder")

  def call(self, x):
    decoded = self.decoder(x)
    return decoded

class AutoEncoder(Model):
    def __init__(self, generator, **kwargs):
        super(AutoEncoder, self).__init__(**kwargs)
        self.generator = generator
        self.encoder = Encoder(generator.zernike_size.shape[0], generator.xsize)
        self.decoder = Decoder(generator.zernike_size.shape[0], generator)
        self.total_loss_tracker = tf.keras.metrics.Mean(name="total_loss")

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
        ]

    def train_step(self, data):
        self.decoder.generator.indexes = data[1]
        self.decoder.generator.current_images = data[0]

        # Precompute batch aligments
        rot_batch = tf.gather(self.decoder.generator.angle_rot, data[1], axis=0)
        tilt_batch = tf.gather(self.decoder.generator.angle_tilt, data[1], axis=0)
        psi_batch = tf.gather(self.decoder.generator.angle_psi, data[1], axis=0)
        row_1 = euler_matrix_row(rot_batch, tilt_batch, psi_batch, 1, self.decoder.generator.batch_size)
        row_2 = euler_matrix_row(rot_batch, tilt_batch, psi_batch, 2, self.decoder.generator.batch_size)
        row_3 = euler_matrix_row(rot_batch, tilt_batch, psi_batch, 3, self.decoder.generator.batch_size)
        self.decoder.generator.r = [row_1, row_2, row_3]

        # Precompute batch CTFs
        defocusU_batch = tf.gather(self.decoder.generator.defocusU, data[1], axis=0)
        defocusV_batch = tf.gather(self.decoder.generator.defocusV, data[1], axis=0)
        defocusAngle_batch = tf.gather(self.decoder.generator.defocusAngle, data[1], axis=0)
        cs_batch = tf.gather(self.decoder.generator.cs, data[1], axis=0)
        kv_batch = self.decoder.generator.kv
        ctf = computeCTF(defocusU_batch, defocusV_batch, defocusAngle_batch, cs_batch, kv_batch, self.decoder.generator.sr,
                         [self.decoder.generator.xsize, int(0.5 * self.decoder.generator.xsize + 1)],
                         self.decoder.generator.batch_size, self.decoder.generator.applyCTF)
        self.decoder.generator.ctf = ctf

        with tf.GradientTape() as tape:
            z_space_x, z_space_y, z_space_z = self.encoder(data[0])
            decoded = self.decoder([z_space_x, z_space_y, z_space_z])

            total_loss = self.generator.loss_correlation(data[0], decoded)

        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        return {
            "loss": self.total_loss_tracker.result(),
        }

    def call(self, input_features):
        z_x, z_y, z_z = self.encoder(input_features)
        decoded = self.decoder([z_x, z_y, z_z])
        return decoded
