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
from tensorflow.keras import layers

from tensorflow_toolkit.utils import euler_matrix_row, computeCTF, euler_matrix_batch
from tensorflow_toolkit.layers.residue_conv2d import ResidueConv2D


class Encoder(tf.keras.Model):
  def __init__(self, latent_dim, input_dim, refinePose, architecture="convnn"):
    super(Encoder, self).__init__()
    self.latent_dim = latent_dim
    l2 = tf.keras.regularizers.l2(1e-3)
    # shift_activation = lambda y: 2 * tf.keras.activations.tanh(y)

    encoder_inputs = tf.keras.Input(shape=(input_dim, input_dim, 1))

    x = tf.keras.layers.Flatten()(encoder_inputs)

    if architecture == "mlpnn":
        for _ in range(12):
            x = layers.Dense(1024, activation='relu', kernel_regularizer=l2)(x)
        x = layers.Dropout(0.3)(x)
        x = layers.BatchNormalization()(x)

    elif architecture == "convnn":
        for _ in range(3):
            x = layers.Dense(64 * 64, activation='relu', kernel_regularizer=l2)(x)

        x = tf.keras.layers.Dense(64 * 64, kernel_regularizer=l2)(x)
        x = tf.keras.layers.Reshape((64, 64, 1))(x)

        x = tf.keras.layers.Conv2D(64, 5, activation="relu", strides=(2, 2), padding="same")(x)
        for _ in range(1):
            x = ResidueConv2D(64, 5, activation="relu", padding="same")(x)
        x = tf.keras.layers.Conv2D(32, 5, activation="relu", strides=(2, 2), padding="same")(x)
        for _ in range(1):
            x = ResidueConv2D(32, 5, activation="relu", padding="same")(x)
        x = tf.keras.layers.Conv2D(16, 3, activation="relu", strides=(2, 2), padding="same")(x)
        x = ResidueConv2D(16, 3, activation="relu", padding="same")(x)
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dropout(.1)(x)
        x = tf.keras.layers.BatchNormalization()(x)

        for _ in range(3):
            x = layers.Dense(3 * 16 * 16, activation='relu', kernel_regularizer=l2)(x)
        x = layers.Dropout(.1)(x)
        x = layers.BatchNormalization()(x)

        # x = tf.keras.layers.Conv2D(4, 5, activation="relu", strides=(2, 2), padding="same")(x)
        # x = tf.keras.layers.Conv2D(8, 5, activation="relu", strides=(2, 2), padding="same")(x)
        # x = tf.keras.layers.Conv2D(16, 3, activation="relu", strides=(2, 2), padding="same")(x)
        # x = tf.keras.layers.Conv2D(16, 3, activation="relu", strides=(2, 2), padding="same")(x)
        # x = tf.keras.layers.Flatten()(x)
        # x = tf.keras.layers.Dropout(.1)(x)
        # x = tf.keras.layers.BatchNormalization()(x)

    if refinePose:
        z_space_x = layers.Dense(latent_dim, activation="linear", name="z_space_x")(x)
        z_space_y = layers.Dense(latent_dim, activation="linear", name="z_space_y")(x)
        z_space_z = layers.Dense(latent_dim, activation="linear", name="z_space_z")(x)
        delta_euler = layers.Dense(3, activation="linear", name="delta_euler")(x)
        # delta_shifts = layers.Dense(2, activation=shift_activation, name="delta_shifts")(x)
        delta_shifts = layers.Dense(2, activation="linear", name="delta_shifts")(x)
        self.encoder = tf.keras.Model(encoder_inputs,
                                      [z_space_x, z_space_y, z_space_z, delta_euler, delta_shifts], name="encoder")
    else:
        z_space_x = layers.Dense(latent_dim, activation="linear", name="z_space_x")(x)
        z_space_y = layers.Dense(latent_dim, activation="linear", name="z_space_y")(x)
        z_space_z = layers.Dense(latent_dim, activation="linear", name="z_space_z")(x)
        self.encoder = tf.keras.Model(encoder_inputs, [z_space_x, z_space_y, z_space_z], name="encoder")

  def call(self, x):
    return self.encoder(x)


class Decoder(tf.keras.Model):
  def __init__(self, latent_dim, generator):
    super(Decoder, self).__init__()
    self.generator = generator

    decoder_inputs_x = tf.keras.Input(shape=(latent_dim,))
    decoder_inputs_y = tf.keras.Input(shape=(latent_dim,))
    decoder_inputs_z = tf.keras.Input(shape=(latent_dim,))
    if self.generator.refinePose:
        delta_euler = tf.keras.Input(shape=(3,))
        delta_shifts = tf.keras.Input(shape=(2,))

    # Compute deformation field
    d_x = layers.Lambda(self.generator.computeDeformationField, trainable=True)(decoder_inputs_x)
    d_y = layers.Lambda(self.generator.computeDeformationField, trainable=True)(decoder_inputs_y)
    d_z = layers.Lambda(self.generator.computeDeformationField, trainable=True)(decoder_inputs_z)

    # Apply deformation field
    c_x = layers.Lambda(lambda inp: self.generator.applyDeformationField(inp, 0), trainable=True)(d_x)
    c_y = layers.Lambda(lambda inp: self.generator.applyDeformationField(inp, 1), trainable=True)(d_y)
    c_z = layers.Lambda(lambda inp: self.generator.applyDeformationField(inp, 2), trainable=True)(d_z)

    # Apply alignment
    if self.generator.refinePose:
        c_r_x = layers.Lambda(lambda inp: self.generator.applyAlignmentDeltaEuler(inp, 0), trainable=True)\
                ([c_x, c_y, c_z, delta_euler])
        c_r_y = layers.Lambda(lambda inp: self.generator.applyAlignmentDeltaEuler(inp, 1), trainable=True)\
                ([c_x, c_y, c_z, delta_euler])
    else:
        c_r_x = layers.Lambda(lambda inp: self.generator.applyAlignmentMatrix(inp, 0), trainable=True)([c_x, c_y, c_z])
        c_r_y = layers.Lambda(lambda inp: self.generator.applyAlignmentMatrix(inp, 1), trainable=True)([c_x, c_y, c_z])

    # Apply shifts
    if self.generator.refinePose:
        c_r_s_x = layers.Lambda(lambda inp: self.generator.applyDeltaShifts(inp, 0), trainable=True)\
                  ([c_r_x, delta_shifts])
        c_r_s_y = layers.Lambda(lambda inp: self.generator.applyDeltaShifts(inp, 1), trainable=True)\
                  ([c_r_y, delta_shifts])
    else:
        c_r_s_x = layers.Lambda(lambda inp: self.generator.applyShifts(inp, 0), trainable=True)(c_r_x)
        c_r_s_y = layers.Lambda(lambda inp: self.generator.applyShifts(inp, 1), trainable=True)(c_r_y)

    # Scatter image and bypass gradient
    scatter_images = layers.Lambda(self.generator.scatterImgByPass, trainable=True)([c_r_s_x, c_r_s_y])

    if self.generator.step > 1 or self.generator.ref_is_struct:
        # Gaussian filter image
        decoded = layers.Lambda(self.generator.gaussianFilterImage)(scatter_images)

        # CTF filter image
        decoded_ctf = layers.Lambda(self.generator.ctfFilterImage)(decoded)
    else:
        # CTF filter image
        decoded_ctf = layers.Lambda(self.generator.ctfFilterImage)(scatter_images)

    if self.generator.refinePose:
        self.decoder = tf.keras.Model([decoder_inputs_x, decoder_inputs_y, decoder_inputs_z,
                                       delta_euler, delta_shifts], decoded_ctf, name="decoder")
    else:
        self.decoder = tf.keras.Model([decoder_inputs_x, decoder_inputs_y, decoder_inputs_z],
                                       decoded_ctf, name="decoder")

  def call(self, x):
    decoded = self.decoder(x)
    # self.generator.def_coords = [d_x, d_y, d_z]
    return decoded

class AutoEncoder(tf.keras.Model):
    def __init__(self, generator, architecture="convnn", **kwargs):
        super(AutoEncoder, self).__init__(**kwargs)
        self.generator = generator
        self.encoder = Encoder(generator.zernike_size.shape[0], generator.xsize,
                               generator.refinePose, architecture=architecture)
        self.decoder = Decoder(generator.zernike_size.shape[0], generator)
        self.total_loss_tracker = tf.keras.metrics.Mean(name="total_loss")
        # self.img_loss_tracker = tf.keras.metrics.Mean(name="img_loss")
        # self.cap_def_loss_tracker = tf.keras.metrics.Mean(name="cap_def_loss")

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            # self.cap_def_loss_tracker,
            # self.img_loss_tracker,
        ]

    def train_step(self, data):
        self.decoder.generator.indexes = data[1]
        self.decoder.generator.current_images = data[0]

        # Precompute batch aligments
        if self.decoder.generator.refinePose:
            self.decoder.generator.rot_batch = tf.gather(self.decoder.generator.angle_rot, data[1], axis=0)
            self.decoder.generator.tilt_batch = tf.gather(self.decoder.generator.angle_tilt, data[1], axis=0)
            self.decoder.generator.psi_batch = tf.gather(self.decoder.generator.angle_psi, data[1], axis=0)
        else:
            rot_batch = tf.gather(self.decoder.generator.angle_rot, data[1], axis=0)
            tilt_batch = tf.gather(self.decoder.generator.angle_tilt, data[1], axis=0)
            psi_batch = tf.gather(self.decoder.generator.angle_psi, data[1], axis=0)
            # row_1 = euler_matrix_row(rot_batch, tilt_batch, psi_batch, 1, self.decoder.generator.batch_size)
            # row_2 = euler_matrix_row(rot_batch, tilt_batch, psi_batch, 2, self.decoder.generator.batch_size)
            # row_3 = euler_matrix_row(rot_batch, tilt_batch, psi_batch, 3, self.decoder.generator.batch_size)
            # self.decoder.generator.r = [row_1, row_2, row_3]
            self.decoder.generator.r = euler_matrix_batch(rot_batch, tilt_batch, psi_batch)

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
            encoded = self.encoder(data[0])
            decoded = self.decoder(encoded)

            # ori_images = self.decoder.generator.applyFourierMask(data[0])
            # decoded = self.decoder.generator.applyFourierMask(decoded)

            # cap_def_loss = self.decoder.generator.capDeformation(d_x, d_y, d_z)
            img_loss = self.decoder.generator.cost_function(data[0], decoded)

            total_loss = img_loss
            # 0.01 * self.decoder.generator.averageDeformation()  # Extra loss term to compensate large deformations
            # self.decoder.generator.centerMassShift()  # Extra loss term to center strucuture

        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        # self.img_loss_tracker.update_state(img_loss)
        # self.cap_def_loss_tracker.update_state(cap_def_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            # "img_loss": self.img_loss_tracker.result(),
            # "cap_def_loss": self.cap_def_loss_tracker.result()
        }

    def call(self, input_features):
        return self.decoder(self.encoder(input_features))
