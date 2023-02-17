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


# import numpy as np

import tensorflow as tf
from tensorflow.keras import layers
# import tensorflow_addons as tfa

from tensorflow_toolkit.utils import euler_matrix_row, computeCTF, euler_matrix_batch, gramSchmidt
# from tensorflow_toolkit.layers.filter_stack import FilterStack


class Encoder(tf.keras.Model):
    def __init__(self, input_dim, refine, architecture="convnn"):
        super(Encoder, self).__init__()
        l2 = tf.keras.regularizers.l2(1e-3)

        encoder_inputs = tf.keras.Input(shape=(input_dim, input_dim, 1))

        if architecture == "mlpnn":
            x = tf.keras.layers.Flatten()(encoder_inputs)
            for _ in range(12):
                x = layers.Dense(1024, activation='relu', kernel_regularizer=l2)(x)

            y = tf.keras.layers.Dense(1024, activation="relu", kernel_regularizer=l2)(x)
            y = tf.keras.layers.Dense(1024, activation="relu", kernel_regularizer=l2)(y)
            y = tf.keras.layers.Dense(1024, activation="relu", kernel_regularizer=l2)(y)
            y = layers.Dropout(0.3)(y)
            y = layers.BatchNormalization()(y)

            z = tf.keras.layers.Dense(1024, activation="relu", kernel_regularizer=l2)(x)
            z = tf.keras.layers.Dense(1024, activation="relu", kernel_regularizer=l2)(z)
            z = tf.keras.layers.Dense(1024, activation="relu", kernel_regularizer=l2)(z)
            z = layers.Dropout(0.3)(z)
            z = layers.BatchNormalization()(z)

        elif architecture == "convnn":
            x = tf.keras.layers.Conv2D(filters=64, kernel_size=(5, 5),
                                       padding="same", activation="relu", use_bias=False)(encoder_inputs)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2))(x)
            x = tf.keras.layers.Conv2D(filters=64, kernel_size=(5, 5),
                                       padding="same", activation="relu", use_bias=False)(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2))(x)

            x = tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3),
                                       padding="same", activation="relu", use_bias=True)(x)
            x = tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3),
                                       padding="same", activation="relu", use_bias=False)(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2))(x)
            x = tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3),
                                       padding="same", activation="relu", use_bias=True)(x)
            x = tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3),
                                       padding="same", activation="relu", use_bias=False)(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2))(x)

            x = tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3),
                                       padding="same", activation="relu", use_bias=False)(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2))(x)

            x = tf.keras.layers.Flatten()(x)

            x = tf.keras.layers.Dense(256, activation="relu")(x)
            x = tf.keras.layers.Dense(256, activation="relu")(x)
            x = tf.keras.layers.Dense(256, activation="relu")(x)
            x = tf.keras.layers.Dense(256, activation="relu")(x)

            y = tf.keras.layers.Dense(256, activation="relu")(x)
            y = tf.keras.layers.Dense(256, activation="relu")(y)
            y = tf.keras.layers.Dense(256, activation="relu")(y)
            y = tf.keras.layers.Dense(256, activation="relu")(y)

            z = tf.keras.layers.Dense(256, activation="relu")(x)
            z = tf.keras.layers.Dense(256, activation="relu")(z)
            z = tf.keras.layers.Dense(256, activation="relu")(z)
            z = tf.keras.layers.Dense(256, activation="relu")(z)

        alignnment = layers.Dense(3, activation="linear", name="alignnment")(y) if refine \
            else layers.Dense(6, activation="linear", name="alignnment")(y)
        shifts = layers.Dense(2, activation="linear", name="shifts")(z)
        self.encoder = tf.keras.Model(encoder_inputs, [alignnment, shifts], name="encoder")

    def call(self, x):
        return self.encoder(x)


class Decoder(tf.keras.Model):
    def __init__(self, generator, CTF="apply"):
        super(Decoder, self).__init__()
        self.generator = generator
        self.CTF = CTF

        alignnment = tf.keras.Input(shape=(3,)) if self.generator.refinePose else tf.keras.Input(shape=(6,))
        shifts = tf.keras.Input(shape=(2,))

        # Apply alignment
        if self.generator.refinePose:
            c_r_x = layers.Lambda(lambda inp: self.generator.applyAlignmentEuler(inp, 0), trainable=False)(alignnment)
            c_r_y = layers.Lambda(lambda inp: self.generator.applyAlignmentEuler(inp, 1), trainable=False)(alignnment)
        else:
            c_r_x = layers.Lambda(lambda inp: self.generator.applyAlignmentMatrix(inp, 0), trainable=False)(alignnment)
            c_r_y = layers.Lambda(lambda inp: self.generator.applyAlignmentMatrix(inp, 1), trainable=False)(alignnment)

        # Apply shifts
        c_r_s_x = layers.Lambda(lambda inp: self.generator.applyShifts(inp, 0), trainable=False)([c_r_x, shifts])
        c_r_s_y = layers.Lambda(lambda inp: self.generator.applyShifts(inp, 1), trainable=False)([c_r_y, shifts])

        # Scatter image and bypass gradient
        scatter_images = layers.Lambda(self.generator.scatterImgByPass, trainable=False)([c_r_s_x, c_r_s_y])

        if self.generator.step >= 1 or self.generator.ref_is_struct:
            # Gaussian filter image
            decoded = layers.Lambda(self.generator.gaussianFilterImage)(scatter_images)

            if self.CTF == "apply":
                # CTF filter image
                decoded_ctf = layers.Lambda(self.generator.ctfFilterImage)(decoded)
        else:
            if self.CTF == "apply":
                # CTF filter image
                decoded_ctf = layers.Lambda(self.generator.ctfFilterImage)(scatter_images)

        self.decoder = tf.keras.Model([alignnment, shifts], decoded_ctf, name="decoder")

    def call(self, x):
        decoded = self.decoder(x)
        return decoded


class AutoEncoder(tf.keras.Model):
    def __init__(self, generator, architecture="convnn", CTF="apply", n_gradients=1, **kwargs):
        super(AutoEncoder, self).__init__(**kwargs)
        self.generator = generator
        self.CTF = CTF
        self.encoder = Encoder(generator.xsize, generator.refinePose, architecture=architecture)
        self.decoder = Decoder(generator, CTF=CTF)
        self.total_loss_tracker = tf.keras.metrics.Mean(name="total_loss")

        self.n_gradients = tf.constant(n_gradients, dtype=tf.int32)
        self.n_acum_step = tf.Variable(0, dtype=tf.int32, trainable=False)
        self.gradient_accumulation = [tf.Variable(tf.zeros_like(v, dtype=tf.float32), trainable=False) for v in
                                      self.trainable_variables]

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
        ]

    def train_step(self, data):
        self.n_acum_step.assign_add(1)

        self.decoder.generator.indexes = data[1]
        self.decoder.generator.current_images = data[0]

        images = data[0]

        # Update batch_size (in case it is incomplete)
        batch_size_scope = tf.shape(data[0])[0]

        # Precompute batch alignments
        if self.decoder.generator.refinePose:
            self.decoder.generator.rot_batch = tf.gather(self.decoder.generator.angle_rot, data[1], axis=0)
            self.decoder.generator.tilt_batch = tf.gather(self.decoder.generator.angle_tilt, data[1], axis=0)
            self.decoder.generator.psi_batch = tf.gather(self.decoder.generator.angle_psi, data[1], axis=0)

        # Precompute batch CTFs
        defocusU_batch = tf.gather(self.decoder.generator.defocusU, data[1], axis=0)
        defocusV_batch = tf.gather(self.decoder.generator.defocusV, data[1], axis=0)
        defocusAngle_batch = tf.gather(self.decoder.generator.defocusAngle, data[1], axis=0)
        cs_batch = tf.gather(self.decoder.generator.cs, data[1], axis=0)
        kv_batch = self.decoder.generator.kv
        ctf = computeCTF(defocusU_batch, defocusV_batch, defocusAngle_batch, cs_batch, kv_batch,
                         self.decoder.generator.sr, self.decoder.generator.pad_factor,
                         [self.decoder.generator.xsize, int(0.5 * self.decoder.generator.xsize + 1)],
                         batch_size_scope, self.decoder.generator.applyCTF)
        self.decoder.generator.ctf = ctf

        if self.CTF == "wiener":
            images = self.decoder.generator.wiener2DFilter(images)

        # images_rot = tfa.image.rotate(images, np.pi, interpolation="bilinear")

        with tf.GradientTape() as tape:
            encoded = self.encoder(images)
            decoded = self.decoder(encoded)

            # encoded = self.encoder(images_rot)
            # decoded_rot = self.decoder(encoded)

            # loss_1 = self.decoder.generator.cost_function(images, decoded)
            # loss_2 = self.decoder.generator.cost_function(images_rot, decoded_rot)
            # total_loss = tf.reduce_min(tf.concat([loss_1, loss_2], axis=1), axis=1)
            total_loss = self.decoder.generator.cost_function(images, decoded)

        # Calculate batch gradients
        gradients = tape.gradient(total_loss, self.trainable_variables)

        # Accumulate batch gradients
        for i in range(len(self.gradient_accumulation)):
            self.gradient_accumulation[i].assign_add(gradients[i])

        # If n_acum_step reach the n_gradients then we apply accumulated gradients to update the variables otherwise do nothing
        tf.cond(tf.equal(self.n_acum_step, self.n_gradients), self.apply_accu_gradients, lambda: None)

        self.total_loss_tracker.update_state(total_loss)
        return {
            "loss": self.total_loss_tracker.result(),
        }

    def apply_accu_gradients(self):
        # apply accumulated gradients
        self.optimizer.apply_gradients(zip(self.gradient_accumulation, self.trainable_variables))

        # reset
        self.n_acum_step.assign(0)
        for i in range(len(self.gradient_accumulation)):
            self.gradient_accumulation[i].assign(tf.zeros_like(self.trainable_variables[i], dtype=tf.float32))

    def call(self, input_features):
        return self.decoder(self.encoder(input_features))

    def predict_step(self, data):
        self.decoder.generator.indexes = data[1]
        self.decoder.generator.current_images = data[0]

        images = data[0]

        # Update batch_size (in case it is incomplete)
        batch_size_scope = tf.shape(data[0])[0]

        # Precompute batch alignments
        if self.decoder.generator.refinePose:
            self.decoder.generator.rot_batch = tf.gather(self.decoder.generator.angle_rot, data[1],
                                                         axis=0)
            self.decoder.generator.tilt_batch = tf.gather(self.decoder.generator.angle_tilt, data[1],
                                                          axis=0)
            self.decoder.generator.psi_batch = tf.gather(self.decoder.generator.angle_psi, data[1],
                                                         axis=0)

        # Precompute batch CTFs
        defocusU_batch = tf.gather(self.decoder.generator.defocusU, data[1], axis=0)
        defocusV_batch = tf.gather(self.decoder.generator.defocusV, data[1], axis=0)
        defocusAngle_batch = tf.gather(self.decoder.generator.defocusAngle, data[1], axis=0)
        cs_batch = tf.gather(self.decoder.generator.cs, data[1], axis=0)
        kv_batch = self.decoder.generator.kv
        ctf = computeCTF(defocusU_batch, defocusV_batch, defocusAngle_batch, cs_batch, kv_batch,
                         self.decoder.generator.sr, self.decoder.generator.pad_factor,
                         [self.decoder.generator.xsize, int(0.5 * self.decoder.generator.xsize + 1)],
                         batch_size_scope, self.decoder.generator.applyCTF)
        self.decoder.generator.ctf = ctf

        if self.CTF == "wiener":
            images = self.decoder.generator.wiener2DFilter(images)

        # Get symmetric inputs
        # images_rot = tfa.image.rotate(images, np.pi, interpolation="bilinear")

        # Decode inputs
        encoded = self.encoder(images)

        return encoded

        # decoded = self.decoder(encoded)

        # Decode rotated inputs
        # encoded_rot = self.encoder(images_rot)
        # decoded_rot = self.decoder(encoded_rot)

        # Convert to matrix if ab initio
        # if not self.decoder.generator.refinePose:
        #     encoded[0] = gramSchmidt(encoded[0])
        #     encoded_rot[0] = gramSchmidt(encoded_rot[0])

        # Correct encoded rot
        # if self.decoder.generator.refinePose:
        #     encoded_rot[0][:, -1] += np.pi
        # else:
        #     A = euler_matrix_batch(tf.zeros([batch_size_scope], tf.float32),
        #                            tf.zeros([batch_size_scope], tf.float32),
        #                            tf.zeros([batch_size_scope], tf.float32) + 180.)
        #     A = tf.stack(A, axis=1)
        #     encoded_rot[0] = tf.matmul(A, encoded_rot[0])

        # Stack encoder outputs
        # pred_algn = tf.stack([encoded[0], encoded_rot[0]], axis=-1)
        # pred_shifts = tf.stack([encoded[1], encoded_rot[1]], axis=-1)

        # Compute losses between symmetric inputs and predictions
        # loss_1 = self.decoder.generator.cost_function(images, decoded)
        # loss_2 = self.decoder.generator.cost_function(images_rot, decoded_rot)
        # loss = tf.concat([loss_1, loss_2], axis=1)

        # Get only best predictions according to symmetric loss
        # index_min = tf.math.argmin(loss, axis=1)
        # pred_algn = pred_algn[tf.range(batch_size_scope, dtype=tf.int32), ..., index_min]
        # pred_shifts = pred_shifts[tf.range(batch_size_scope, dtype=tf.int32), ..., index_min]

        # return pred_algn, pred_shifts, loss
