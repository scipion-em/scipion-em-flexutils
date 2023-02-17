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
from tensorflow.keras import backend as K

from tensorflow_toolkit.generators.generator_template import DataGeneratorBase
from tensorflow_toolkit.utils import basisDegreeVectors, computeBasis, euler_matrix_batch, gramSchmidt, \
                                     fft_pad, ifft_pad


class Generator(DataGeneratorBase):
    def __init__(self, refinePose=False, **kwargs):
        super().__init__(**kwargs)

        self.refinePose = refinePose

        # Initialize pose information
        if refinePose:
            self.rot_batch = np.zeros(self.batch_size)
            self.tilt_batch = np.zeros(self.batch_size)
            self.psi_batch = np.zeros(self.batch_size)

        # Cost function
        cost = kwargs.get("cost")
        if cost == "corr":
            self.cost_function = self.correlation_coefficient_loss
        elif cost == "corr-fpc":
            self.cost_function = self.corr_fpc


    # ----- Utils -----#

    def applyAlignmentMatrix(self, q, axis):
        # Get rotation matrix
        A = gramSchmidt(q)

        # Get coords to move
        c_x = tf.constant(self.coords[:, 0], dtype=tf.float32)
        c_y = tf.constant(self.coords[:, 1], dtype=tf.float32)
        c_z = tf.constant(self.coords[:, 2], dtype=tf.float32)

        # Apply alignment
        c_r_1 = tf.multiply(c_x[None, :], A[:, axis, 0][:, None])
        c_r_2 = tf.multiply(c_y[None, :], A[:, axis, 1][:, None])
        c_r_3 = tf.multiply(c_z[None, :], A[:, axis, 2][:, None])
        return tf.add(tf.add(c_r_1, c_r_2), c_r_3)

    def applyAlignmentEuler(self, q, axis):
        # Get rotation matrix
        A = euler_matrix_batch(self.rot_batch + q[:, 0],
                               self.tilt_batch + q[:, 1],
                               self.psi_batch + q[:, 2])
        A = tf.stack(A, axis=1)

        # Get coords to move
        c_x = tf.constant(self.coords[:, 0], dtype=tf.float32)
        c_y = tf.constant(self.coords[:, 1], dtype=tf.float32)
        c_z = tf.constant(self.coords[:, 2], dtype=tf.float32)

        # Apply alignment
        c_r_1 = tf.multiply(c_x[None, :], A[:, axis, 0][:, None])
        c_r_2 = tf.multiply(c_y[None, :], A[:, axis, 1][:, None])
        c_r_3 = tf.multiply(c_z[None, :], A[:, axis, 2][:, None])
        return tf.add(tf.add(c_r_1, c_r_2), c_r_3)

    def applyShifts(self, c, axis):
        shifts_batch = tf.gather(self.shifts[axis], self.indexes, axis=0) + c[1][:, axis]
        return tf.add(tf.subtract(c[0], shifts_batch[:, None]), self.xmipp_origin[axis])

    def scatterImgByPass(self, c):
        # Get current batch size (function scope)
        batch_size_scope = tf.shape(c[0])[0]

        c_x = tf.reshape(c[0], [batch_size_scope, -1, 1])
        c_y = tf.reshape(c[1], [batch_size_scope, -1, 1])
        c_sampling = tf.concat([c_y, c_x], axis=2)

        imgs = tf.zeros((batch_size_scope, self.xsize, self.xsize), dtype=tf.float32)

        bamp = tf.constant(self.values, dtype=tf.float32)

        bposf = tf.floor(c_sampling)
        bposi = tf.cast(bposf, tf.int32)
        bposf = c_sampling - bposf

        # Bilinear interpolation to provide forward mapping gradients
        bamp0 = bamp[None, :] * (1.0 - bposf[:, :, 0]) * (1.0 - bposf[:, :, 1])  # 0,0
        bamp1 = bamp[None, :] * (bposf[:, :, 0]) * (1.0 - bposf[:, :, 1])  # 1,0
        bamp2 = bamp[None, :] * (bposf[:, :, 0]) * (bposf[:, :, 1])  # 1,1
        bamp3 = bamp[None, :] * (1.0 - bposf[:, :, 0]) * (bposf[:, :, 1])  #
        bampall = tf.concat([bamp0, bamp1, bamp2, bamp3], axis=1)
        bposall = tf.concat([bposi, bposi + (1, 0), bposi + (1, 1), bposi + (0, 1)], 1)
        # images = tf.stack([tf.tensor_scatter_nd_add(imgs[i], bposall[i], bampall[i]) for i in range(imgs.shape[0])])

        fn = lambda inp: tf.tensor_scatter_nd_add(inp[0], inp[1], inp[2])
        images = tf.map_fn(fn, [imgs, bposall, bampall], fn_output_signature=tf.float32)
        # images = tf.vectorized_map(fn, [imgs, bposall, bampall])


        images = tf.reshape(images, [-1, self.xsize, self.xsize, 1])

        return images

    # def centerMassShift(self):
    #     coords_o = tf.constant(self.coords, dtype=tf.float32)
    #     coords_o_x = tf.transpose(tf.gather(coords_o, 0, axis=1))
    #     coords_o_y = tf.transpose(tf.gather(coords_o, 1, axis=1))
    #     coords_o_z = tf.transpose(tf.gather(coords_o, 2, axis=1))
    #
    #     diff_x = self.def_coords[0] - coords_o_x[:, None]
    #     diff_y = self.def_coords[1] - coords_o_y[:, None]
    #     diff_z = self.def_coords[2] - coords_o_z[:, None]
    #
    #     mean_diff_x = tf.reduce_mean(diff_x)
    #     mean_diff_y = tf.reduce_mean(diff_y)
    #     mean_diff_z = tf.reduce_mean(diff_z)
    #
    #     cm = tf.sqrt(mean_diff_x * mean_diff_x + mean_diff_y * mean_diff_y + mean_diff_z * mean_diff_z)
    #
    #     return cm
    #     # return tf.keras.activations.relu(cm, threshold=0.2)

    def fourier_phase_correlation(self, x, y):
        x = tf.signal.fftshift(tf.signal.rfft2d(x[:, :, :, 0]))
        y = tf.signal.fftshift(tf.signal.rfft2d(y[:, :, :, 0]))

        # Sizes
        # pad_size = tf.constant(int(self.pad_factor * self.xsize), dtype=tf.int32)

        # x = fft_pad(x, pad_size, pad_size)
        # y = fft_pad(y, pad_size, pad_size)

        # In case we want to exclude some high (noisy) frequencies from the cost (using hard or
        # soft circular masks in Fourier space)
        # x = self.applyFourierMask(x)
        # y = self.applyFourierMask(y)

        epsilon = 10e-5
        num = tf.abs(tf.reduce_sum(x * tf.math.conj(y), axis=(1, 2)))
        d_1 = tf.reduce_sum(tf.abs(x) ** 2, axis=(1, 2))
        d_2 = tf.reduce_sum(tf.abs(y) ** 2, axis=(1, 2))
        den = tf.sqrt(d_1 * d_2)
        cross_power_spectrum = num / (den + epsilon)

        loss = 1 - cross_power_spectrum

        return loss[:, None]

    @tf.function()
    def correlation_coefficient_loss(self, x, y):
        epsilon = 10e-5
        mx = K.mean(x, axis=[1, 2, 3])[:, None, None, None]
        my = K.mean(y, axis=[1, 2, 3])[:, None, None, None]
        xm, ym = x - mx, y - my
        r_num = K.sum(xm * ym, axis=[1, 2, 3])
        x_square_sum = K.sum(xm * xm, axis=[1, 2, 3])
        y_square_sum = K.sum(ym * ym, axis=[1, 2, 3])
        r_den = K.sqrt(x_square_sum * y_square_sum)
        r = r_num / (r_den + epsilon)

        loss = 1 - r

        return loss[:, None]

    def corr_fpc(self, x, y):
        return tf.sqrt(self.correlation_coefficient_loss(x, y)
                       * self.fourier_phase_correlation(x, y))

    def mse(self, x, y):
        # Sizes
        pad_size = tf.constant(int(self.pad_factor * self.xsize), dtype=tf.int32)

        x = fft_pad(x, pad_size, pad_size)
        y = fft_pad(y, pad_size, pad_size)

        real = (tf.math.real(x) - tf.math.real(y)) ** 2
        imag = (tf.math.imag(x) - tf.math.imag(y)) ** 2

        loss = tf.reduce_mean(real + imag, axis=[1, 2])

        return loss[:, None]

    # ----- -------- -----#
