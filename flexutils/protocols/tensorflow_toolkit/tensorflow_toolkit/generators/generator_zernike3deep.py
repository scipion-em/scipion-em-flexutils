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

from tensorflow_toolkit.generators.generator_template import DataGeneratorBase
from tensorflow_toolkit.utils import basisDegreeVectors, computeBasis, euler_matrix_batch


class Generator(DataGeneratorBase):
    def __init__(self, L1=3, L2=2, refinePose=True, **kwargs):
        super().__init__(**kwargs)

        self.refinePose = refinePose

        # Get Zernike3D vector size
        self.zernike_size = basisDegreeVectors(L1, L2)

        # Precompute Zernike3D basis
        self.Z = computeBasis(self.coords, L1=L1, L2=L2, r=0.5 * self.xsize)

        # Initialize pose information
        if refinePose:
            self.rot_batch = np.zeros(self.batch_size)
            self.tilt_batch = np.zeros(self.batch_size)
            self.psi_batch = np.zeros(self.batch_size)


    # ----- Utils -----#

    def computeDeformationField(self, z):
        Z = tf.constant(self.Z, dtype=tf.float32)
        d = tf.matmul(Z, tf.transpose(z))
        return d

    def applyDeformationField(self, d, axis):
        coords = tf.constant(self.coords, dtype=tf.float32)
        coords_axis = tf.transpose(tf.gather(coords, axis, axis=1))
        return tf.add(coords_axis[:, None], d)

    def applyAlignmentMatrix(self, c, axis):
        c_r_1 = tf.multiply(c[0], tf.cast(tf.gather(self.r[axis], 0, axis=1), dtype=tf.float32))
        c_r_2 = tf.multiply(c[1], tf.cast(tf.gather(self.r[axis], 1, axis=1), dtype=tf.float32))
        c_r_3 = tf.multiply(c[2], tf.cast(tf.gather(self.r[axis], 2, axis=1), dtype=tf.float32))
        return tf.add(tf.add(c_r_1, c_r_2), c_r_3)

    def applyAlignmentDeltaEuler(self, inputs, axis):
        r = euler_matrix_batch(self.rot_batch + inputs[3][:, 0],
                               self.tilt_batch + inputs[3][:, 1],
                               self.psi_batch + inputs[3][:, 2])

        c_r_1 = tf.multiply(inputs[0], tf.cast(tf.gather(r[axis], 0, axis=1), dtype=tf.float32))
        c_r_2 = tf.multiply(inputs[1], tf.cast(tf.gather(r[axis], 1, axis=1), dtype=tf.float32))
        c_r_3 = tf.multiply(inputs[2], tf.cast(tf.gather(r[axis], 2, axis=1), dtype=tf.float32))
        return tf.add(tf.add(c_r_1, c_r_2), c_r_3)

    def applyShifts(self, c, axis):
        shifts_batch = tf.gather(self.shifts[axis], self.indexes, axis=0)
        return tf.add(tf.subtract(c, shifts_batch[None, :]), self.xmipp_origin[axis])

    def applyDeltaShifts(self, c, axis):
        shifts_batch = tf.gather(self.shifts[axis], self.indexes, axis=0) + c[1][:, axis]
        return tf.add(tf.subtract(c[0], shifts_batch[None, :]), self.xmipp_origin[axis])

    def scatterImgByPass(self, c):
        c_x = tf.reshape(tf.transpose(c[0]), [self.batch_size, -1, 1])
        c_y = tf.reshape(tf.transpose(c[1]), [self.batch_size, -1, 1])
        c_sampling = tf.concat([c_y, c_x], axis=2)

        imgs = tf.zeros((self.batch_size, self.xsize, self.xsize), dtype=tf.float32)

        bamp = tf.constant(self.values, dtype=tf.float32)

        bposf = tf.floor(c_sampling)
        bposi = tf.cast(bposf, tf.int32)
        bposf = c_sampling - bposf

        # Bilinear interpolation to provide forward mapping gradients
        bamp0 = bamp[None, :] * (1.0 - bposf[:, :, 0]) * (1.0 - bposf[:, :, 1])
        bamp1 = bamp[None, :] * (bposf[:, :, 0]) * (1.0 - bposf[:, :, 1])
        bamp2 = bamp[None, :] * (bposf[:, :, 0]) * (bposf[:, :, 1])
        bamp3 = bamp[None, :] * (1.0 - bposf[:, :, 0]) * (bposf[:, :, 1])
        bampall = tf.concat([bamp0, bamp1, bamp2, bamp3], axis=1)
        bposall = tf.concat([bposi, bposi + (1, 0), bposi + (1, 1), bposi + (0, 1)], 1)
        images = tf.stack([tf.tensor_scatter_nd_add(imgs[i], bposall[i], bampall[i]) for i in range(imgs.shape[0])])

        images = tf.reshape(images, [-1, self.xsize, self.xsize, 1])

        return images

    def centerMassShift(self):
        coords_o = tf.constant(self.coords, dtype=tf.float32)
        coords_o_x = tf.transpose(tf.gather(coords_o, 0, axis=1))
        coords_o_y = tf.transpose(tf.gather(coords_o, 1, axis=1))
        coords_o_z = tf.transpose(tf.gather(coords_o, 2, axis=1))

        diff_x = self.def_coords[0] - coords_o_x[:, None]
        diff_y = self.def_coords[1] - coords_o_y[:, None]
        diff_z = self.def_coords[2] - coords_o_z[:, None]

        mean_diff_x = tf.reduce_mean(diff_x)
        mean_diff_y = tf.reduce_mean(diff_y)
        mean_diff_z = tf.reduce_mean(diff_z)

        cm = tf.sqrt(mean_diff_x * mean_diff_x + mean_diff_y * mean_diff_y + mean_diff_z * mean_diff_z)

        return cm
        # return tf.keras.activations.relu(cm, threshold=0.2)

    def averageDeformation(self):
        d_x = self.def_coords[0]
        d_y = self.def_coords[1]
        d_z = self.def_coords[2]

        rmsdef = tf.reduce_mean(tf.sqrt(tf.reduce_mean(d_x * d_x + d_y * d_y + d_z * d_z, axis=0)))

        return rmsdef

    # ----- -------- -----#
