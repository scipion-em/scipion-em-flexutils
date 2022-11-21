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

from flexutils_tensorflow.generators.generator_template import DataGeneratorBase
from flexutils_tensorflow.utils import basisDegreeVectors, computeBasis


class Generator(DataGeneratorBase):
    def __init__(self, L1=3, L2=2, **kwargs):
        super().__init__(**kwargs)

        # Get Zernike3D vector size
        self.zernike_size = basisDegreeVectors(L1, L2)

        # Precompute Zernike3D basis
        self.Z = computeBasis(self.coords, L1=L1, L2=L2, r=0.5 * self.xsize)


    # ----- Utils -----#

    def computeDeformationField(self, z):
        Z = tf.constant(self.Z, dtype=tf.float32)
        d = tf.matmul(Z, tf.transpose(z))
        return d

    def applyDeformationField(self, d, axis):
        coords = tf.constant(self.coords, dtype=tf.float32)
        coords_axis = tf.transpose(tf.gather(coords, axis, axis=1))
        return tf.add(coords_axis[:, None], d)

    def applyAlignment(self, c, axis):
        c_r_1 = tf.multiply(c[0], tf.cast(tf.gather(self.r[axis], 0, axis=1), dtype=tf.float32))
        c_r_2 = tf.multiply(c[1], tf.cast(tf.gather(self.r[axis], 1, axis=1), dtype=tf.float32))
        c_r_3 = tf.multiply(c[2], tf.cast(tf.gather(self.r[axis], 2, axis=1), dtype=tf.float32))
        return tf.add(tf.add(c_r_1, c_r_2), c_r_3)

    def applyShifts(self, c, axis):
        shifts_batch = tf.gather(self.shifts[axis], self.indexes, axis=0)
        return tf.add(tf.subtract(c, shifts_batch[None, :]), self.xmipp_origin[axis])

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
        bamp0 = bamp[None, :] * (1.0 - bposf[:, :, 0]) * (1.0 - bposf[:, :, 1])  # 0,0
        bamp1 = bamp[None, :] * (bposf[:, :, 0]) * (1.0 - bposf[:, :, 1])  # 1,0
        bamp2 = bamp[None, :] * (bposf[:, :, 0]) * (bposf[:, :, 1])  # 1,1
        bamp3 = bamp[None, :] * (1.0 - bposf[:, :, 0]) * (bposf[:, :, 1])  #
        bampall = tf.concat([bamp0, bamp1, bamp2, bamp3], axis=1)
        bposall = tf.concat([bposi, bposi + (1, 0), bposi + (1, 1), bposi + (0, 1)], 1)
        images = tf.stack([tf.tensor_scatter_nd_add(imgs[i], bposall[i], bampall[i]) for i in range(imgs.shape[0])])

        images = tf.reshape(images, [-1, self.xsize, self.xsize, 1])

        return images

    # ----- -------- -----#
