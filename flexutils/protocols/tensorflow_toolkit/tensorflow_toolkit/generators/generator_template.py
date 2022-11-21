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
import h5py
import mrcfile
import os

import tensorflow as tf
from tensorflow.keras import backend as K
import tensorflow_addons as tfa

from tensorflow_toolkit.utils import getXmippOrigin


class DataGeneratorBase(tf.keras.utils.Sequence):
    def __init__(self, h5_file, batch_size=32, shuffle=True, step=1, splitTrain=0.8, keepMap=False):
        # Attributes
        self.step = step
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.indexes = np.arange(self.batch_size)

        # Read metadata
        mask, volume = self.readH5Metadata(h5_file)

        # Are images fitted in memory?
        stack = os.path.join(self.images_path[0], "stack.mrc")
        if os.path.isfile(stack):
            with mrcfile.open(stack) as mrc:
                self.mrc = mrc.data
            self.fitInMemory = True
        else:
            self.fitInMemory = False

        # Read volume data
        self.readVolumeData(mask, volume, keepMap)

        # Get train dataset
        # self.getTrainDataset(splitTrain)

        # Prepare CTF
        self.ctf = np.zeros([self.batch_size, self.xsize, int(0.5 * self.xsize + 1)])

        # Prepare alignment
        self.r = [np.zeros([self.batch_size, 3]), np.zeros([self.batch_size, 3]), np.zeros([self.batch_size, 3])]


    #----- Initialization methods -----#

    def readH5Metadata(self, h5_file):
        hf = h5py.File(h5_file, 'r')
        images_path = np.array(hf.get('images_path'))
        self.images_path = [str(n) for n in images_path.astype(str)]
        mask = np.array(hf.get('mask'))
        mask = [str(n) for n in mask.astype(str)]
        volume = np.array(hf.get('volume'))
        volume = [str(n) for n in volume.astype(str)]
        self.angle_rot = tf.constant(np.asarray(hf.get('angle_rot')), dtype=tf.float32)
        self.angle_tilt = tf.constant(np.asarray(hf.get('angle_tilt')), dtype=tf.float32)
        self.angle_psi = tf.constant(np.asarray(hf.get('angle_psi')), dtype=tf.float32)
        shift_x = np.asarray(hf.get('shift_x'))
        self.file_idx = np.arange(shift_x.size)
        self.shift_x = tf.constant(shift_x, dtype=tf.float32)
        self.shift_y = tf.constant(np.asarray(hf.get('shift_y')), dtype=tf.float32)
        self.shift_z = tf.constant(np.zeros(self.shift_x.shape), dtype=tf.float32)
        self.shifts = [self.shift_x, self.shift_y, self.shift_z]
        self.defocusU = tf.constant(np.asarray(hf.get('defocusU')), dtype=tf.float32)
        self.defocusV = tf.constant(np.asarray(hf.get('defocusV')), dtype=tf.float32)
        self.defocusAngle = tf.constant(np.asarray(hf.get('defocusAngle')), dtype=tf.float32)
        self.cs = tf.constant(np.asarray(hf.get('cs')), dtype=tf.float32)
        self.kv = tf.constant(hf.get('kv')[0], dtype=tf.float32)
        self.sr = tf.constant(np.asarray(hf.get('sr')), dtype=tf.float32)
        self.applyCTF = np.asarray(hf.get('applyCTF'))
        hf.close()

        return mask, volume

    def readVolumeData(self, mask, volume, keepMap=False):
        with mrcfile.open(mask[0]) as mrc:
            self.xsize = mrc.data.shape[0]
            self.xmipp_origin = getXmippOrigin(mrc.data)
            coords = np.asarray(np.where(mrc.data == 1))
            coords = np.transpose(np.asarray([coords[2, :], coords[1, :], coords[0, :]]))
            self.coords = coords - self.xmipp_origin

            if keepMap:
                self.mask_map = mrc.data

        # Apply step to coords and values
        coords = coords[::self.step]
        self.coords = self.coords[::self.step]

        with mrcfile.open(volume[0]) as mrc:
            self.values = mrc.data[coords[:, 2], coords[:, 1], coords[:, 0]]

            if keepMap:
                self.vol = mrc.data

    def getTrainDataset(self, splitTrain):
        indexes = np.arange(self.file_idx.size)
        np.random.shuffle(indexes)
        self.file_idx = self.file_idx[indexes[:int(splitTrain * indexes.size)]]

    # ----- -------- -----#


    # ----- Data generation methods -----#

    def on_epoch_end(self):
        if self.shuffle == True:
            indexes = np.arange(self.file_idx.size)
            np.random.shuffle(indexes)
            self.file_idx = self.file_idx[indexes]

    def __data_generation(self):
        if self.fitInMemory:
            images = self.mrc[self.indexes, :, :]
        else:
            images = []
            for index in self.indexes:
                with mrcfile.open(os.path.join(self.images_path[0], "theo_%d.mrc" % index)) as mrc:
                    images.append(mrc.data)
            images = np.vstack(images)

        # self.current_img = images
        return images.reshape(-1, self.xsize, self.xsize, 1), self.indexes

    def __getitem__(self, index):
        # Generate indexes of the batch
        self.indexes = self.file_idx[index * self.batch_size:(index + 1) * self.batch_size]
        # Generate data
        X, y = self.__data_generation()
        return X, y

    def __len__(self):
        return int(np.floor(len(self.file_idx) / self.batch_size))

    # ----- -------- -----#


    # ----- Utils -----#

    def gaussianFilterImage(self, images):
        images = tfa.image.gaussian_filter2d(images, 2 * self.step, self.step)
        return images

    def ctfFilterImage(self, images):
        ft_images = tf.signal.fftshift(tf.signal.rfft2d(images[:, :, :, 0]))
        ft_ctf_images_real = tf.multiply(tf.math.real(ft_images), self.ctf)
        ft_ctf_images = tf.complex(ft_ctf_images_real, tf.math.imag(ft_images))
        ctf_images = tf.signal.irfft2d(tf.signal.ifftshift(ft_ctf_images))
        return tf.reshape(ctf_images, [self.batch_size, self.xsize, self.xsize, 1])

    # ----- -------- -----#


    # ----- Losses -----#

    def loss_correlation(self, y_true, y_pred):
        return self.correlation_coefficient_loss(y_true, y_pred)

    def correlation_coefficient_loss(self, x, y):
        epsilon = 10e-5
        mx = K.mean(x)
        my = K.mean(y)
        xm, ym = x - mx, y - my
        r_num = K.sum(xm * ym)
        x_square_sum = K.sum(xm * xm)
        y_square_sum = K.sum(ym * ym)
        r_den = K.sqrt(x_square_sum * y_square_sum)
        r = r_num / (r_den + epsilon)
        return 1 - K.mean(r)

    # ----- -------- -----#