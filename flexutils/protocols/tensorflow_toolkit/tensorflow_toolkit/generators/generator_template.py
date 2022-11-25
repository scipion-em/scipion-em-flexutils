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
from scipy.ndimage import gaussian_filter
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
        # self.cap_def = 3.

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

        # Prepare circular mask
        # radius_mask = 0.85 * 0.5 * self.xsize
        # circular_mask = self.create_circular_mask(self.xsize, self.xsize, radius=radius_mask)
        # circular_mask = tf.constant(circular_mask, dtype=tf.float32)
        # circular_mask = tf.signal.ifftshift(circular_mask[None, :, :])
        # size = int(0.5 * self.xsize + 1)
        # circular_mask = tf.signal.fftshift(circular_mask[:, :, :size])
        # self.circular_mask = circular_mask[0, :, :]

        # Cap deformation vals
        # self.inv_sqrt_N = tf.constant(1. / np.sqrt(self.coords.shape[0]), dtype=tf.float32)
        # self.inv_bs = tf.constant(1. / self.batch_size, dtype=tf.float32)

        # Fourier rings
        self.getFourierRings()
        # self.radial_masks, self.spatial_freq = self.get_radial_masks()

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

    def create_circular_mask(self, h, w, center=None, radius=None):

        if center is None:  # use the middle of the image
            center = (int(w / 2), int(h / 2))
        if radius is None:  # use the smallest distance between the center and image walls
            radius = min(center[0], center[1], w - center[0], h - center[1])

        Y, X = np.ogrid[:h, :w]
        dist_from_center = np.sqrt((X - center[0]) ** 2 + (Y - center[1]) ** 2)

        mask = (dist_from_center <= radius).astype(np.float32)
        mask = gaussian_filter(mask, sigma=2.)

        return mask

    def applyFourierMask(self, images):
        ft_images = tf.signal.fftshift(tf.signal.rfft2d(images[:, :, :, 0]))
        ft_masked_images_real = tf.multiply(tf.math.real(ft_images), self.circular_mask[None, :, :])
        ft_masked_images = tf.complex(ft_masked_images_real, tf.math.imag(ft_images))
        masked_images = tf.signal.irfft2d(tf.signal.ifftshift(ft_masked_images))
        return tf.reshape(masked_images, [self.batch_size, self.xsize, self.xsize, 1])

    # def capDeformation(self, d_x, d_y, d_z):
    #     num = tf.sqrt(tf.reduce_sum(d_x * d_x + d_y * d_y + d_z * d_z))
    #     rmsdef = self.inv_sqrt_N * self.inv_bs * num
    #     return tf.keras.activations.relu(rmsdef, threshold=self.cap_def)
    #     # return tf.math.pow(1000., rmsdef - self.cap_def)

    def getFourierRings(self):
        idx = np.indices((self.xsize, self.xsize)) - self.xsize // 2
        idx = np.fft.fftshift(idx)
        idx = idx[:, :, :self.xsize // 2 + 1]

        # self.idxft = (idx / self.xsize).astype(np.float32)
        # self.rrft = np.sqrt(np.sum(idx ** 2, axis=0)).astype(np.float32)  ## batch, npts, x-y

        rr = np.round(np.sqrt(np.sum(idx ** 2, axis=0))).astype(int)
        self.rings = np.zeros((self.xsize, self.xsize // 2 + 1, self.xsize // 2), dtype=np.float32)  #### Fourier rings
        for i in range(self.xsize // 2):
            self.rings[:, :, i] = (rr == i)

        # self.xvec = tf.constant(np.fromfunction(lambda i, j: 1.0 - 2 * ((i + j) % 2),
        #                                         (self.xsize, self.xsize // 2 + 1), dtype=np.float32))

    def radial_mask(self, r, cx=64, cy=64, sx=np.arange(0, 128), sy=np.arange(0, 128), delta=1):
        ind = (sx[np.newaxis, :] - cx) ** 2 + (sy[:, np.newaxis] - cy) ** 2
        ind1 = ind <= ((r[0] + delta) ** 2)  # one liner for this and below?
        ind2 = ind > (r[0] ** 2)
        return ind1 * ind2

    @tf.function
    def get_radial_masks(self):
        bxsize = self.xsize
        half_bxsize = self.xsize // 2
        freq_nyq = int(np.floor(int(bxsize) / 2.0))
        radii = np.arange(half_bxsize).reshape(half_bxsize, 1)  # image size 256, binning = 3
        radial_masks = np.apply_along_axis(self.radial_mask, 1, radii, half_bxsize, half_bxsize,
                                           np.arange(0, bxsize), np.arange(0, bxsize), 1)
        radial_masks = np.expand_dims(radial_masks, 1)
        radial_masks = np.expand_dims(radial_masks, 1)

        spatial_freq = radii.astype(np.float32) / freq_nyq
        spatial_freq = spatial_freq / max(spatial_freq)

        return radial_masks, spatial_freq

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

    def frc_loss(self, y_true, y_pred, minpx=1, maxpx=-1):
        y_true = tf.signal.rfft2d(y_true[:, :, :, 0])
        y_pred = tf.signal.rfft2d(y_pred[:, :, :, 0])
        mreal, mimag = tf.math.real(y_pred), tf.math.imag(y_pred)
        dreal, dimag = tf.math.real(y_true), tf.math.imag(y_true)
        #### normalization per ring
        nrm_img = mreal ** 2 + mimag ** 2
        nrm_data = dreal ** 2 + dimag ** 2

        nrm0 = tf.tensordot(nrm_img, self.rings, [[1, 2], [0, 1]])
        nrm1 = tf.tensordot(nrm_data, self.rings, [[1, 2], [0, 1]])

        nrm = tf.sqrt(nrm0) * tf.sqrt(nrm1)
        nrm = tf.maximum(nrm, 1e-4)  #### so we do not divide by 0

        #### average FRC per batch
        ccc = mreal * dreal + mimag * dimag
        frc = tf.tensordot(ccc, self.rings, [[1, 2], [0, 1]]) / nrm

        frcval = tf.reduce_mean(frc[:, minpx:maxpx], axis=1)
        return frcval

    # @tf.function
    # def fourier_ring_correlation(self, image1, image2, rn, spatial_freq):
    #     # we need the channels first format for this loss
    #     image1 = tf.transpose(image1, perm=[0, 3, 1, 2])
    #     image2 = tf.transpose(image2, perm=[0, 3, 1, 2])
    #     image1 = tf.cast(image1, tf.complex64)
    #     image2 = tf.cast(image2, tf.complex64)
    #     rn = tf.cast(rn, tf.complex64)
    #     fft_image1 = tf.signal.fftshift(tf.signal.fft2d(image1), axes=[2, 3])
    #     fft_image2 = tf.signal.fftshift(tf.signal.fft2d(image2), axes=[2, 3])
    #
    #     t1 = tf.multiply(fft_image1, rn)  # (128, BS?, 3, 256, 256)
    #     t2 = tf.multiply(fft_image2, rn)
    #     c1 = tf.math.real(tf.reduce_sum(tf.multiply(t1, tf.math.conj(t2)), [2, 3, 4]))
    #     c2 = tf.reduce_sum(tf.math.abs(t1) ** 2, [2, 3, 4])
    #     c3 = tf.reduce_sum(tf.math.abs(t2) ** 2, [2, 3, 4])
    #     frc = tf.math.divide(c1, tf.math.sqrt(tf.math.multiply(c2, c3)))
    #     frc = tf.where(tf.compat.v1.is_inf(frc), tf.zeros_like(frc), frc)  # inf
    #     frc = tf.where(tf.compat.v1.is_nan(frc), tf.zeros_like(frc), frc)  # nan
    #
    #     t = spatial_freq
    #     y = frc
    #     riemann_sum = tf.reduce_sum(tf.multiply(t[1:] - t[:-1], (y[:-1] + y[1:]) / 2.), 0)
    #     return riemann_sum
    #
    # @tf.function
    # def compute_loss_frc(self, y_true, y_pred):
    #     loss = -self.fourier_ring_correlation(y_true, y_pred, self.radial_masks, self.spatial_freq)
    #
    #     loss = tf.math.reduce_mean(loss)
    #     return loss

    # ----- -------- -----#