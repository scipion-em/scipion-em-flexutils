#!/usr/bin/env python
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


import os
import numpy as np
import h5py

import tensorflow as tf

from tensorflow_toolkit.generators.generator_zernike3deep import Generator
from tensorflow_toolkit.networks.zernike3deep import AutoEncoder

# # os.environ["CUDA_VISIBLE_DEVICES"]="0,2,3,4"
# physical_devices = tf.config.list_physical_devices('GPU')
# for gpu_instance in physical_devices:
#     tf.config.experimental.set_memory_growth(gpu_instance, True)


def predict(h5_file, weigths_file, L1, L2, refinePose, architecture, ctfType, pad=2):
    # Create data generator
    generator = Generator(L1, L2, h5_file=h5_file, shuffle=True, batch_size=32,
                          step=1, splitTrain=1.0, refinePose=refinePose, pad_factor=pad)

    # Load model
    autoencoder = AutoEncoder(generator, architecture=architecture, CTF=ctfType)
    autoencoder.load_weights(weigths_file).expect_partial()

    # Get Zernike3DSpace
    # zernike_space = []
    # delta_euler = []
    # delta_shifts = []

    # Predict step
    print("------------------ Predicting Zernike3D coefficients... ------------------")
    encoded = autoencoder.predict(generator)

    # Get encoded data in right format
    zernike_space = np.hstack([encoded[0], encoded[1], encoded[2]])

    if refinePose:
        delta_euler = encoded[3]
        delta_shifts = encoded[4]

    # for data in generator:
    #     encoded = autoencoder.encoder(data[0])
    #
    #     zernike_vec = np.hstack([encoded[0].numpy(), encoded[1].numpy(), encoded[2].numpy()])
    #     zernike_space.append(zernike_vec)
    #
    #     if refinePose:
    #         delta_euler.append(encoded[3].numpy())
    #         delta_shifts.append(encoded[4].numpy())
    #
    # zernike_space = np.vstack(zernike_space)

    # Save space to metadata file
    with h5py.File(h5_file, 'a') as hf:
        hf.create_dataset('zernike_space', data=zernike_space)

        if refinePose:
            delta_euler = np.vstack(delta_euler)
            delta_shifts = np.vstack(delta_shifts)

            hf.create_dataset('delta_angle_rot', data=delta_euler[:, 0])
            hf.create_dataset('delta_angle_tilt', data=delta_euler[:, 1])
            hf.create_dataset('delta_angle_psi', data=delta_euler[:, 2])
            hf.create_dataset('delta_shift_x', data=delta_shifts[:, 0])
            hf.create_dataset('delta_shift_y', data=delta_shifts[:, 1])


if __name__ == '__main__':
    import argparse

    # Input parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--h5_file', type=str, required=True)
    parser.add_argument('--weigths_file', type=str, required=True)
    parser.add_argument('--L1', type=int, required=True)
    parser.add_argument('--L2', type=int, required=True)
    parser.add_argument('--refine_pose', action='store_true')
    parser.add_argument('--architecture', type=str, required=True)
    parser.add_argument('--ctf_type', type=str, required=True)
    parser.add_argument('--pad', type=int, required=False, default=2)
    parser.add_argument('--gpu', type=str)

    args = parser.parse_args()

    if args.gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    physical_devices = tf.config.list_physical_devices('GPU')
    for gpu_instance in physical_devices:
        tf.config.experimental.set_memory_growth(gpu_instance, True)

    inputs = {"h5_file": args.h5_file, "weigths_file": args.weigths_file,
              "L1": args.L1, "L2": args.L2, "refinePose": args.refine_pose,
              "architecture": args.architecture, "ctfType": args.ctf_type,
              "pad": args.pad}

    # Initialize volume slicer
    predict(**inputs)
