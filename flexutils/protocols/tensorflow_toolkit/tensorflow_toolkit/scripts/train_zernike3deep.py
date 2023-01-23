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
import mrcfile

import tensorflow as tf

from tensorflow_toolkit.generators.generator_zernike3deep import Generator
from tensorflow_toolkit.networks.zernike3deep import AutoEncoder

# # os.environ["CUDA_VISIBLE_DEVICES"]="0,2,3,4"
# physical_devices = tf.config.list_physical_devices('GPU')
# for gpu_instance in physical_devices:
#     tf.config.experimental.set_memory_growth(gpu_instance, True)


def train(outPath, h5_file, L1, L2, batch_size, shuffle, step, splitTrain, epochs, cost,
          radius_mask, smooth_mask, refinePose, architecture="convnn"):
    # Create data generator
    generator = Generator(L1, L2, h5_file=h5_file, shuffle=shuffle, batch_size=batch_size,
                          step=step, splitTrain=splitTrain, cost=cost, radius_mask=radius_mask,
                          smooth_mask=smooth_mask, refinePose=refinePose)

    # Train model
    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        autoencoder = AutoEncoder(generator, architecture=architecture)
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-5)
    autoencoder.compile(optimizer=optimizer)
    autoencoder.fit(generator, epochs=epochs)

    # Save model
    autoencoder.save_weights(os.path.join(outPath, "zernike3deep_model"))

    # Get Zernike3DSpace
    zernike_space = []
    delta_euler = []
    delta_shifts = []
    if generator.fitInMemory:
        images = generator.mrc
        # images = images.reshape(-1, generator.xsize, generator.xsize, 1)
        for image in images:
            encoded = autoencoder.encoder(image[None, :, :, None])
            zernike_vec = np.hstack([encoded[0].numpy(), encoded[1].numpy(), encoded[2].numpy()])
            zernike_space.append(zernike_vec)

            if refinePose:
                delta_euler.append(encoded[3].numpy())
                delta_shifts.append(encoded[4].numpy())
    else:
        images_id = np.arange(generator.angle_rot.numpy().size)

        for index in images_id:
            with mrcfile.open(os.path.join(generator.images_path[0], "theo_%d.mrc" % index)) as mrc:
                image = mrc.data
            encoded = autoencoder.encoder(image[None, :, :, None])
            zernike_vec = np.hstack([encoded[0].numpy(), encoded[1].numpy(), encoded[2].numpy()])
            zernike_space.append(zernike_vec)

            if refinePose:
                delta_euler.append(encoded[3].numpy())
                delta_shifts.append(encoded[4].numpy())

    zernike_space = np.vstack(zernike_space)

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
    parser.add_argument('--out_path', type=str, required=True)
    parser.add_argument('--L1', type=int, required=True)
    parser.add_argument('--L2', type=int, required=True)
    parser.add_argument('--batch_size', type=int, required=True)
    parser.add_argument('--shuffle', action='store_true')
    parser.add_argument('--step', type=int, required=True)
    parser.add_argument('--split_train', type=float, required=True)
    parser.add_argument('--epochs', type=int, required=True)
    parser.add_argument('--cost', type=str, required=True)
    parser.add_argument('--architecture', type=str, required=True)
    parser.add_argument('--radius_mask', type=float, required=False, default=2)
    parser.add_argument('--smooth_mask', action='store_true')
    parser.add_argument('--refine_pose', action='store_true')
    parser.add_argument('--gpu', type=str)

    args = parser.parse_args()

    if args.gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    physical_devices = tf.config.list_physical_devices('GPU')
    for gpu_instance in physical_devices:
        tf.config.experimental.set_memory_growth(gpu_instance, True)

    inputs = {"h5_file": args.h5_file, "outPath": args.out_path, "L1": args.L1,
              "L2": args.L2, "batch_size": args.batch_size, "shuffle": args.shuffle,
              "step": args.step, "splitTrain": args.split_train, "epochs": args.epochs,
              "cost": args.cost, "radius_mask": args.radius_mask, "smooth_mask": args.smooth_mask,
              "refinePose": args.refine_pose, "architecture": args.architecture}

    # Initialize volume slicer
    train(**inputs)
