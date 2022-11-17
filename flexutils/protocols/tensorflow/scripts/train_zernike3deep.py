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

from flexutils.protocols.tensorflow.generators.generator_zernike3deep import Generator
from flexutils.protocols.tensorflow.networks.zernike3deep import AutoEncoder

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)


def train(outPath, h5_file, L1, L2, batch_size, shuffle, step, splitTrain, epochs):
    # Create data generator
    generator = Generator(L1, L2, h5_file=h5_file, shuffle=shuffle, batch_size=batch_size,
                          step=step, splitTrain=splitTrain)

    # Train model
    autoencoder = AutoEncoder(generator)
    optimizer = tf.keras.optimizers.Adam(lr=1e-5)
    autoencoder.compile(optimizer=optimizer)
    autoencoder.fit(generator, epochs=epochs)

    # Save model
    autoencoder.save_weights(os.path.join(outPath, "zernike3deep_model"))

    # Get Zernike3DSpace
    if generator.fitInMemory:
        images = generator.mrc
        images = images.reshape(-1, generator.xsize, generator.xsize, 1)
        zernike_space = []
        for image in images:
            z_x, z_y, z_z = autoencoder.encoder(image[None, :, :, :])
            zernike_vec = np.hstack([z_x.numpy(), z_y.numpy(), z_z.numpy()])
            zernike_space.append(zernike_vec)
        zernike_space = np.vstack(zernike_space)
    else:
        zernike_space = []
        images_id = np.arange(generator.angle_rot.numpy().size)

        for index in images_id:
            with mrcfile.open(os.path.join(generator.images_path[0], "theo_%d.mrc" % index)) as mrc:
                image = mrc.data
            z_x, z_y, z_z = autoencoder.encoder(image[None, :, :, None])
            zernike_vec = np.hstack([z_x.numpy(), z_y.numpy(), z_z.numpy()])
            zernike_space.append(zernike_vec)
        zernike_space = np.vstack(zernike_space)

    # Save space to metadata file
    with h5py.File(h5_file, 'a') as hf:
        hf.create_dataset('zernike_space', data=zernike_space)


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

    args = parser.parse_args()

    inputs = {"h5_file": args.h5_file, "outPath": args.out_path, "L1": args.L1,
              "L2": args.L2, "batch_size": args.batch_size, "shuffle": args.shuffle,
              "step": args.step, "splitTrain": args.split_train, "epochs": args.epochs}

    # Initialize volume slicer
    train(**inputs)
