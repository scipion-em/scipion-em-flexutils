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
import multiprocessing
import tqdm

import tensorflow as tf
import tensorflow_addons as tfa

from tensorflow_toolkit.generators.generator_deep_pose import Generator
from tensorflow_toolkit.networks.deep_pose import AutoEncoder
from tensorflow_toolkit.utils import xmippEulerFromMatrix, computeCTF, euler_matrix_batch, gramSchmidt

# # os.environ["CUDA_VISIBLE_DEVICES"]="0,2,3,4"
# physical_devices = tf.config.list_physical_devices('GPU')
# for gpu_instance in physical_devices:
#     tf.config.experimental.set_memory_growth(gpu_instance, True)


def train(outPath, h5_file, batch_size, shuffle, step, splitTrain, epochs, cost,
          radius_mask, smooth_mask, refinePose, architecture="convnn", weigths_file=None,
          ctfType="apply", pad=2):

    # Create data generator
    generator = Generator(h5_file=h5_file, shuffle=shuffle, batch_size=batch_size,
                          step=step, splitTrain=splitTrain, cost=cost, radius_mask=radius_mask,
                          smooth_mask=smooth_mask, refinePose=refinePose, pad_factor=pad)

    # Train model
    # strategy = tf.distribute.MirroredStrategy()
    # with strategy.scope():
    #     autoencoder = AutoEncoder(generator, architecture=architecture)
    autoencoder = AutoEncoder(generator, architecture=architecture, CTF=ctfType)

    # Fine tune a previous model
    if weigths_file:
        autoencoder.load_weights(weigths_file)

    # if tf.__version__ == '2.11.0':
    #     # optimizer = tf.keras.optimizers.experimental.AdamW(learning_rate=1e-4)
    #     optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
    # else:
    #     optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)

    autoencoder.compile(optimizer=optimizer)
    autoencoder.fit(generator, epochs=epochs)

    # Save model
    autoencoder.save_weights(os.path.join(outPath, "deep_pose_model"))


if __name__ == '__main__':
    import argparse

    # Input parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--h5_file', type=str, required=True)
    parser.add_argument('--out_path', type=str, required=True)
    parser.add_argument('--batch_size', type=int, required=True)
    parser.add_argument('--shuffle', action='store_true')
    parser.add_argument('--step', type=int, required=True)
    parser.add_argument('--split_train', type=float, required=True)
    parser.add_argument('--epochs', type=int, required=True)
    parser.add_argument('--cost', type=str, required=True)
    parser.add_argument('--architecture', type=str, required=True)
    parser.add_argument('--ctf_type', type=str, required=True)
    parser.add_argument('--pad', type=int, required=False, default=2)
    parser.add_argument('--radius_mask', type=float, required=False, default=2)
    parser.add_argument('--smooth_mask', action='store_true')
    parser.add_argument('--refine_pose', action='store_true')
    parser.add_argument('--weigths_file', type=str, required=False, default=None)
    parser.add_argument('--gpu', type=str)

    args = parser.parse_args()

    if args.gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    physical_devices = tf.config.list_physical_devices('GPU')
    for gpu_instance in physical_devices:
        tf.config.experimental.set_memory_growth(gpu_instance, True)

    inputs = {"h5_file": args.h5_file, "outPath": args.out_path,
              "batch_size": args.batch_size, "shuffle": args.shuffle,
              "step": args.step, "splitTrain": args.split_train, "epochs": args.epochs,
              "cost": args.cost, "radius_mask": args.radius_mask, "smooth_mask": args.smooth_mask,
              "refinePose": args.refine_pose, "architecture": args.architecture,
              "weigths_file": args.weigths_file, "ctfType": args.ctf_type, "pad": args.pad}

    # Initialize volume slicer
    train(**inputs)