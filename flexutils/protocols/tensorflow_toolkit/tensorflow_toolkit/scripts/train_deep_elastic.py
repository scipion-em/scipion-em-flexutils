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
from sklearn.cluster import KMeans

import tensorflow as tf

from tensorflow_toolkit.networks.deep_elastic import VAExplode, cluster_explosion

# # os.environ["CUDA_VISIBLE_DEVICES"]="0,2,3,4"
# physical_devices = tf.config.list_physical_devices('GPU')
# for gpu_instance in physical_devices:
#     tf.config.experimental.set_memory_growth(gpu_instance, True)


def train(space, output, splitTrain, n_clusters, init_power, end_power, vae_sigma, lat_dim,
          loss_lambda):
    # Prepare data
    data = np.loadtxt(space)

    # Remove outliers FIXME: Do we want this?
    # norm = np.linalg.norm(data, axis=1)
    # norm_mean = np.mean(norm)
    # norm_std = np.std(norm)
    # ind = np.where((norm - norm_mean) < norm_std)
    # data = data[ind]

    # Split train dataset
    train_data = np.random.choice(np.arange(data.shape[0]), size=int(splitTrain * data.shape[0]), replace=False)
    data_train = data[train_data, :]

    # Initial clustering
    clusters = KMeans(n_clusters=n_clusters, n_init=1).fit(data_train)
    clusters_all = KMeans(n_clusters=n_clusters, n_init=1, init=clusters.cluster_centers_).fit(data)

    # Explosion/Implosion power
    powers = np.linspace(init_power, end_power, 3)  # Previously it was 5

    # Network
    encoder = VAExplode(lat_dim, data.shape[1], kl_lambda=0., dist_sigma=vae_sigma,
                        loss_lambda=loss_lambda, train_log_var=False)

    # Train model
    lr = 0.001  # Initial 0.1, then 0.001
    batch_size = 64  # Initial 256, then 32
    epochs = 25
    for power in powers:
        print("------ Training network with explosion power %.2f ------" % power)
        if power != powers[0]:
            lr = 0.0001  # Initial 0.001
            batch_size = 32  # Initial 64, then 16
            epochs = 10
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr)  # Initial Adamax
        encoder.compile(optimizer=optimizer)
        data_train_aux = cluster_explosion(data_train, clusters, power)
        np.random.shuffle(data_train_aux)
        encoder.fit(data_train_aux, epochs=epochs, shuffle=True, batch_size=batch_size)

    # Encode initial space
    encoded_space = encoder.encoder(cluster_explosion(data, clusters_all, powers[-1]))[-1].numpy()

    # Save space
    np.savetxt(output, encoded_space)


if __name__ == '__main__':
    import argparse

    # Input parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--space', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--split_train', type=float, required=True)
    parser.add_argument('--clusters', type=int, required=True)
    parser.add_argument('--init_power', type=float, required=True)
    parser.add_argument('--end_power', type=float, required=True)
    parser.add_argument('--vae_sigma', type=float, required=True)
    parser.add_argument('--lat_dim', type=int, required=True)
    parser.add_argument('--loss_lambda', type=float, required=True)
    parser.add_argument('--gpu', type=str)

    args = parser.parse_args()

    if args.gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    physical_devices = tf.config.list_physical_devices('GPU')
    for gpu_instance in physical_devices:
        tf.config.experimental.set_memory_growth(gpu_instance, True)

    inputs = {"space": args.space, "output": args.output, "splitTrain": args.split_train,
              "n_clusters": args.clusters, "init_power": args.init_power, "end_power": args.end_power,
              "vae_sigma": args.vae_sigma, "lat_dim": args.lat_dim, "loss_lambda": args.loss_lambda}

    # Initialize volume slicer
    train(**inputs)
