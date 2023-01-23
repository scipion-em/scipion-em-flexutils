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


class ResidueConv2D(tf.keras.layers.Layer):
    def __init__(self, nk, sz, activation='linear', **args):
        super(ResidueConv2D, self).__init__()

        self.nk = nk
        self.sz = sz
        self.act = activation

    def get_config(self):
        config = super().get_config()
        config.update({
            "nk": self.nk,
            "sz": self.sz,
            "activation": self.act,
        })
        return config

    def build(self, shp):
        nk = self.nk
        sz = self.sz
        self.conv1_w = self.add_weight(shape=(sz, sz, nk, nk), initializer='random_normal', trainable=True,
                                       name="conv1_w")
        self.conv1_b = self.add_weight(shape=(nk,), initializer='zeros', trainable=True, name="conv1_b")

        if shp[3] == self.nk:
            self.skip = True
        else:
            self.conv0_w = self.add_weight(shape=(1, 1, shp[-1], nk), initializer='random_normal', trainable=True,
                                           name="conv0_w")
            self.conv0_b = self.add_weight(shape=(nk,), initializer='zeros', trainable=True, name="conv0_b")
            self.skip = False

    def call(self, inp):
        if self.skip:
            out = tf.nn.conv2d(inp, self.conv1_w, (1, 1), "SAME") + self.conv1_b + inp

        else:
            mid = tf.nn.conv2d(inp, self.conv0_w, (1, 1), "SAME") + self.conv0_b
            out = tf.nn.conv2d(mid, self.conv1_w, (1, 1), "SAME") + self.conv1_b + mid

        if self.act == 'relu':
            out = tf.nn.relu(out)

        return out