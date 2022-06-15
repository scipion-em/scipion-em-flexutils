# -*- coding: utf-8 -*-
# **************************************************************************
# *
# * Authors:     David Herreros Calero (dherreros@cnb.csic.es)
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


def getOutputSuffix(protocol, cls):
    """ Get the name to be used for a new output.
    For example: output3DCoordinates7.
    It should take into account previous outputs
    and number with a higher value.
    """
    maxCounter = -1
    for attrName, _ in protocol.iterOutputAttributes(cls):
        suffix = attrName.replace(protocol.OUTPUT_PREFIX, '')
        try:
            counter = int(suffix)
        except:
            counter = 1  # when there is not number assume 1
        maxCounter = max(counter, maxCounter)

    return str(maxCounter + 1) if maxCounter > 0 else ''  # empty if not output

def readZernikeFile(filename):
    with open(filename, 'r') as fid:
        lines = fid.readlines()
    basis_params = np.fromstring(lines[0].strip('\n'), sep=' ')

    z_clnm = []
    for line in lines[1:]:
        z_clnm.append(np.fromstring(line.strip('\n'), sep=' '))
    z_clnm = np.asarray(z_clnm)

    return basis_params, z_clnm

def computeNormRows(array):
    norm = []
    size = int(array.shape[1] / 3)
    for vec in array:
        c_3d = np.vstack([vec[:size], vec[size:2 * size], vec[2 * size:]])
        norm.append(np.linalg.norm(np.linalg.norm(c_3d, axis=1)))
    return np.vstack(norm).flatten()
