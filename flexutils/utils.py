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
from pathlib import Path
import os
from scipy.ndimage import gaussian_filter
from xmipp_metadata.image_handler import ImageHandler

from pyworkflow.utils import getExt
from pyworkflow.utils.process import runJob

import flexutils


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


def getXmippFileName(filename):
    if getExt(filename) == ".mrc":
        filename += ":mrc"
    return filename


def coordsToMap(coords, values, xsize, thr=None):
    indices = coords + np.floor(0.5 * xsize)
    indices = indices.astype(int)

    # Place values on grid
    xsize = int(xsize)
    volume = np.zeros((xsize, xsize, xsize), dtype=np.float32)
    np.add.at(volume, (indices[:, 2], indices[:, 1], indices[:, 0]), values)

    # Filter map
    volume = gaussian_filter(volume, sigma=1.)

    # Volume mask
    if thr is None:
        mask = np.ones(volume.shape)
    else:
        mask = np.zeros(volume.shape)
        mask[volume >= thr] = 1

    return volume, mask


def saveMap(filename, map):
    ImageHandler().write(map, filename, overwrite=True)


def generateVolumesHetSIREN(weigths_file, x_het, outdir, step, architecture):
    args = _getEvalVolArgs(x_het, weigths_file, "het_file", outdir, step=step, architecture=architecture)
    program = flexutils.Plugin.getTensorflowProgram("predict_map_het_siren.py", python=False)
    runJob(None, program, ' '.join(args), numberOfMpi=1)


def generateVolumesDeepNMA(weigths_file, c_nma, outdir, sr, xsize):
    args = _getEvalVolArgs(c_nma, weigths_file, "nma_file", outdir, sr=sr, xsize=xsize)
    program = flexutils.Plugin.getTensorflowProgram("predict_map_deep_nma.py", python=False)
    runJob(None, program, ' '.join(args), numberOfMpi=1)


def _getEvalVolArgs(x_het, weigths_file, x_het_param, outdir, step=None, sr=None, xsize=None, architecture=None):
    if not os.path.exists(outdir):
        os.mkdir(outdir)

    hetFilePath = Path(outdir, "zfile.txt")
    np.savetxt(hetFilePath, x_het)
    hetFilePath = os.path.abspath(os.path.join(outdir, 'zfile.txt'))

    args = ['--weigths_file %s' % weigths_file,
            '--%s %s' % (x_het_param, hetFilePath),
            '--out_path %s' % outdir,
            ]

    if step:
        args.append('--step %d' % step)

    if sr:
        args.append('--sr %f' % sr)

    if xsize:
        args.append('--xsize %d' % xsize)

    if architecture:
        args.append('--architecture %s' % architecture)

    return args
