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
from scipy.ndimage import gaussian_filter

import pyworkflow.utils as pwutils

import flexutils.protocols.xmipp.utils.utils as utl

import xmipp3


def computeZernikeDeformation(file_input, file_z_clnm, file_output):

    # Read data
    start_map = utl.readMap(file_input)
    coords = utl.getCoordsAtLevel(start_map, 1)

    # Get Xmipp origin
    xmipp_origin = utl.getXmippOrigin(start_map)

    # Get Zernike3D parameters
    basis_params, z_clnm_vec = utl.readZernikeFile(file_z_clnm)

    #### Zernike3D coefficient computation ####

    # Apply Xmipp origin
    zernike_coords = coords - xmipp_origin

    # Compute basis
    Z = utl.computeBasis(L1=int(basis_params[0]), L2=int(basis_params[1]),
                         pos=zernike_coords, r=basis_params[2])

    # Get Zernike3D coeffcients and compute deformation
    deformation = []
    for z_clnm in z_clnm_vec:
        A = utl.resizeZernikeCoefficients(z_clnm)
        df = Z @ A.T
        deformation.append(np.sqrt(np.mean(np.sum(df ** 2, axis=1))))

    # Save deformation
    np.savetxt(file_output, deformation)


if __name__ == '__main__':
    import argparse

    # Input parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--i', type=str, required=True)
    parser.add_argument('--z_clnm', type=str, required=True)
    parser.add_argument('--o', type=str, required=True)

    args = parser.parse_args()

    # Initialize volume slicer
    computeZernikeDeformation(args.i, args.z_clnm, args.o)
