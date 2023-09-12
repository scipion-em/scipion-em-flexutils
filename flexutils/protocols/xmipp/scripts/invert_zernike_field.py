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

import flexutils.protocols.xmipp.utils.utils as utl


def invertZernikeField(file_input, file_reference, file_z_clnm, file_output):

    # Read data
    input_map = utl.readMap(file_input)
    reference_map = utl.readMap(file_reference)
    c_i = utl.getCoordsAtLevel(input_map, 1)
    c_r = utl.getCoordsAtLevel(reference_map, 1)

    # Get Xmipp origin
    xmipp_origin = utl.getXmippOrigin(input_map)

    # Get Zernike3D parameters
    basis_params, A_i = utl.readZernikeFile(file_z_clnm)

    #### Zernike3D coefficient computation ####

    # Apply Xmipp origin
    c_i = c_i - xmipp_origin
    c_r = c_r - xmipp_origin

    # Compute input basis
    Z_i = utl.computeBasis(L1=int(basis_params[0]), L2=int(basis_params[1]),
                           pos=c_i, r=basis_params[2])
    # Compute deformation field
    A_i = utl.resizeZernikeCoefficients(A_i[0])
    df = Z_i @ A_i.T

    # Compute reference basis
    Z_r = utl.computeBasis(L1=int(basis_params[0]), L2=int(basis_params[1]),
                           pos=c_i + df, r=basis_params[2])
    # Z_r = utl.computeBasis(L1=int(basis_params[0]), L2=int(basis_params[1]),
    #                        pos=c_r, r=basis_params[2])


    # Recompute Zernike3D coefficients
    A_r = utl.computeZernikeCoefficients(-df, Z_r)
    A_r = utl.resizeZernikeCoefficients(A_r)

    # Save new coefficients
    utl.writeZernikeFile(file_output, A_r, int(basis_params[0]), int(basis_params[1]),
                         basis_params[2])


if __name__ == '__main__':
    import argparse

    # Input parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--i', type=str, required=True)
    parser.add_argument('--r', type=str, required=True)
    parser.add_argument('--z_clnm', type=str, required=True)
    parser.add_argument('--o', type=str, required=True)

    args = parser.parse_args()

    # Initialize volume slicer
    invertZernikeField(args.i, args.r, args.z_clnm, args.o)
