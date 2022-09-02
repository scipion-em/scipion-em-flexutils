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
import farneback3d

import pyworkflow.utils as pwutils

import flexutils.protocols.xmipp.utils.utils as utl

import xmipp3


def computeZernikeCoefficientsMap(file_input, file_target, file_output, L1, L2):
    # Read data
    start_map = utl.readMap(file_input)
    target_map = utl.readMap(file_target)

    # Map to binary
    start_mask = utl.maskMapYen(start_map)
    target_mask = utl.maskMapYen(target_map)
    start_mask = utl.removeSmallCC(start_mask, 50)
    target_mask = utl.removeSmallCC(target_mask, 50)
    start_mask = gaussian_filter(start_mask.astype(float), sigma=0.25)
    target_mask = gaussian_filter(target_mask.astype(float), sigma=0.25)
    start_mask = start_mask > 0.01
    start_mask = start_mask.astype(int)
    target_mask = target_mask > 0.01
    target_mask = target_mask.astype(int)

    # Get original coords in mask
    xmipp_origin = utl.getXmippOrigin(start_map)
    start_coords = utl.getCoordsAtLevel(start_mask, 1)
    start_coords_xo = start_coords - xmipp_origin

    # Optical flow (close regions)
    optflow = farneback3d.Farneback(levels=10, num_iterations=5, poly_n=5, poly_sigma=2, winsize=9,
                                    pyr_scale=0.8)
    flow = optflow.calc_flow(100 * target_mask.astype(np.float32), 100 * start_mask.astype(np.float32))

    # Deformation field (close regions)
    d_x = flow[0][start_mask == 1]
    d_y = flow[1][start_mask == 1]
    d_z = flow[2][start_mask == 1]
    deformation_field = np.vstack([d_x, d_y, d_z]).T

    #### Zernike3D coefficient computation ####
    rmsd_def = []

    # Zernike3D basis
    r = np.round(0.5 * start_map.shape[0])
    Z = utl.computeBasis(L1=L1, L2=L2, pos=start_coords_xo, r=r)

    # Find Zernike3D coefficients
    A = utl.computeZernikeCoefficients(deformation_field, Z)

    # Compute RMSD and deformation
    d = Z @ A.T
    rmsd_def.append(np.sqrt(np.mean(np.sum(d ** 2, axis=1))))

    # Save results
    rmsd_def_file = os.path.join(os.path.dirname(file_output), "rmsd_def.txt")
    rmsd_def = np.array(rmsd_def)
    np.savetxt(rmsd_def_file, rmsd_def)

    z_clnm_file = os.path.join(os.path.dirname(file_output), "z_clnm.txt")
    z_clnm = utl.resizeZernikeCoefficients(A)
    utl.writeZernikeFile(z_clnm_file, z_clnm, L1, L2, r)

    # Save deformed structure
    mask_file = os.path.join(os.path.dirname(file_output), "mask.mrc")
    utl.saveMap(start_mask, mask_file)
    if pwutils.getExt(file_input) == ".mrc":
        file_input += ":mrc"
    params = '-i %s --mask %s:mrc --step 1 --blobr 1 -o %s --clnm %s' % \
             (file_input, mask_file, file_output, z_clnm_file)
    xmipp3.Plugin.runXmippProgram('xmipp_volume_apply_coefficient_zernike3d', params)


if __name__ == '__main__':
    import argparse

    # Input parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--i', type=str, required=True)
    parser.add_argument('--r', type=str, required=True)
    parser.add_argument('--o', type=str, required=True)
    parser.add_argument('--l1', type=int, required=True)
    parser.add_argument('--l2', type=int, required=True)

    args = parser.parse_args()

    # Initialize volume slicer
    computeZernikeCoefficientsMap(args.i, args.r, args.o, args.l1, args.l2)