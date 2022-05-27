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


def computeZernikeCoefficientsMap(file_input, file_target, file_output, L1, L2,
                                  global_search=None, mode="skeleton"):
    # Read data
    start_map = utl.readMap(file_input)
    target_map = utl.readMap(file_target)

    # Filter input maps (this will be optional in the protocol and will really on Xmipp side)
    # start_map = gaussian_filter(start_map, sigma=1)
    # target_map = gaussian_filter(target_map, sigma=1)

    # Get Xmipp origin
    xmipp_origin = utl.getXmippOrigin(start_map)

    # Pre-align maps in ChimeraX using fitmap
    start_map, pre_Tr = utl.alignMapsChimeraX(file_input, file_target, global_search=global_search)
    pre_Tr_inv = np.linalg.inv(pre_Tr)

    # Map to binary
    start_map = utl.maskMapOtsu(start_map)
    target_map = utl.maskMapOtsu(target_map)

    # Save start map mask
    mask_file = os.path.join(os.path.dirname(file_output), "mask.mrc")
    utl.saveMap(start_map, mask_file)

    # Get skeleton of binary mask (this and the previous part might be two methods in the protocol)
    if mode == "skeleton":
        start_map_skeleton = utl.extractSkeleton(start_map, 1.0)
        target_map_skeleton = utl.extractSkeleton(target_map, 1.0)
    elif mode == "border":
        start_map_skeleton = None
        target_map_skeleton = None

    # Get border of binary mask
    start_map = utl.extractBorder(start_map)
    target_map = utl.extractBorder(target_map)

    if start_map_skeleton is None:
        # Disconnect highly heterogeneous regions (only useful if border is used - not for skeleton)
        start_map_close, start_map_far = utl.disconnectHetRegions(start_map, target_map, 2.0)  # 2 for spike
        target_map_close, target_map_far = utl.disconnectHetRegions(target_map, start_map, 2.0)  # 2 for spike

        # Remove small components (only needed for "far" map)
        start_map_far = utl.removeSmallCC(start_map_far, 300)  # 300 for spike
        target_map_far = utl.removeSmallCC(target_map_far, 300)  # 300 for spike

        # Match CC
        matched_levels = utl.matchCC(start_map_far, target_map_far)

        # Initialize parameters for Zernike3D computations
        zernike_coords = []
        deformation_field = []
        start_coords_match = []
        target_coords_match = []


        #### Align matched CC ####
        for levels in matched_levels:
            start_cc = utl.extractCC(start_map_far, levels[0])
            target_cc = utl.extractCC(target_map_far, levels[1])

            # Improve connectivity of CC
            start_cc = utl.reextractComponent(start_cc, start_map)
            target_cc = utl.reextractComponent(target_cc, target_map)
            start_cc = utl.improveCCMask(start_cc, 5, 5)
            target_cc = utl.improveCCMask(target_cc, 5, 5)

            # Get coordinates of CC
            start_coords = utl.getCoordsAtLevel(start_cc, 1)
            target_coords = utl.getCoordsAtLevel(target_cc, 1)

            # Find transformation matrix aligning the CC
            Tr = utl.icp(start_coords, target_coords)

            # Apply transformation and get deformation fields for CC
            start_coords_alg, Df_CC = utl.applyTransformation(start_coords, Tr)

            # Fill Zernike3D parameters based on CC alignment
            zernike_coords.append(start_coords)
            deformation_field.append(Df_CC)

            # Matching coords (for RMSD computation only)
            idx_match_start, idx_match_target = utl.matchCoords(start_coords_alg, target_coords, False)
            start_coords_match.append(start_coords[idx_match_start])
            target_coords_match.append(target_coords[idx_match_target])


        #### Align close regions ####

        # Get coordinates of close components
        start_coords = utl.getCoordsAtLevel(start_map_close, 1)
        target_coords = utl.getCoordsAtLevel(target_map_close, 1)

        # Find transformation matrix aligning the close components
        Tr = utl.icp(start_coords, target_coords)

        # Apply transformation and get deformation fields for close components
        start_coords_alg, _ = utl.applyTransformation(start_coords, Tr)

        # Fill Zernike3D parameters based on close components alignment
        # zernike_coords.append(start_coords)
        # deformation_field.append(Df_CC)

        # Matching coords (for RMSD computation only)
        idx_match_start, idx_match_target = utl.matchCoords(start_coords_alg, target_coords, True)
        start_coords_match.append(start_coords[idx_match_start])
        target_coords_match.append(target_coords[idx_match_target])

        # This deformation field choice is better
        zernike_coords.append(start_coords[idx_match_start])
        deformation_field.append(target_coords[idx_match_target] - start_coords[idx_match_start])

    else:
        # Get coordinates of skeleton and border
        start_coords_skeleton = utl.getCoordsAtLevel(start_map_skeleton, 1)
        target_coords_skeleton = utl.getCoordsAtLevel(target_map_skeleton, 1)
        start_coords_border = utl.getCoordsAtLevel(start_map, 1)
        target_coords_border = utl.getCoordsAtLevel(target_map, 1)

        # Associate border coords to skeleton coords
        indeces = utl.associateBorderAndSkeleton(start_coords_border, start_coords_skeleton)

        # Find transformation matrix aligning the close components
        Tr = utl.icp(start_coords_skeleton, target_coords_skeleton)

        # Apply transformation and get deformation fields for close components
        start_coords_alg, _ = utl.applyTransformation(start_coords_skeleton, Tr)

        # Fill Zernike3D parameters based on close components alignment
        # zernike_coords.append(start_coords)
        # deformation_field.append(Df_CC)

        # Matching coords (for RMSD computation only)
        idx_match_start, idx_match_target = utl.matchCoords(start_coords_alg, target_coords_skeleton, False)
        start_coords_match = start_coords_skeleton[idx_match_start]
        target_coords_match = target_coords_skeleton[idx_match_target]

        # This deformation field choice is better
        zernike_coords_skeleton = start_coords_skeleton[idx_match_start]
        deformation_field_skeleton = target_coords_skeleton[idx_match_target] - start_coords_skeleton[idx_match_start]

        # Match deformation field of skeleton to border
        zernike_coords = []
        deformation_field = []
        for pos, idx in enumerate(idx_match_start):
            indeces_border = np.where(indeces == idx)
            zernike_coords.append(start_coords_border[indeces_border])
            deformation_field.append(np.tile(deformation_field_skeleton[pos], (zernike_coords[-1].shape[0], 1)))


    #### Zernike3D coefficient computation ####
    rmsd_def = []
    zernike_coords = np.vstack(zernike_coords)
    zernike_coords, pre_deformation_field = utl.applyTransformation(zernike_coords, pre_Tr_inv, order=None)
    deformation_field = np.vstack(deformation_field)
    deformation_field = deformation_field - pre_deformation_field
    start_coords_match, _ = utl.applyTransformation(np.vstack(start_coords_match), pre_Tr_inv, order=None)
    target_coords_match, _ = utl.applyTransformation(np.vstack(target_coords_match), pre_Tr_inv, order=None)

    # Apply Xmipp origin
    zernike_coords = zernike_coords - xmipp_origin
    start_coords_match = start_coords_match - xmipp_origin
    target_coords_match = target_coords_match - xmipp_origin

    # Compute RMSD for alignment evaluation
    rmsd_def.append(utl.computeRMSD(start_coords_match, target_coords_match))

    # Zernike3D basis
    r = utl.inscribedRadius(utl.getCoordsAtLevel(start_map, 1))
    Z = utl.computeBasis(L1=L1, L2=L2, pos=zernike_coords, r=r)

    # Find Zernike3D coefficients
    A = utl.computeZernikeCoefficients(deformation_field, Z)
    A *= 1.0

    # Compute RMSD and deformation
    Z = utl.computeBasis(L1=L1, L2=L2, pos=start_coords_match, r=r)
    d = Z @ A.T
    c_start_moved = start_coords_match + d
    rmsd_def.append(utl.computeRMSD(c_start_moved, target_coords_match))
    rmsd_def.append(np.sqrt(np.mean(np.sum(d ** 2, axis=1))))

    # Save results
    rmsd_def_file = os.path.join(os.path.dirname(file_output), "rmsd_def.txt")
    rmsd_def = np.array(rmsd_def)
    np.savetxt(rmsd_def_file, rmsd_def)

    z_clnm_file = os.path.join(os.path.dirname(file_output), "z_clnm.txt")
    z_clnm = utl.resizeZernikeCoefficients(A)
    utl.writeZernikeFile(z_clnm_file, z_clnm, L1, L2, r)

    # Save deformed structure
    if pwutils.getExt(file_input) == ".mrc":
        file_input += ":mrc"
    params = '-i %s --mask %s:mrc --step 1 --blobr 2 -o %s --clnm %s' % \
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
    parser.add_argument('--gs', type=int)
    parser.add_argument('--skeleton', action='store_true')
    parser.add_argument('--border', action='store_true')

    args = parser.parse_args()

    if args.gs is None:
        gs = None
    else:
        gs = args.gs

    mode = "skeleton"
    if args.skeleton:
        mode = "skeleton"
    elif args.border:
        mode = "border"

    # Initialize volume slicer
    computeZernikeCoefficientsMap(args.i, args.r, args.o, args.l1, args.l2, gs, mode)