# **************************************************************************
# *
# * Authors:     David Herreros Calero (dherreos@cnb.csic.es)
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
from joblib import Parallel, delayed
from xmipp_metadata.image_handler import ImageHandler
from xmipp_metadata.metadata import XmippMetaData

import flexutils.protocols.xmipp.utils.utils as utl


def computeNewDeformationField(Z, Z_new, A):
    # Get deformation field
    A = utl.resizeZernikeCoefficients(A.reshape(-1))

    # Deformation field
    d_f = Z @ A.T

    # New coefficients
    A_new = utl.computeZernikeCoefficients(d_f, Z_new)

    # New deformation field
    d_f_new = Z_new @ A_new.T

    # Print error
    rmsd = np.sqrt(np.sum((d_f - d_f_new) ** 2) / d_f.shape[0])

    return utl.resizeZernikeCoefficients(A_new), rmsd


def field_to_binary_zernike3d(md_file, mask_reg, mask_bin, boxsize, L1, L2, thr):
    # Get coords (regions mask)
    mask = ImageHandler(mask_reg).getData()
    indices = np.asarray(np.where(mask > 0))
    indices = np.transpose(np.asarray([indices[2, :], indices[1, :], indices[0, :]]))
    coords_reg = indices - 0.5 * boxsize
    groups_reg = mask[indices[:, 2], indices[:, 1], indices[:, 0]]

    centers_reg = []
    for group in np.unique(groups_reg):
        centers_reg.append(np.mean(coords_reg[groups_reg == group], axis=0))
    centers_reg = np.asarray(centers_reg)

    # Get coords (binary mask)
    mask = ImageHandler(mask_bin).getData()
    indices = np.asarray(np.where(mask > 0))
    indices = np.transpose(np.asarray([indices[2, :], indices[1, :], indices[0, :]]))
    coords_bin = indices - 0.5 * boxsize

    # Read Zernike params
    md = XmippMetaData(md_file)
    A_all = np.asarray([np.fromstring(item, sep=',') for item in md[:, 'zernikeCoefficients']])
    r = 0.5 * boxsize

    # Compute basis
    Z = utl.computeBasis(L1=int(L1), L2=int(L2), pos=coords_reg, r=r, groups=groups_reg, centers=centers_reg)
    Z_new = utl.computeBasis(L1=int(L1), L2=int(L2), pos=coords_bin, r=r, groups=None, centers=None)

    # Compute new coefficients
    results = Parallel(n_jobs=thr, verbose=100) \
        (delayed(computeNewDeformationField)(Z, Z_new, A) for A in A_all)
    A_all_new, rmsd = map(list, zip(*results))
    A_all_new = np.asarray(A_all_new)
    rmsd = np.asarray(rmsd)

    # Save new coefficients
    md[:, 'zernikeCoefficients'] = np.asarray([",".join(item) for item in A_all_new.astype(str)])
    md[:, 'fieldApproximationError'] = rmsd
    md.write(md_file, overwrite=True)


if __name__ == '__main__':
    import argparse

    # Input parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--md_file', type=str, required=True)
    parser.add_argument('--mask_reg', type=str, required=True)
    parser.add_argument('--mask_bin', type=str, required=True)
    parser.add_argument('--boxsize', type=int, required=True)
    parser.add_argument('--l1', type=int, required=True)
    parser.add_argument('--l2', type=int, required=True)
    parser.add_argument('--thr', type=int, required=True)

    args = parser.parse_args()

    # Initialize volume slicer
    field_to_binary_zernike3d(args.md_file, args.mask_reg, args.mask_bin,
                              args.boxsize, args.l1, args.l2, args.thr)
