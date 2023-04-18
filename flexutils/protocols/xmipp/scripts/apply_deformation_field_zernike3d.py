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
import os.path

import numpy as np
from pathlib import Path
import prody as pd
from scipy.ndimage import gaussian_filter
from xmipp_metadata.image_handler import ImageHandler

import flexutils.protocols.xmipp.utils.utils as utl


def apply_deformation_field_zernike3d(ref_file, vol_file, z_file, out_file, boxsize, sr):
    ref_file = Path(ref_file)

    # Get coords
    if ref_file.suffix == ".mrc":
        mode = "volume"
        mask = ImageHandler(ref_file).getData()
        volume = ImageHandler(vol_file).getData()
        indices = np.asarray(np.where(mask > 0))
        indices = np.transpose(np.asarray([indices[2, :], indices[1, :], indices[0, :]]))
        coords = indices - 0.5 * boxsize
        groups = mask[indices[:, 2], indices[:, 1], indices[:, 0]]
        values = volume[indices[:, 2], indices[:, 1], indices[:, 0]]

        centers = []
        for group in np.unique(groups):
            centers.append(np.mean(coords[groups == group], axis=0))
        centers = np.asarray(centers)

    elif ref_file.suffix == ".pdb":
        mode = "structure"
        pd_struct = pd.parsePDB(str(ref_file), subset=None, compressed=False)
        coords = pd_struct.getCoords()
        centers = None
        groups = None

    # Read Zernike params
    basis_params, A = utl.readZernikeFile(z_file)
    L1, L2, r = basis_params
    r = sr * r if mode == "structure" else r

    # Zernike coefficients
    A = utl.resizeZernikeCoefficients(A.reshape(-1))

    # Compute basis
    Z = utl.computeBasis(L1=int(L1), L2=int(L2), pos=coords, r=r, groups=groups, centers=centers)

    # Compute deformation field
    d_f = Z @ A.T

    # Apply deformation field
    if mode == "volume":
        indices_moved = (indices + d_f).astype(int)

        # Scatter in volume
        def_vol = np.zeros((boxsize, boxsize, boxsize))
        for idx in range(indices_moved.shape[0]):
            def_vol[indices_moved[idx, 2], indices_moved[idx, 1], indices_moved[idx, 0]] += values[idx]
        # np.add.at(def_vol, [indices_moved[:, 2], indices_moved[:, 1], indices_moved[:, 0]], 1)

        # Gaussian filter map
        def_vol = gaussian_filter(def_vol, sigma=1.0)

        # Save results
        ImageHandler().write(def_vol, filename=out_file, overwrite=True, sr=sr)

    elif mode == "structure":
        # Move coords
        coords_moved = coords + d_f

        # Save results
        pd_struct_out = pd_struct.copy()
        pd_struct_out.setCoords(coords_moved)
        pd.writePDB(out_file, pd_struct_out)


if __name__ == '__main__':
    import argparse

    # Input parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--ref_file', type=str, required=True)
    parser.add_argument('--vol_file', type=str, required=True)
    parser.add_argument('--z_file', type=str, required=True)
    parser.add_argument('--out_file', type=str, required=True)
    parser.add_argument('--boxsize', type=int, required=True)
    parser.add_argument('--sr', type=float, required=True)

    args = parser.parse_args()

    # Initialize volume slicer
    apply_deformation_field_zernike3d(args.ref_file, args.vol_file, args.z_file, args.out_file,
                                      args.boxsize, args.sr)
