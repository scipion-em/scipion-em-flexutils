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

import mrcfile
import numpy as np
from scipy.ndimage import gaussian_filter


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------
def scatter_nd_add(target, indices, updates):
    indices = tuple(indices.reshape(-1, indices.shape[-1]).T)
    np.add.at(target, indices, updates)
    return target

def compute_Strain_Rotation(Gx, Gy, Gz, indices):
    boxsize = Gx.shape[0]
    U = np.zeros((3, 3))
    D = np.zeros((3, 3))
    H = np.zeros((3, 3))
    LS = np.zeros((boxsize, boxsize, boxsize))
    LR = np.zeros((boxsize, boxsize, boxsize))

    k, i, j = 0.0, 0.0, 0.0
    km1, im1, jm1 = 0.0, 0.0, 0.0
    kp1, ip1, jp1 = 0.0, 0.0, 0.0
    km2, im2, jm2 = 0.0, 0.0, 0.0
    kp2, ip2, jp2 = 0.0, 0.0, 0.0

    Dx = lambda V: (V[k, i, jm2] - 8 * V[k, i, jm1] + 8 * V[k, i, jp1] - V[k, i, jp2]) / 12.0
    Dy = lambda V: (V[k, im2, j] - 8 * V[k, im1, j] + 8 * V[k, ip1, j] - V[k, ip2, j]) / 12.0
    Dz = lambda V: (V[km2, i, j] - 8 * V[km1, i, j] + 8 * V[kp1, i, j] - V[kp2, i, j]) / 12.0

    for idx in range(indices.shape[0]):
        pos = indices[idx]

        k = pos[0]
        i = pos[1]
        j = pos[2]

        km1 = k - 1
        kp1 = k + 1
        km2 = k - 2
        kp2 = k + 2

        im1 = i - 1
        ip1 = i + 1
        im2 = i - 2
        ip2 = i + 2

        jm1 = j - 1
        jp1 = j + 1
        jm2 = j - 2
        jp2 = j + 2

        if jp2 < boxsize and jm2 >= 0 and \
           ip2 < boxsize and im2 >= 0 and \
           kp2 < boxsize and km2 >= 0:

            U[0, 0], U[0, 1], U[0, 2] = Dx(Gx), Dy(Gx), Dz(Gx)
            U[1, 0], U[1, 1], U[1, 2] = Dx(Gy), Dy(Gy), Dz(Gy)
            U[2, 0], U[2, 1], U[2, 2] = Dx(Gz), Dy(Gz), Dz(Gz)

            D[0, 0] = U[0, 0]
            D[0, 1] = D[1, 0] = 0.5 * (U[0, 1] + U[1, 0])
            D[0, 2] = D[2, 0] = 0.5 * (U[0, 2] + U[2, 0])
            D[1, 1] = U[1, 1]
            D[1, 2] = D[2, 1] = 0.5 * (U[1, 2] + U[2, 1])
            D[2, 2] = U[2, 2]

            H[0, 1] = 0.5 * (U[0, 1] - U[1, 0])
            H[0, 2] = 0.5 * (U[0, 2] - U[2, 0])
            H[1, 2] = 0.5 * (U[1, 2] - U[2, 1])
            H[1, 0] = -H[0, 1]
            H[2, 0] = -H[0, 2]
            H[2, 1] = -H[1, 2]

            LS[k, i, j] += np.abs(np.linalg.det(D))

            eigs, _ = np.linalg.eig(H.astype(np.complex128))
            for n in range(eigs.size):
                imagabs = np.abs(eigs[n].imag)
                if imagabs > 1e-6:
                    LR[k, i, j] += imagabs * 180 / np.pi
                    break

    return LS, LR


# ---------------------------------------------------------------------------
# Main functions
# ---------------------------------------------------------------------------
def analyzeRS(field_file, indices_file, out_path, boxsize):
    # Get deformation field and indices
    d_f = np.loadtxt(field_file)
    indices = np.loadtxt(indices_file).astype(int)  # Indices must be in ZYX order

    # Compute deformation field volumes
    Gx = scatter_nd_add(np.zeros([boxsize, boxsize, boxsize]), indices, d_f[:, 0])
    Gy = scatter_nd_add(np.zeros([boxsize, boxsize, boxsize]), indices, d_f[:, 1])
    Gz = scatter_nd_add(np.zeros([boxsize, boxsize, boxsize]), indices, d_f[:, 2])

    # Deformation field volume smoothing (to smooth derivatives)
    Gx = gaussian_filter(Gx, sigma=2.0)
    Gy = gaussian_filter(Gy, sigma=2.0)
    Gz = gaussian_filter(Gz, sigma=2.0)

    # Compute strain and rotation volumes
    LS, LR = compute_Strain_Rotation(Gx, Gy, Gz, indices=indices)

    # Normalize strain and rotation volumes
    LS, LR = LS / (np.mean(LS[:]) + 1e-6), LR / (np.mean(LR[:]) + 1e-6)

    # Remove outliers from strain and rotation volumes
    LS_std, LR_std = np.std(LS[:]), np.std(LR[:])
    LS[LS > 3 * LS_std] = LS_std
    LR[LR > 3 * LR_std] = LR_std

    # Filter strain and rotation volumes to ensure continuity
    LS = gaussian_filter(LS, sigma=2.0)
    LR = gaussian_filter(LR, sigma=2.0)

    # Save strain
    with mrcfile.new(os.path.join(out_path, "strain.mrc"), overwrite=True) as mrc:
        mrc.set_data(LS.astype(np.float32))

    # Save rotation
    with mrcfile.new(os.path.join(out_path, "rotation.mrc"), overwrite=True) as mrc:
        mrc.set_data(LR.astype(np.float32))


if __name__ == '__main__':
    import argparse

    # Input parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--field', type=str, required=True)
    parser.add_argument('--indices', type=str, required=True)
    parser.add_argument('--out_path', type=str, required=True)
    parser.add_argument('--boxsize', type=int, required=False)

    args = parser.parse_args()

    # Initialize volume slicer
    analyzeRS(args.field, args.indices, args.out_path, args.boxsize)
