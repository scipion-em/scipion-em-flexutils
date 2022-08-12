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

import flexutils.protocols.xmipp.utils.utils as utl

import pwem.emlib.metadata as md

from joblib import Parallel, delayed


def iterRowsAndClone(metadata):
    row = md.Row()
    for objId in metadata:
        row.readFromMd(metadata, objId)
        yield row.clone()


def computeNewDeformationField(row, Z, Zpp, Ap):
    outRow = row
    z_clnm = np.asarray(row.getValue(md.MDL_SPH_COEFFICIENTS))
    A = utl.resizeZernikeCoefficients(z_clnm)

    # Compute Zernike coefficients for new reference
    App = utl.reassociateCoefficients(Z, Zpp, Ap, A)

    # Write Zernike3D coefficients to file
    z_clnm = utl.resizeZernikeCoefficients(App)
    outRow.setValue(md.MDL_SPH_COEFFICIENTS, z_clnm.tolist())

    # For evaluation (debuggin purposes)
    d = Zpp @ App.T

    # Compute mean deformation
    deformation = np.sqrt(np.mean(np.sum(d ** 2, axis=1)))
    outRow.setValue(md.MDL_SPH_DEFORMATION, deformation)

    return outRow


def reassignReference(md_file, maski, maskr, file_z_clnm_r, prevL1, prevL2, L1, L2, Rmax, thr):
    # Read data
    start_mask = utl.readMap(maski)
    # target_mask = utl.readMap(maskr)

    # Get Xmipp origin
    xmipp_origin = utl.getXmippOrigin(start_mask)

    # Get original coords in mask
    start_coords = utl.getCoordsAtLevel(start_mask, 1)
    start_coords_xo = start_coords - xmipp_origin

    # Get new coords
    # target_coords = utl.getCoordsAtLevel(target_mask, 1)
    # target_coords_xo = target_coords - xmipp_origin

    # Metadata
    metadata = md.MetaData(md_file)
    metadata.sort()
    # rows = [row.clone() for row in md.iterRows(metadata)]
    rows = Parallel(n_jobs=thr, verbose=100)(delayed(lambda x: x.clone())(row) for row in iterRowsAndClone(metadata))

    # Compute basis
    _, z_clnm_vec = utl.readZernikeFile(file_z_clnm_r)
    Z = utl.computeBasis(L1=prevL1, L2=prevL2, pos=start_coords_xo, r=Rmax)
    # Zpp = utl.computeBasis(L1=L1, L2=L2, pos=target_coords_xo, r=Rmax)

    # Compute deformation field (from original to new reference)
    Ap = utl.resizeZernikeCoefficients(z_clnm_vec[0])

    # Get new coords
    df = Z @ Ap.T
    target_coords_xo = start_coords_xo + df
    # target_coords = utl.getCoordsAtLevel(target_mask, 1)
    # target_coords_xo = target_coords - xmipp_origin

    # Compute basis
    # _, z_clnm_vec = utl.readZernikeFile(file_z_clnm_r)
    # Z = utl.computeBasis(L1=prevL1, L2=prevL2, pos=start_coords_xo, r=Rmax)
    Zpp = utl.computeBasis(L1=L1, L2=L2, pos=target_coords_xo, r=Rmax)

    # Compute new deformation field
    outRows = Parallel(n_jobs=thr, verbose=100) \
              (delayed(lambda x: computeNewDeformationField(x, Z, Zpp, Ap))(row)
              for row in rows)

    # Fill output metadata
    metadata_out = md.MetaData()
    [outRow.addToMd(metadata_out) for outRow in outRows]

    dir = os.path.dirname(md_file)
    metadata_out.write(os.path.join(dir, "inputParticles_reassigned.xmd"))


if __name__ == '__main__':
    import argparse

    # Input parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--i', type=str, required=True)
    parser.add_argument('--maski', type=str, required=True)
    parser.add_argument('--maskr', type=str, required=True)
    parser.add_argument('--zclnm_r', type=str, required=True)
    parser.add_argument('--prevl1', type=int, required=True)
    parser.add_argument('--prevl2', type=int, required=True)
    parser.add_argument('--l1', type=int, required=True)
    parser.add_argument('--l2', type=int, required=True)
    parser.add_argument('--rmax', type=float, required=True)
    parser.add_argument('--thr', type=int, required=True)

    args = parser.parse_args()

    # Initialize volume slicer
    reassignReference(args.i, args.maski, args.maskr, args.zclnm_r, args.prevl1, args.prevl2,
                      args.l1, args.l2, args.rmax, args.thr)
