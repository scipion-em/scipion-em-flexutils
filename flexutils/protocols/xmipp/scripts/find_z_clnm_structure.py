import os
import numpy as np
import prody as pd
import flexutils.protocols.xmipp.utils.utils as utl

def computeZernikeCoefficientsStructure(file_input, file_target, file_output, L1, L2):
    start = pd.parsePDB(file_input, subset='ca', compressed=False)
    target = pd.parsePDB(file_target, subset='ca', compressed=False)

    ens = pd.buildPDBEnsemble([start, target])

    coords_start = ens.getCoordsets()[0]
    coords_target = ens.getCoordsets()[1]
    center_mass = np.mean(ens.getCoords(), axis=0)
    coords_start = coords_start - center_mass
    coords_target = coords_target - center_mass
    rmsd = pd.calcRMSD(coords_start, coords_target)

    # Deformation field
    Df = coords_target - coords_start

    # Zernikes
    r = utl.inscribedRadius(coords_start)
    Z = utl.computeBasis(L1=L1, L2=L2, pos=coords_start, r=r)

    # Find Zernike3D coefficients
    A = utl.computeZernikeCoefficients(Df, Z)
    # A *= 1

    # Find and save deformed structure
    d = Z @ A.T
    c_start_moved = coords_start + d
    start_mv = start.copy()
    start_mv.setCoords(c_start_moved + center_mass)
    pd.writePDB(file_output, start_mv)

    # Compute mean deformation
    deformation = np.sqrt(np.mean(np.sum(d ** 2, axis=1)))

    # Save results
    rmsd_def_file = os.path.join(os.path.dirname(file_output), "rmsd_def.txt")
    rmsd_def = np.array([rmsd,
                         pd.calcRMSD(c_start_moved, coords_target),
                         deformation])
    np.savetxt(rmsd_def_file, rmsd_def)

    z_clnm_file = os.path.join(os.path.dirname(file_output), "z_clnm.txt")
    z_clnm = utl.resizeZernikeCoefficients(A)
    utl.writeZernikeFile(z_clnm_file, z_clnm, L1, L2, r)


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
    computeZernikeCoefficientsStructure(args.i, args.r, args.o, args.l1, args.l2)
