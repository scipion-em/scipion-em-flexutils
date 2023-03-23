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

import prody as pd

from tqdm import tqdm

import flexutils.protocols.xmipp.utils.utils as utl


def umeyama(P, Q):
    assert P.shape == Q.shape
    n, dim = P.shape

    centeredP = P - P.mean(axis=0)
    centeredQ = Q - Q.mean(axis=0)

    C = np.dot(np.transpose(centeredP), centeredQ) / n

    V, S, W = np.linalg.svd(C)
    d = (np.linalg.det(V) * np.linalg.det(W)) < 0.0

    if d:
        S[-1] = -S[-1]
        V[:, -1] = -V[:, -1]

    R = np.dot(V, W)

    varP = np.var(P, axis=0).sum()
    c = 1/varP * np.sum(S) # scale factor

    t = Q.mean(axis=0) - P.mean(axis=0).dot(c*R)

    return c, R, t


def structureRMSDClustering(z_space, pdb_file, box_size, sr, dist_thr, L1, L2, output_dir):

    # Read Zernike3D space
    z_space = np.loadtxt(z_space)

    # Parse PDB Structure
    structure = pd.parsePDB(pdb_file, subset='ca', compressed=False)
    structure_coords = structure.getCoords()

    # We assume structure has been aligned to the map inside Scipion, so we need to
    # move its origin accordingly
    half_box = np.floor(0.5 * box_size) * sr
    structure_coords = structure_coords

    # Compute Zernike3D basis
    Z = utl.computeBasis(L1=L1, L2=L2, pos=structure_coords, r=half_box)

    # Cluster generation
    clusters = []
    imgs_cluster = []
    for idi, z in enumerate(tqdm(z_space)):
        A = utl.resizeZernikeCoefficients(z)
        d_z = Z @ A.T
        coords_struct_d = structure_coords + d_z
        _, R, t = umeyama(coords_struct_d, structure_coords)
        coords_struct_d = coords_struct_d.dot(R) + t
        if clusters:
            found = False
            for idx in range(len(clusters)):
                d = np.sum((clusters[idx] - coords_struct_d) ** 2, axis=1)
                rmsd = np.sqrt(np.sum(d) / d.size)
                if rmsd < dist_thr:
                    clusters[idx] = clusters[idx] + ((coords_struct_d - clusters[idx]) / (len(imgs_cluster[idx]) + 1))
                    imgs_cluster[idx].append(idi)
                    found = True
                    break
            if not found:
                clusters.append(coords_struct_d)
                imgs_cluster.append([idi])
        else:
            clusters.append(coords_struct_d)
            imgs_cluster.append([idi])


    # Write clustered structures to output folder
    clustered_struct_folder = os.path.join(output_dir, "clustered_structures")
    os.mkdir(clustered_struct_folder)
    for sid, cluster_coords in enumerate(tqdm(clusters)):
        clustered_struct = structure.copy()
        clustered_struct.setCoords(cluster_coords + half_box)
        pd.writePDB(os.path.join(clustered_struct_folder, "cluster_%d.pdb" % sid), clustered_struct)

    # Write image ids for each cluster in folder
    cluster_ids_folder = os.path.join(output_dir, "cluster_img_ids")
    os.mkdir(cluster_ids_folder)
    for sid, vec_ids in enumerate(tqdm(imgs_cluster)):
        np.savetxt(os.path.join(cluster_ids_folder, "cluster_%d.txt" % sid), np.asarray(vec_ids))



if __name__ == '__main__':
    import argparse

    # Input parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--z_space', type=str, required=True)
    parser.add_argument('--pdb', type=str, required=True)
    parser.add_argument('--boxSize', type=float, required=True)
    parser.add_argument('--sr', type=float, required=True)
    parser.add_argument('--distThr', type=float, required=True)
    parser.add_argument('--L1', type=int, required=True)
    parser.add_argument('--L2', type=int, required=True)
    parser.add_argument('--odir', type=str, required=True)

    args = parser.parse_args()

    # Initialize volume slicer
    structureRMSDClustering(args.z_space, args.pdb, args.boxSize, args.sr, args.distThr,
                            args.L1, args.L2, args.odir)
