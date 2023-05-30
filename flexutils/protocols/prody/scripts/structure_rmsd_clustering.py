# **************************************************************************
# *
# * Authors:     David Herreros Calero (dherreros@cnb.csic.es)
# *              James Krieger (jmkrieger@cnb.csic.es)
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

def structureRMSDClustering(ensemble_file, pdb_file, dist_thr, output_dir):

    if ensemble_file.endswith('.dcd'):
        ensemble = pd.parseDCD(ensemble_file)
    elif ensemble_file.endswith('.ens.npz'):
        ensemble = pd.loadEnsemble(ensemble_file)

    structure = pd.parsePDB(pdb_file, subset='ca')
    dist_thr = dist_thr

    structure_coords = structure.getCoords()
    ensemble.setCoords(structure_coords)

    # Cluster generation
    clusters = []
    frames_cluster = []
    for idi, conf in enumerate(tqdm(ensemble)):
        coords_struct_d = conf.getCoords()
        if clusters:
            found = False
            for idx in range(len(clusters)):
                d = np.sum((clusters[idx] - coords_struct_d) ** 2, axis=1)
                rmsd = np.sqrt(np.sum(d) / d.size)
                if rmsd < dist_thr:
                    clusters[idx] = clusters[idx] + ((coords_struct_d - clusters[idx]) / (len(frames_cluster[idx]) + 1))
                    frames_cluster[idx].append(idi)
                    found = True
                    break
            if not found:
                clusters.append(coords_struct_d)
                frames_cluster.append([idi])
        else:
            clusters.append(coords_struct_d)
            frames_cluster.append([idi])

    # Write clustered structures to output folder
    clustered_struct_folder = os.path.join(output_dir, "clustered_structures")
    os.mkdir(clustered_struct_folder)
    for sid, cluster_coords in enumerate(tqdm(clusters)):
        clustered_struct = structure.copy()
        clustered_struct.setCoords(cluster_coords)
        pd.writePDB(os.path.join(clustered_struct_folder, "cluster_%d.pdb" % sid), clustered_struct)

    # Write image ids for each cluster in folder
    cluster_ids_folder = os.path.join(output_dir, "cluster_img_ids")
    os.mkdir(cluster_ids_folder)
    for sid, vec_ids in enumerate(tqdm(frames_cluster)):
        np.savetxt(os.path.join(cluster_ids_folder, "cluster_%d.txt" % sid), np.asarray(vec_ids))

if __name__ == '__main__':
    import argparse

    # Input parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--ensemble', type=str, required=True)
    parser.add_argument('--pdb', type=str, required=True)
    parser.add_argument('--distThr', type=float, required=True)
    parser.add_argument('--odir', type=str, required=True)

    args = parser.parse_args()

    # Initialize RMSD clustering
    structureRMSDClustering(args.ensemble, args.pdb, args.distThr, args.odir)
