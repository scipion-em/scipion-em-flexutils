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
import prody as pd
import numpy as np
from joblib import Parallel, delayed
from itertools import repeat
import multiprocessing

import pwem.emlib.metadata as md

import flexutils.protocols.xmipp.utils.utils as utl
from flexutils.viewers.threshold_atoms_viewer import ModelThresholdView


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------
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


def grid_subsampling(points, voxel_size):

  # nb_vox = np.ceil((np.max(points, axis=0) - np.min(points, axis=0))/voxel_size)
  non_empty_voxel_keys, inverse, nb_pts_per_voxel = np.unique(((points - np.min(points, axis=0)) // voxel_size).astype(int), axis=0, return_inverse=True, return_counts=True)
  idx_pts_vox_sorted = np.argsort(inverse)
  voxel_grid = {}
  grid_barycenter, grid_candidate_center = [], []
  last_seen = 0

  for idx, vox in enumerate(non_empty_voxel_keys):
    voxel_grid[tuple(vox)] = points[idx_pts_vox_sorted[last_seen:last_seen+nb_pts_per_voxel[idx]]]
    grid_barycenter.append(np.mean(voxel_grid[tuple(vox)], axis=0))
    grid_candidate_center.append(voxel_grid[tuple(vox)][np.linalg.norm(voxel_grid[tuple(vox)]-np.mean(voxel_grid[tuple(vox)], axis=0), axis=1).argmin()])
    last_seen += nb_pts_per_voxel[idx]

  return np.vstack(grid_candidate_center)


# ---------------------------------------------------------------------------
# Parallel computation of motion statistics
# ---------------------------------------------------------------------------
def summers(z_space, coords, Z, processes, batches=32):
    pool = multiprocessing.Pool(processes=processes)
    # pbar = tqdm(total=z_space.shape[0])

    class Sum:
        def __init__(self, coords):
            self.coords = coords
            self.count = 0

        def getWorkerIdent(self):
            return multiprocessing.current_process().ident

        def computation(self, z):
            A = utl.resizeZernikeCoefficients(z)
            d_z = Z @ A.T
            coords_d = self.coords + d_z
            _, R, t = umeyama(coords_d, self.coords)
            coords_d = coords_d.dot(R) + t
            df_norm = np.linalg.norm(self.coords - coords_d, axis=1)
            # worker_id = multiprocessing.current_process().ident
            # self.mean_pos[worker_id] += df_norm
            # self.std_pos[worker_id] += df_norm * df_norm
            return df_norm, df_norm * df_norm

        def initAccumulators(self, worker_ids):
            self.mean_pos = {worker_id: np.zeros(coords.shape[0]) for worker_id in worker_ids}
            self.std_pos = {worker_id: np.zeros(coords.shape[0]) for worker_id in worker_ids}

        def worker(self, func, args_batch):
            """Call func with every packet of arguments received and update
            result array on the run.

            Worker function which runs the job in each spawned process.
            """
            mean_pos = np.zeros(self.coords.shape[0])
            std_pos = np.zeros(self.coords.shape[0])
            for args_ in args_batch:
                update_mean_pos, update_std_pos = func(args_)
                np.sum([mean_pos, update_mean_pos], axis=0, out=mean_pos)
                np.sum([std_pos, update_std_pos], axis=0, out=std_pos)
            return mean_pos, std_pos

    sumArr = Sum(coords)

    with Parallel(n_jobs=processes, verbose=100) as parallel:
        funcs = repeat(sumArr.computation, batches)
        args_batches = np.array_split(z_space, batches, axis=0)
        jobs = zip(funcs, args_batches)

        result_batches = parallel(delayed(sumArr.worker)(*job) for job in jobs)

        mean_pos = np.zeros(coords.shape[0])
        std_pos = np.zeros(coords.shape[0])
        for batch_result in result_batches:
            mean_pos += batch_result[0]
            std_pos += batch_result[1]

    return mean_pos, std_pos

def appender(z_space, coords, Z, keep_pos, save, processes):

    def computation(z):
        A = utl.resizeZernikeCoefficients(z)
        d_z = Z @ A.T
        coords_d = coords + d_z
        _, R, t = umeyama(coords_d, coords)
        coords_d = coords_d.dot(R) + t
        if save == "structures":
            return coords_d[keep_pos].flatten()
        elif save == "residuals":
            return (coords - coords_d)[keep_pos].flatten()

    result = Parallel(n_jobs=processes, verbose=100)(delayed(computation)(z) for z in z_space)

    return np.asarray(result)


# ---------------------------------------------------------------------------
# Main functions
# ---------------------------------------------------------------------------

def thresholdAtoms(z_space_file, structure_file, out_path, sampling_rate, bxsize,
                   L1, L2, thr):

    # Parse structure and subsample with sampling rate spacing
    structure = pd.parsePDB(structure_file, subset='ca', compressed=False)
    coords_struct = structure.getCoords()
    coords_struct = coords_struct - np.mean(coords_struct, axis=0)
    coords_struct = grid_subsampling(coords_struct, sampling_rate)

    # Get Zernike3D space
    z_space = sampling_rate * np.loadtxt(z_space_file)

    # Compute Zernike3D bassis
    Z = utl.computeBasis(L1=L1, L2=L2, pos=coords_struct, r=0.5 * sampling_rate * bxsize)

    # Get deformation field mean and std
    mean_pos, std_pos = summers(z_space, coords_struct, Z, thr, batches=32)
    mean_pos /= z_space.shape[0]
    std_df = np.sqrt(std_pos / z_space.shape[0] - mean_pos * mean_pos)

    # Interactive structure thresholding
    m = ModelThresholdView(coords=coords_struct, std_df=std_df)
    m.configure_traits()

    # Save thresholding info
    np.savetxt(os.path.join(out_path, "keep_pos.txt"), m.keep_pos)

def getReducedSpace(z_space_file, structure_file, out_path, sampling_rate, bxsize,
                    L1, L2, save, red_mode, thr, **kwargs):

    # Parse structure and subsample with sampling rate spacing
    structure = pd.parsePDB(structure_file, subset='ca', compressed=False)
    coords_struct = structure.getCoords()
    coords_struct = coords_struct - np.mean(coords_struct, axis=0)
    coords_struct = grid_subsampling(coords_struct, sampling_rate)

    # Get Zernike3D space
    z_space = sampling_rate * np.loadtxt(z_space_file)

    # Compute Zernike3D bassis
    Z = utl.computeBasis(L1=L1, L2=L2, pos=coords_struct, r=0.5 * sampling_rate * bxsize)

    # Read thresholding info
    keep_pos = np.loadtxt(os.path.join(out_path, "keep_pos.txt")).astype(bool)

    # Get structure space
    reduced_ensemble_structs = appender(z_space, coords_struct, Z, keep_pos, save, thr)

    # Reduce space to 2D or 3D
    n_components = kwargs.pop("n_components", None)
    if red_mode == "pca":
        from sklearn.decomposition import PCA
        pca = PCA(n_components=n_components).fit(reduced_ensemble_structs)
        coords = pca.transform(reduced_ensemble_structs)
    elif red_mode == "umap":
        from umap import UMAP
        n_neighbors = kwargs.pop("n_neighbors", 5)
        n_epochs = kwargs.pop("n_epochs", 5)
        densmap = kwargs.pop("densmap", 5)
        umap = UMAP(n_components=n_components, n_neighbors=n_neighbors,
                    n_epochs=n_epochs, densmap=densmap, n_jobs=thr).fit(reduced_ensemble_structs)
        coords = umap.transform(reduced_ensemble_structs)
    np.savetxt(os.path.join(out_path, "red_coords.txt"), coords)


if __name__ == '__main__':
    import argparse

    # Input parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--structure', type=str, required=True)
    parser.add_argument('--z_space', type=str, required=True)
    parser.add_argument('--sr', type=float, required=True)
    parser.add_argument('--bxsize', type=int, required=False)
    parser.add_argument('--L1', type=int, required=True)
    parser.add_argument('--L2', type=int, required=True)
    parser.add_argument('--save', type=str, required=False)
    parser.add_argument('--pca', action='store_true')
    parser.add_argument('--umap', action='store_true')
    parser.add_argument('--n_components', type=int, required=False)
    parser.add_argument('--n_neighbors', type=int, required=False)
    parser.add_argument('--n_epochs', type=int, required=False)
    parser.add_argument('--densmap', action='store_true')
    parser.add_argument('--out_path', type=str, required=True)
    parser.add_argument('--thr', type=int, required=True)

    args = parser.parse_args()

    # Initialize volume slicer
    if args.pca:
        getReducedSpace(args.z_space, args.structure, args.out_path, args.sr, args.bxsize,
                        args.L1, args.L2, args.save, "pca", args.thr,
                        n_components=args.n_components)
    elif args.umap:
        getReducedSpace(args.z_space, args.structure, args.out_path, args.sr, args.bxsize,
                        args.L1, args.L2, args.save, "umap", args.thr,
                        n_components=args.n_components, n_neighbors=args.n_neighbors,
                        n_epochs=args.n_epochs, densmap=args.densmap)
    else:
        thresholdAtoms(args.z_space, args.structure, args.out_path, args.sr, args.bxsize,
                       args.L1, args.L2, args.thr)
