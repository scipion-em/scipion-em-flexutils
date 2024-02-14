# **************************************************************************
# *
# * Authors:     David Herreros Calero (dherreros@cnb.csic.es)
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
from scipy.spatial.distance import cdist
from python_tsp.heuristics import solve_tsp_simulated_annealing, solve_tsp_local_search


def compute_distances(coordinates):
    M, N = coordinates.shape

    diff = coordinates[:, np.newaxis, :] - coordinates[np.newaxis, :, :]
    squared_diff = diff ** 2.
    distances = np.sqrt(np.sum(squared_diff, axis=2))

    u, v = np.meshgrid(np.arange(M), np.arange(M), indexing='ij')

    # Filter out the diagonal entries where u == v
    mask = u != v
    filtered_u = u[mask]
    filtered_v = v[mask]
    filtered_distances = distances[mask]

    result = np.stack((filtered_u, filtered_v, filtered_distances), axis=1)

    return result

def salesmanSolver(coords, outpath):
    # Find optimum path
    # num_clusters = coords.shape[0]

    # Compute distance matrix
    distance_matrix = cdist(coords, coords)
    distance_matrix[:, 0] = 0  # No need to be a closed path

    # Shortest path
    permutation_mh, distance_mh = solve_tsp_simulated_annealing(distance_matrix)
    permutation_mh_lc, distance_mh_lc = solve_tsp_local_search(distance_matrix, x0=permutation_mh,
                                                               perturbation_scheme="ps3")

    # fitness_coords = mlrose.TravellingSales(distances=distances)
    # problem_fit = mlrose.TSPOpt(length=num_clusters, fitness_fn=fitness_coords,
    #                             maximize=False)
    # best_state, best_fitness = mlrose.simulated_annealing(problem_fit, max_attempts=10,  # We might try 100
    #                                                       random_state=np.arange(num_clusters),
    #                                                       schedule=mlrose.ArithDecay(init_temp=10.0))

    # Save optimum path and fitness
    with open(outpath, 'w') as fid:
        fid.write(','.join(map(str, [state + 1 for state in permutation_mh_lc])) + "\n")
        fid.write(('%.2f' % distance_mh_lc) + "\n")

def readZernike3DFile(path, num_vol):
    # Read selected coefficients
    z_clnm_vw = []
    with open(path) as f:
        lines = f.readlines()
        for line in lines:
            z_clnm_vw.append(np.fromstring(line, dtype=float, sep=' '))
    return np.asarray(z_clnm_vw[num_vol:])


if __name__ == '__main__':
    import argparse

    # Input parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--coords', type=str, required=True)
    parser.add_argument('--outpath', type=str, required=True)
    parser.add_argument('--num_vol', type=int, required=True)

    args = parser.parse_args()

    # Read and generate data
    coords = np.loadtxt(args.coords)[args.num_vol:]
    # coords = readZernike3DFile(args.coords, args.num_vol)

    # Initialize volume slicer
    salesmanSolver(coords=coords, outpath=args.outpath)
