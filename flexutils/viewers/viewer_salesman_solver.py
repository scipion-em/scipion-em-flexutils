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
import mlrose


def salesmanSolver(coords, outpath):
    # Find optimum path
    num_clusters = coords.shape[0]
    fitness_coords = mlrose.TravellingSales(coords=coords)
    problem_fit = mlrose.TSPOpt(length=num_clusters, fitness_fn=fitness_coords,
                                maximize=False)
    best_state, best_fitness = mlrose.genetic_alg(problem_fit, random_state=2)

    # Save optimum path and fitness
    with open(outpath, 'w') as fid:
        fid.write(','.join(map(str, [state + 1 for state in best_state])) + "\n")
        fid.write(('%.2f' % best_fitness) + "\n")

def readZernike3DFile(path, num_vol):
    # Read selected coefficients
    z_clnm_vw = []
    with open(path) as f:
        lines = f.readlines()
        for line in lines:
            z_clnm_vw.append(np.fromstring(line, dtype=float, sep=' '))
    return np.asarray(z_clnm_vw[1+num_vol:])


if __name__ == '__main__':
    import argparse

    # Input parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--coords', type=str, required=True)
    parser.add_argument('--outpath', type=str, required=True)
    parser.add_argument('--num_vol', type=int, required=True)

    args = parser.parse_args()

    # Read and generate data
    coords = readZernike3DFile(args.coords, args.num_vol)

    # Initialize volume slicer
    salesmanSolver(coords=coords, outpath=args.outpath)
