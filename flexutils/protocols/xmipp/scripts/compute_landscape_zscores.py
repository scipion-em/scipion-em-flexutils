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
from scipy.spatial import cKDTree
import pynndescent



def computeLandscapeZScores(landscape_file, outPath, fast, neighbours=10):
    distribution = []

    # Read lanscape
    landscape = np.loadtxt(landscape_file)

    # Create Tree
    if fast:
        idx = 1
        tree = pynndescent.NNDescent(landscape)
        tree.prepare()
    else:
        idx = 0
        tree = cKDTree(landscape)

    # Compute distance distributions
    for z in landscape:
        distances = tree.query(z, k=neighbours)[idx]
        distribution.append(np.mean(distances[1:]))

    # Compute Z-Scores
    distribution = np.asarray(distribution)
    z_scores = np.abs((distribution - np.mean(distribution)) / np.std(distribution))

    # Save results
    np.savetxt(os.path.join(outPath, "z_scores.txt"), z_scores)


if __name__ == '__main__':
    import argparse

    # Input parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--i', type=str, required=True)
    parser.add_argument('--o', type=str, required=True)
    parser.add_argument('--fast', action='store_true')
    parser.add_argument('--neighbours', type=int, required=True)

    args = parser.parse_args()

    # Initialize volume slicer
    computeLandscapeZScores(args.i, args.o, args.fast, args.neighbours)