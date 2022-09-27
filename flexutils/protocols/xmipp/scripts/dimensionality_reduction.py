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


import numpy as np


def reduceDimensions(coords_ld, outFile, mode, **kwargs):
    n_components = kwargs.pop("n_components", None)
    if mode == "pca":
        from sklearn.decomposition import PCA
        pca = PCA(n_components=n_components).fit(coords_ld)
        coords = pca.transform(coords_ld)
    elif mode == "umap":
        from umap import UMAP
        n_neighbors = kwargs.pop("n_neighbors", 5)
        n_epochs = kwargs.pop("n_epochs", 5)
        densmap = kwargs.pop("densmap", 5)
        umap = UMAP(n_components=n_components, n_neighbors=n_neighbors,
                    n_epochs=n_epochs, densmap=densmap).fit(coords_ld)
        coords = umap.transform(coords_ld)
    np.savetxt(outFile, coords)


if __name__ == '__main__':
    import argparse

    # Input parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--pca', action='store_true')
    parser.add_argument('--umap', action='store_true')
    parser.add_argument('--n_components', type=int, required=True)
    parser.add_argument('--n_neighbors', type=int, required=False)
    parser.add_argument('--n_epochs', type=int, required=False)
    parser.add_argument('--densmap', action='store_true')
    parser.add_argument('--output', type=str, required=True)

    args = parser.parse_args()

    coords = np.loadtxt(args.input)

    # Initialize volume slicer
    if args.pca:
        reduceDimensions(coords, args.output, "pca", n_components=args.n_components)
    elif args.umap:
        reduceDimensions(coords, args.output, "umap", n_components=args.n_components,
                         n_neighbors=args.n_neighbors, n_epochs=args.n_epochs,
                         densmap=args.densmap)
