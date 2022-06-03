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
import pickle
import math
from shapely.geometry import Polygon


def sortPolygonPoints(roi_border):
    cent = (sum([p[0] for p in roi_border]) /
            len(roi_border), sum([p[1] for p in roi_border]) / len(roi_border))
    # sort by polar angle
    roi_border.sort(key=lambda p: math.atan2(p[1] - cent[1], p[0] - cent[0]))

    return roi_border

def createPolygons(coords, sampling, outFile):
    # Define polygons based on selected borders
    polygons = []
    for idx in coords[:, 3]:
        roi_border = sampling * np.squeeze(coords[np.where(coords[:, 3] == idx), :2])
        roi_border = roi_border.tolist()
        roi_border = sortPolygonPoints(roi_border)
        polygons.append(Polygon(roi_border))

    with open(outFile, "wb") as f:
        pickle.dump(polygons, f)


if __name__ == '__main__':
    import argparse

    # Input parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--angs', type=float, required=True)
    parser.add_argument('--output', type=str, required=True)

    args = parser.parse_args()

    coords = np.loadtxt(args.input, delimiter=',')

    # Initialize volume slicer
    createPolygons(coords, args.angs, args.output)
