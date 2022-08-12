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
from shapely.geometry import Point


def createPolygons(coords, polygons_file, outFile):
    # Define polygons based on selected borders
    with open(polygons_file, "rb") as f:
        polygons = pickle.load(f)

    in_area_vec = []
    for coord in coords:
        in_area = 0
        point = Point(coord[0], coord[1])
        for polygon in polygons:
            if polygon.contains(point):
                in_area = 1
                break
        in_area_vec.append(in_area)

    np.savetxt(outFile, np.asarray(in_area_vec))


if __name__ == '__main__':
    import argparse

    # Input parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--polygons', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)

    args = parser.parse_args()

    coords = np.loadtxt(args.input, delimiter=' ')

    # Initialize volume slicer
    createPolygons(coords, args.polygons, args.output)
