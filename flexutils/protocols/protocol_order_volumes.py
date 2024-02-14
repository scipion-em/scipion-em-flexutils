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


import os

import numpy as np
from sklearn.neighbors import NearestNeighbors

from pyworkflow import NEW
from pyworkflow.protocol.params import PointerParam

from pwem.protocols import ProtAnalysis3D, ProtFlexBase
from pwem.objects import VolumeFlex

import flexutils


class ProtFlexOrderVolumes(ProtAnalysis3D, ProtFlexBase):
    """ Order a series of volumes along the shortest path represneted by their landscape representation """

    _label = 'find volumes order'
    _devStatus = NEW
    OUTPUT_PREFIX = 'orderedVolumes'

    # --------------------------- DEFINE param functions ----------------------
    def _defineParams(self, form):
        form.addSection(label='General parameters')
        form.addParam('inputVolumes', PointerParam, label="Input volumes",
                      pointerClass='SetOfVolumesFlex', important=True,
                      help="Volumes to be ordered")

    # --------------------------- INSERT steps functions ----------------------
    def _insertAllSteps(self):
        self._insertFunctionStep(self.orderVolumes)
        self._insertFunctionStep(self.createOutputStep)

    # --------------------------- STEPS functions -----------------------------
    def orderVolumes(self):
        inputVolumes = self.inputVolumes.get()
        dataPath = self._getExtraPath("z_space.txt")
        outPath = self._getExtraPath("path.txt")

        # Save flex space to file
        flex_space = []
        for volume in inputVolumes.iterItems():
            flex_space.append(volume.getZFlex())
        flex_space = np.asarray(flex_space)

        # Get extreme points
        vertexes = self.getBorderPoints(flex_space,
                                        threshold_distance=0.6 * self.automaticThresholdDistance(flex_space))

        best_distance = np.inf
        best_path = None
        for idx in range(vertexes.shape[0]):
            reordered_coords, index_in_original = self.reorderPoints(flex_space, vertexes, idx)
            np.savetxt(dataPath, reordered_coords)

            # Run salesman's solver
            program = os.path.join(os.path.dirname(flexutils.__file__), "viewers", "path_finder_tools",
                                   "viewer_salesman_solver.py")
            args = "--coords %s --outpath %s --num_vol 0 " \
                   % (dataPath, outPath)
            program = flexutils.Plugin.getProgram(program)
            self.runJob(program, args, numberOfMpi=1)

            # Read path and distance
            path, distance = self.readPath(outPath)

            if distance < best_distance:
                best_distance = distance
                best_path = self.mapIndicesToOriginal(path, index_in_original, reordered_coords.shape[0])

        # Save best solution
        np.savetxt(outPath, best_path)

    def createOutputStep(self):
        inputVolumes = self.inputVolumes.get()
        outPath = self._getExtraPath("path.txt")

        # Get found path
        path = np.loadtxt(outPath, dtype=int)
        path -= 1

        # Generate ordered volumes
        outVols = self._createSetOfVolumesFlex()
        outVols.copyInfo(inputVolumes)
        volumes = [volume for volume in inputVolumes.iterItems(iterate=False)]
        for objId in path:
            volume = VolumeFlex()
            volume.copyInfo(volumes[objId])
            outVols.append(volume)

        # Save new output
        name = self.OUTPUT_PREFIX
        args = {}
        args[name] = outVols
        self._defineOutputs(**args)
        self._defineSourceRelation(inputVolumes, outVols)


    # --------------------------- UTILS functions ----------------------------
    def readPath(self, file):
        # Open the file for reading
        with open(file, 'r') as file:
            # Read the first line from the file
            line = file.readline().strip()  # Remove any leading/trailing whitespace

            # Split the line into a list of strings, then convert each to an integer
            integer_set = [float(number) for number in line.split(',')]

            # Read the second line from the file
            distance = float(file.readline().strip())

        return np.asarray(integer_set).astype(int), distance

    def getBorderPoints(self, points, n_neighbors=3, threshold_distance=0.5):
        # Fit the model
        nbrs = NearestNeighbors(n_neighbors=n_neighbors).fit(points)
        distances, indices = nbrs.kneighbors(points)

        # Identify points with a mean distance to neighbors greater than a threshold
        border_points = points[np.mean(distances, axis=1) > threshold_distance]

        return border_points

    def reorderPoints(self, original_points, border_points, idx):
        # Find the index of the border point in the original points array
        border_point_to_place_first = border_points[idx]
        index_in_original = np.where((original_points == border_point_to_place_first).all(axis=1))[0][0]

        # Rotate the original_points array so that the border point is the first element
        reordered_points = np.concatenate((original_points[index_in_original:], original_points[:index_in_original]),
                                          axis=0)
        return reordered_points, index_in_original

    def automaticThresholdDistance(self, points, n_neighbors=2):
        # Compute the nearest neighbor distances
        nbrs = NearestNeighbors(n_neighbors=n_neighbors).fit(points)
        distances, _ = nbrs.kneighbors(points)

        # Use the distance to the first nearest neighbor (excluding self)
        nearest_neighbor_distance = distances[:, 1]

        mean_distance = np.mean(nearest_neighbor_distance)
        std_distance = np.std(nearest_neighbor_distance)

        # Example: Set threshold as mean distance plus one standard deviation
        threshold_distance = mean_distance + std_distance

        return threshold_distance

    def mapIndicesToOriginal(self, reordered_indices, start_idx, N):
        """
        Map indices from a reordered array back to their original positions.

        :param reordered_indices: Indices in the reordered array.
        :param start_idx: The index in the original array of the point that was moved to the start.
        :param N: Total number of points in the array.
        :return: A list of indices in their original positions.
        """
        original_indices = [(i + start_idx) % N for i in reordered_indices]
        return original_indices


    # ----------------------- VALIDATE functions ----------------------------------------
    def validate(self):
        """ Try to find errors on define params. """
        errors = []
        return errors

    # --------------------------- INFO functions -----------------------------
    def _summary(self):
        summary = []
        outPath = self._getExtraPath("path.txt")
        if os.path.isfile(outPath):
            path = np.loadtxt(outPath, dtype=int)
            summary.append("Detected order: {}".format(path))
        else:
            summary.append("Ordering volumes...")
        return summary

    def _methods(self):
        return [
            "Ordering of volumes based on the shortest path determined by their conformational landscape representation"
        ]
