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

import re
import glob
import numpy as np
from xmipp_metadata.image_handler import ImageHandler

from pyworkflow import NEW
from pyworkflow.protocol.params import PointerParam, FloatParam
import pyworkflow.utils as pwutils

from pwem.protocols import ProtAnalysis3D

import xmipp3

class ProtFlexAutoReference(ProtAnalysis3D):
    """ Automatic selection of best reference volume for Zernike3D analysis """

    _label = 'auto reference'
    _devStatus = NEW
    OUTPUT_PREFIX = 'selectedReference'

    # --------------------------- DEFINE param functions ----------------------
    def _defineParams(self, form):
        form.addSection(label='General parameters')
        form.addParam('maps', PointerParam, label="Maps to select reference from",
                      pointerClass='SetOfVolumes', important=True)
        form.addParam('targetResolution', FloatParam, label="Target resolution (A)", default=8.0,
                      help="In Angstroms, the images and the volume are rescaled so that this resolution is at "
                           "2/3 of the Fourier spectrum.")
        form.addParam('angSampling', FloatParam, default=5.0, label="Angular sampling",
                      help="Angular step determining how many combinations of (rot,tilt) will be "
                           "evaluated during the comparison")
        form.addParallelSection(threads=4, mpi=0)

    # --------------------------- INSERT steps functions ----------------------
    def _insertAllSteps(self):
        self._insertFunctionStep(self.computeDecisionMatrix)
        self._insertFunctionStep(self.createOutputStep)

    def createOutputStep(self):
        if not hasattr(self, 'decision_matrix'):
            files = sorted(glob.glob(self._getTmpPath('map_*.mrc')), key=self.numericalSort)
            num_files = len(files)
            ih = ImageHandler()
            self.decision_matrix = np.zeros([num_files, num_files])
            for idx in range(num_files - 1):
                imgFile = self._getExtraPath("Matrix_entry_image_%d_%d.xmp" % (idx, idx + 1))
                density_cmp_image = ih.read(imgFile).getData()
                sum_density_cmp = np.sum(density_cmp_image)
                if sum_density_cmp > 0.0:
                    self.decision_matrix[idx, idx + 1] = 1.0
                    self.decision_matrix[idx + 1, idx] = -1.0
                elif sum_density_cmp < 0.0:
                    self.decision_matrix[idx, idx + 1] = -1.0
                    self.decision_matrix[idx + 1, idx] = 1.0

        row_sum = np.abs(self.decision_matrix.sum(axis=0))
        volId_decision = np.argmax(row_sum) + 1
        maps = self.maps.get()
        outVol = maps[int(volId_decision)].clone()

        # Save new output
        name = self.OUTPUT_PREFIX
        args = {}
        args[name] = outVol
        self._defineOutputs(**args)
        self._defineSourceRelation(maps, outVol)

    # --------------------------- STEPS functions -----------------------------
    def computeDecisionMatrix(self):
        maps = self.maps.get()
        angSampling = self.angSampling.get()

        # Compute new dimensions
        newTs = self.getNewTs()
        newXDim = self.getNewDim(newTs)

        # Move maps to tmp path and compute corr image
        ih = ImageHandler()
        Xdim = maps[1].getDim()[0]
        self.newAngSampling = 360. / np.round(360. / angSampling)
        for idx, compareMap in enumerate(maps.iterItems()):
            inputFile = compareMap.getFileName()
            mapFile = self._getTmpPath('map_%d.mrc' % (idx + 1))
            if pwutils.getExt(inputFile) == ".mrc":
                inputFile += ":mrc"
            if Xdim != newXDim:
                self.runJob("xmipp_image_resize",
                            "-i %s -o %s --dim %d " % (inputFile,
                                                       mapFile,
                                                       newXDim),
                            numberOfMpi=1, env=xmipp3.Plugin.getEnviron())
            else:
                ih.convert(inputFile, mapFile)

        # Compute density comparison image
        files = sorted(glob.glob(self._getTmpPath('map_*.mrc')), key=self.numericalSort)
        num_files = len(files)
        self.decision_matrix = np.zeros([num_files, num_files])
        for idx in range(num_files - 1):
            file_1 = files[idx]
            file_2 = files[idx + 1]
            outFile = self._getExtraPath("Matrix_entry_image_%d_%d.xmp" % (idx, idx + 1))
            self.runJob("xmipp_compare_density",
                        "-v1 %s -v2 %s -o %s --degstep %f --thr %d "
                        % (file_1, file_2, outFile, self.newAngSampling, self.numberOfThreads.get()),
                        env=xmipp3.Plugin.getEnviron())

            # Fill matrix entry
            density_cmp_image = ih.read(outFile).getData()
            sum_density_cmp = np.sum(density_cmp_image)
            if sum_density_cmp > 0.0:
                self.decision_matrix[idx, idx + 1] = 1.0
                self.decision_matrix[idx + 1, idx] = -1.0
            elif sum_density_cmp < 0.0:
                self.decision_matrix[idx, idx + 1] = -1.0
                self.decision_matrix[idx + 1, idx] = 1.0

    # --------------------------- UTILS functions ----------------------------
    def getNewDim(self, Ts):
        map = self.maps.get()[1]
        Xdim = map.getXDim()
        return int(Xdim * map.getSamplingRate() / Ts)

    def getNewTs(self):
        map = self.maps.get()[1]
        targetResolution = self.targetResolution.get()
        Ts = map.getSamplingRate()
        newTs = targetResolution / 3.0
        newTs = max(Ts, newTs)
        return newTs

    def numericalSort(self, value):
        numbers = re.compile(r'(\d+)')
        parts = numbers.split(value)
        parts[1::2] = map(int, parts[1::2])
        return parts

    # --------------------------- INFO functions -----------------------------
    def _summary(self):
        summary = []
        if self.getOutputsSize() >= 1:
            for _, outRegions in self.iterOutputAttributes():
                summary.append("Found best reference")
        else:
            summary.append("Finding best reference in set...")
        return summary

    def _methods(self):
        return [
            "Automatic reference selection for Zernike3D analysis",
        ]
