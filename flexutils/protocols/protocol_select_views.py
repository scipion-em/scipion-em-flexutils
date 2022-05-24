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


import math
import numpy as np
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon

from pyworkflow import BETA
from pyworkflow.protocol.params import PointerParam, FloatParam, EnumParam
import pyworkflow.utils as pwutils

from pwem.viewers.showj import *
from pwem.protocols import ProtAnalysis3D
from pwem.emlib.image import ImageHandler
from pwem.objects import SetOfVolumes, Volume, SetOfParticles

from flexutils.viewers.viewer_ij import launchIJForSelection
from flexutils.utils import getOutputSuffix

from xmipp3.convert import geometryFromMatrix


class ProtFlexSelectViews(ProtAnalysis3D):
    """ Compare different (rot,tilt) views of different maps and interactively select regions to filter/score
     a SetOfParticles """

    _label = 'select views'
    _devStatus = BETA
    OUTPUT_PREFIX = 'selectedParticles'

    # --------------------------- DEFINE param functions ----------------------
    def _defineParams(self, form):
        form.addSection(label='General parameters')
        form.addParam('reference', PointerParam, label="Reference map",
                      pointerClass='Volume', important=True,
                      help='Map to be used as reference for the comparisons')
        form.addParam('compareMaps', PointerParam, label="Maps to compare to",
                      pointerClass='SetOfVolumes, Volume', important=True,
                      help='Maps to compare the reference views to at different (rot,tilt) combinations')
        form.addParam('particles', PointerParam, label="Particles to filter/score",
                      pointerClass='SetOfParticles', important=True)
        form.addParam('mode', EnumParam, choices=['score', 'filter'],
                      default=0, display=EnumParam.DISPLAY_HLIST,
                      label="Score or filter particles?",
                      help="\t * Score: all particles will be added to the output, but they will be "
                           "enabled/disabled based on the selection results \n"
                           "\t * Filter: only particles belonging to the selections will be added "
                           "to the output")
        form.addParam('targetResolution', FloatParam, label="Target resolution (A)", default=8.0,
                      help="In Angstroms, the images and the volume are rescaled so that this resolution is at "
                           "2/3 of the Fourier spectrum.")
        form.addParam('angSampling', FloatParam, default=5.0, label="Angular sampling",
                      help="Angular step determining how many combinations of (rot,tilt) will be "
                           "evaluated during the comparison")
        form.addParallelSection(threads=4, mpi=0)

    # --------------------------- INSERT steps functions ----------------------
    def _insertAllSteps(self):
        self._insertFunctionStep(self.computeCorrImageStep)
        self._insertFunctionStep(self.launchIJGUIStep, interactive=True)

    def _createOutput(self):
        particles = self.particles.get()
        mode = self.mode.get()
        angSampling = self.angSampling.get()
        outFile = self._getExtraPath("combined_corrImage.txt")
        self.rois_border = np.loadtxt(outFile, delimiter=',')

        # Move maps to tmp path and compute corr image
        self.newAngSampling = 360. / np.round(360. / angSampling)

        # Define polygons based on selected borders
        polygons = []
        for idx in self.rois_border[:, 3]:
            self.roi_border = self.newAngSampling * \
                              np.squeeze(self.rois_border[np.where(self.rois_border[:, 3] == idx), :2])
            self.roi_border = self.roi_border.tolist()
            self.sortPolygonPoints()
            polygons.append(Polygon(self.roi_border))

        # Create output object
        suffix = getOutputSuffix(self, SetOfParticles)
        output_particles = self._createSetOfParticles(suffix)
        output_particles.copyInfo(particles)

        # Loop particles and filter/score them according to polygon delimited areas
        for particle in particles.iterItems():
            in_area = False
            _, angles = geometryFromMatrix(particle.getTransform().getMatrix(), True)
            rot = angles[0]
            tilt = angles[1]
            # Make sure angles are positive
            rot = rot if rot >= 0. else rot + 360.
            tilt = tilt if tilt >= 0. else tilt + 360.
            point = Point(tilt, rot)
            for polygon in polygons:
                if polygon.contains(point):
                    in_area = True
                    break
            aux_particle = particle.clone()
            if in_area:
                output_particles.append(aux_particle)
            elif not in_area and mode == 0:
                aux_particle.setEnabled(False)
                output_particles.append(aux_particle)

        # Save new output
        name = self.OUTPUT_PREFIX + suffix
        args = {}
        args[name] = output_particles
        self._defineOutputs(**args)
        self._defineSourceRelation(particles, output_particles)
        self._updateOutputSet(name, output_particles, state=output_particles.STREAM_CLOSED)

    # --------------------------- STEPS functions -----------------------------
    def computeCorrImageStep(self):
        reference = self.reference.get()
        compareMaps = self.compareMaps.get()
        angSampling = self.angSampling.get()

        # Compute new dimensions
        newTs = self.getNewTs()
        newXDim = self.getNewDim(newTs)

        # Move reference to tmp path
        ih = ImageHandler()
        Xdim = self.reference.get().getDim()[0]
        inputFile = reference.getFileName()
        refFile = self._getTmpPath('reference.mrc')
        if pwutils.getExt(inputFile) == ".mrc":
            inputFile += ":mrc"
        if Xdim != newXDim:
            self.runJob("xmipp_image_resize",
                        "-i %s -o %s --dim %d " % (inputFile,
                                                   refFile,
                                                   newXDim), numberOfMpi=1)
        else:
            ih.convert(inputFile, refFile)

        # Move maps to tmp path and compute corr image
        self.newAngSampling = 360. / np.round(360. / angSampling)
        size_rot = int(360. / self.newAngSampling)
        size_tlt = int(180. / self.newAngSampling)
        combined_corr_image = np.ones([size_rot + 1, size_tlt + 1])

        if isinstance(compareMaps, SetOfVolumes):
            iterator = compareMaps.iterItems()
        elif isinstance(compareMaps, Volume):
            iterator = [compareMaps]

        for idx, compareMap in enumerate(iterator):
            inputFile = compareMap.getFileName()
            mapFile = self._getTmpPath('map_%d.mrc' % (idx + 1))
            if pwutils.getExt(inputFile) == ".mrc":
                inputFile += ":mrc"
            corrImageFile = self._getExtraPath('corrImage_%d.mrc' % (idx + 1))
            if Xdim != newXDim:
                self.runJob("xmipp_image_resize",
                            "-i %s -o %s --dim %d " % (inputFile,
                                                       mapFile,
                                                       newXDim), numberOfMpi=1)
            else:
                ih = ImageHandler()
                ih.convert(inputFile, mapFile)

            # Compute corr image
            self.runJob("xmipp_compare_views",
                        "-v1 %s -v2 %s -o %s --degstep %f --thr %d "
                        % (refFile, mapFile, corrImageFile, self.newAngSampling, self.numberOfThreads.get()))

            # Combine all corr images
            corr_image = ih.read(corrImageFile).getData()
            combined_corr_image *= corr_image

        # Saved combined corr image
        image = ih.createImage()
        image.setData(combined_corr_image.astype(np.float32))
        image.write(self._getExtraPath("combined_corrImage.mrc"))


    def launchIJGUIStep(self):
        corrImageFile = self._getExtraPath("combined_corrImage.mrc")

        path = self._getExtraPath()
        launchIJForSelection(path, corrImageFile)

        if os.path.isfile(self._getExtraPath("combined_corrImage.txt")):
            self._createOutput()

    # --------------------------- UTILS functions ----------------------------
    def getNewDim(self, Ts):
        reference = self.reference.get()
        Xdim = reference.getXDim()
        return int(Xdim * reference.getSamplingRate() / Ts)

    def getNewTs(self):
        reference = self.reference.get()
        targetResolution = self.targetResolution.get()
        Ts = reference.getSamplingRate()
        newTs = targetResolution / 3.0
        newTs = max(Ts, newTs)
        return newTs

    def sortPolygonPoints(self):
        cent = (sum([p[0] for p in self.roi_border]) /
                len(self.roi_border), sum([p[1] for p in self.roi_border]) / len(self.roi_border))
        # sort by polar angle
        self.roi_border.sort(key=lambda p: math.atan2(p[1] - cent[1], p[0] - cent[0]))

    # --------------------------- INFO functions -----------------------------
    def _summary(self):
        summary = []
        if self.getOutputsSize() >= 1:
            for _, outRegions in self.iterOutputAttributes():
                summary.append("Output *%s*:" % outRegions.getNameId().split('.')[1])
                summary.append("    * Total # particles: *%s*" % outRegions.getSize())
        else:
            summary.append("Output ROIs not ready yet.")
        return summary

    def _methods(self):
        return [
            "Interactive selection of best views for flexibility analysis with ImageJ and "
            "filtering/scoring based on selections",
        ]
