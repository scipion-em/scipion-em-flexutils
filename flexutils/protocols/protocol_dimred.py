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
from sklearn.neighbors import KDTree

from pyworkflow import BETA
from pyworkflow.object import String, CsvList
from pyworkflow.protocol import LEVEL_ADVANCED
from pyworkflow.protocol.params import PointerParam, EnumParam, IntParam, BooleanParam, MultiPointerParam
import pyworkflow.utils as pwutils
from pyworkflow.utils.properties import Message
from pyworkflow.gui.dialog import askYesNo

from pwem.emlib.image import ImageHandler
from pwem.protocols import ProtAnalysis3D
from pwem.objects import SetOfClasses3D, Class3D, Volume, SetOfVolumes

import flexutils
from flexutils.utils import getOutputSuffix, computeNormRows
import flexutils.constants as const

import xmipp3


class ProtFlexDimRedSpace(ProtAnalysis3D):
    """ Dimensionality reduction of spaces based on different methods """

    _label = 'dimred space'
    _devStatus = BETA
    OUTPUT_PREFIX = 'outputParticles'
    DIMENSIONS = [2, 3]

    # --------------------------- DEFINE param functions ----------------------
    def _defineParams(self, form):
        form.addSection(label='General parameters')
        form.addParam('particles', PointerParam, label="Input particles",
                      pointerClass='SetOfParticles', important=True,
                      help="Particles must have a flexibility information associated (Zernike3D, CryoDrgn...")
        form.addParam('l1', IntParam, default=3,
                      label='Zernike Degree',
                      expertLevel=LEVEL_ADVANCED,
                      condition="particles and hasattr(particles.getFirstItem(),'_xmipp_sphCoefficients')"
                                "and not hasattr(particles,'L1')",
                      help='Degree Zernike Polynomials of the deformation=1,2,3,...')
        form.addParam('l2', IntParam, default=2,
                      label='Harmonical Degree',
                      condition="particles and hasattr(particles.getFirstItem(),'_xmipp_sphCoefficients')"
                                "and not hasattr(particles,'L2')",
                      expertLevel=LEVEL_ADVANCED,
                      help='Degree Spherical Harmonics of the deformation=1,2,3,...')
        form.addParam('mode', EnumParam, choices=['UMAP', 'PCA'],
                      default=0, display=EnumParam.DISPLAY_HLIST,
                      label="Dimensionality reduction method",
                      help="\t * UMAP: usually leads to more meaningfull spaces, although execution "
                           "is higher\n"
                           "\t * PCA: faster but less meaningfull spaces \n"
                           "UMAP and PCA are only computed the first time the are used. Afterwards, they "
                           "will be reused to increase performance")
        form.addParam('nb_umap', IntParam, label="UMAP neighbors",
                      default=5, condition="mode==0",
                      help="Number of neighbors to associate to each point in the space when computing "
                           "the UMAP space. The higher the number of neighbors, the more predominant "
                           "global in the original space features will be")
        form.addParam('epochs_umap', IntParam, label="Number of UMAP epochs",
                      default=1000, condition="mode==0",
                      help="Increasing the number of epochs will lead to more accurate UMAP spaces at the cost "
                           "of larger execution times")
        form.addParam('densmap_umap', BooleanParam, label="Compute DENSMAP?",
                      default=False, condition="mode==0",
                      help="DENSMAP will try to bring densities in the UMAP space closer to each other. Execution time "
                           "will increase when computing a DENSMAP")
        form.addParam('dimensions', EnumParam, choices=['2D', '3D'],
                      default=0, display=EnumParam.DISPLAY_HLIST,
                      label="Landscape space dimensions?",
                      help="Determine if the original landscape will be reduced to have "
                           "2 or 3 dimensions.")
        form.addParallelSection(threads=4, mpi=0)


    # --------------------------- INSERT steps functions ----------------------
    def _insertAllSteps(self):
        self._insertFunctionStep(self.computeReducedSpace)
        self._insertFunctionStep(self.createOutputStep)

    def createOutputStep(self):
        file_coords = self._getExtraPath("red_coords.txt")
        red_space = np.loadtxt(file_coords)

        inputSet = self.particles.get()
        partSet = self._createSetOfParticles()

        partSet.copyInfo(inputSet)
        partSet.__dict__ = inputSet.__dict__.copy()
        partSet.setAlignmentProj()

        for idx, particle in enumerate(inputSet.iterItems()):
            # z = correctionFactor * zernike_space[idx]

            csv_z_space = CsvList()
            for c in red_space[idx]:
                csv_z_space.append(c)

            particle._red_space = csv_z_space

            partSet.append(particle)

        self._defineOutputs(outputParticles=partSet)
        self._defineTransformRelation(self.particles, partSet)

    # --------------------------- STEPS functions -----------------------------
    def computeReducedSpace(self):
        particles = self.particles.get()
        self.num_vol = 0

        # ********* Get Z space *********
        z_space = []
        if hasattr(particles.getFirstItem(), "_xmipp_sphCoefficients"):
            for particle in particles.iterItems():
                z_space.append(np.fromstring(particle._xmipp_sphCoefficients.get(), sep=","))
            z_space = np.asarray(z_space)
        elif hasattr(particles.getFirstItem(), "_cryodrgnZValues"):
            for particle in particles.iterItems():
                z_space.append(np.fromstring(particle._cryodrgnZValues.get(), sep=","))
            z_space = np.asarray(z_space)

        # ********************

        # Generate files to call command line
        file_z_space = self._getExtraPath("z_space.txt")
        np.savetxt(file_z_space, z_space)

        # Compute reduced space
        file_coords = self._getExtraPath("red_coords.txt")
        mode = self.mode.get()
        if mode == 0:
            args = "--input %s --umap --output %s --n_neighbors %d --n_epochs %d " \
                   "--n_components %d --thr %d" \
                   % (file_z_space, file_coords, self.nb_umap.get(), self.epochs_umap.get(),
                      self.DIMENSIONS[self.dimensions.get()], self.numberOfThreads.get())
            if self.densmap_umap.get():
                args += " --densmap"
            program = os.path.join(const.XMIPP_SCRIPTS, "dimensionality_reduction.py")
            program = flexutils.Plugin.getProgram(program)
            self.runJob(program, args)
        elif mode == 1:
            args = "--input %s --pca --n_components %d --output %s" \
                   % (file_z_space, self.DIMENSIONS[self.dimensions.get()], file_coords)
            program = os.path.join(const.XMIPP_SCRIPTS, "dimensionality_reduction.py")
            program = flexutils.Plugin.getProgram(program)
            self.runJob(program, args)

    # --------------------------- INFO functions -----------------------------
    def _summary(self):
        summary = []
        if self.getOutputsSize() >= 1:
            for _, outParticles in self.iterOutputAttributes():
                summary.append("Output *%s*:" % outParticles.getNameId().split('.')[1])
        else:
            summary.append("Output particles not ready yet")
        return summary

    def _methods(self):
        return [
            "Dimensionality reduction of spaces based on different methods",
        ]
