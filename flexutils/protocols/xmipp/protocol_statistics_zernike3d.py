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
from xmipp_metadata.metadata import XmippMetaData

import xmipp3
from pyworkflow import NEW
from pyworkflow.protocol.params import PointerParam

from pwem.protocols import ProtAnalysis3D
from pwem.objects import SetOfParticlesFlex

import flexutils
import flexutils.constants as const

from xmipp3.convert import writeSetOfParticles

from flexutils.utils import getXmippFileName


class XmippProtStatisticsZernike3D(ProtAnalysis3D):
    """ 3D visualization of motion statistics """

    _label = 'motion statistics - Zernike3D'
    _devStatus = NEW
    OUTPUT_PREFIX = 'zernike3DClasses'

    # --------------------------- DEFINE param functions ----------------------
    def _defineParams(self, form):
        form.addSection(label='General parameters')
        form.addParam('particles', PointerParam, label="Zernike3D particles",
                      pointerClass='SetOfParticlesFlex', important=True,
                      help="Particles must have a set of Zernike3D coefficients associated")
        form.addParam('structure', PointerParam, label="Atomic structure",
                      pointerClass="AtomStruct", allowsNull=True,
                      help="Optional: If provided, it will be possible to visualize "
                           "the PDB when analyzing the motion statistics")
        form.addParallelSection(threads=4, mpi=0)

    # --------------------------- INSERT steps functions ----------------------
    def _insertAllSteps(self):
        self._insertFunctionStep(self.launchVolumeViewer, interactive=True)

    # --------------------------- STEPS functions -----------------------------
    def launchVolumeViewer(self):
        particles = self.particles.get()
        mask = particles.getFlexInfo().refMask.get()

        L1 = particles.getFlexInfo().L1.get()
        L2 = particles.getFlexInfo().L2.get()

        z_space = np.asarray([particle.getZFlex() for particle in particles.iterItems()])

        if not os.path.isfile(self._getExtraPath("particles.xmd")):
            writeSetOfParticles(particles, self._getExtraPath("particles.xmd"))

            # Write extra attributes (if needed)
            md = XmippMetaData(self._getExtraPath("particles.xmd"))
            md[:, "zernikeCoefficients"] = np.asarray([",".join(item) for item in z_space.astype(str)])
            md.write(self._getExtraPath("particles.xmd"), overwrite=True)

        # Run viewer
        args = "--i %s --r %s --L1 %d --L2 %d --sr %f --thr %d" \
               % (self._getExtraPath("particles.xmd"), getXmippFileName(mask), L1, L2,
                  particles.getSamplingRate(), self.numberOfThreads.get())
        if self.structure.get():
            args += " --atom_file %s" % self.structure.get().getFileName()
        program = "viewer_map.py"
        program = flexutils.Plugin.getProgram(program)
        self.runJob(program, args)

    # --------------------------- INFO functions -----------------------------
    def _summary(self):
        pass

    def _methods(self):
        return [
            "3D visualization of motion statistics",
        ]

    # ----------------------- VALIDATE functions ----------------------------------------
    def validate(self):
        """ Try to find errors on define params. """
        errors = []
        particles = self.particles.get()
        if isinstance(particles, SetOfParticlesFlex):
            if particles.getFlexInfo().getProgName() != const.ZERNIKE3D:
                errors.append("The flexibility information associated with the particles is not "
                              "coming from the Zernike3D algorithm. Please, provide a set of particles "
                              "with the correct flexibility information.")
        return errors
