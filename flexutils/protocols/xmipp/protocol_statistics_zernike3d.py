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

from pyworkflow import BETA
from pyworkflow.protocol.params import PointerParam, IntParam

from pwem.protocols import ProtAnalysis3D

import flexutils
import flexutils.constants as const

import pwem.emlib.metadata as md
from xmipp3.convert import writeSetOfImages, imageToRow, coordinateToRow


class XmippProtStatisticsZernike3D(ProtAnalysis3D):
    """ 3D visualization of motion statistics """

    _label = 'motion statistics - Zernike3D'
    _devStatus = BETA
    OUTPUT_PREFIX = 'zernike3DClasses'

    # --------------------------- DEFINE param functions ----------------------
    def _defineParams(self, form):
        form.addSection(label='General parameters')
        form.addParam('particles', PointerParam, label="Zernike3D particles",
                      pointerClass='SetOfParticles', important=True,
                      help="Particles must have a set of Zernike3D coefficients associated")
        form.addParam('mask', PointerParam, label="Motion mask",
                      condition="particles and not hasattr(particles,'refMap')",
                      pointerClass='VolumeMask', important=True,
                      help="Mask determining where to analyze the motions")
        form.addParam('L1', IntParam, label="Zernike degree",
                      condition="particles and not hasattr(particles,'L1')",
                      important=True,
                      help="Zernike degree used to compute the particle's deformation"
                           "field")
        form.addParam('L2', IntParam, label="Sph degree",
                      condition="particles and not hasattr(particles,'L2')",
                      important=True,
                      help="Sph degree used to compute the particle's deformation"
                           "field")
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
        mask = particles.refMask.get() if hasattr(particles, "refMask") else self.mask.get().getFileName()

        L1 = particles.L1.get() if hasattr(particles, "L1") else self.L1.get()
        L2 = particles.L2.get() if hasattr(particles, "L2") else self.L2.get()
        z_clnm_vec = {}
        def_vec = {}

        for particle in particles.iterItems():
            z_clnm = np.fromstring(particle._xmipp_sphCoefficients.get(), sep=",")
            z_clnm_vec[particle.getObjId()] = z_clnm.reshape(-1)
            def_vec[particle.getObjId()] = particle._xmipp_sphDeformation.get()

        def zernikeRow(part, partRow, **kwargs):
            imageToRow(part, partRow, md.MDL_IMAGE, **kwargs)
            coord = part.getCoordinate()
            idx = part.getObjId()
            if coord is not None:
                coordinateToRow(coord, partRow, copyId=False)
            if part.hasMicId():
                partRow.setValue(md.MDL_MICROGRAPH_ID, int(part.getMicId()))
                partRow.setValue(md.MDL_MICROGRAPH, str(part.getMicId()))
            partRow.setValue(md.MDL_SPH_COEFFICIENTS, z_clnm_vec[idx].tolist())
            partRow.setValue(md.MDL_SPH_DEFORMATION, def_vec[idx])

        if not os.path.isfile(self._getExtraPath("particles.xmd")):
            writeSetOfImages(particles, self._getExtraPath("particles.xmd"), zernikeRow)

        # Run viewer
        args = "--i %s --r %s --L1 %d --L2 %d --sr %f --thr %d" \
               % (self._getExtraPath("particles.xmd"), mask, L1, L2,
                  particles.getSamplingRate(), self.numberOfThreads.get())
        if self.structure.get():
            args += " --atom_file %s" % self.structure.get().getFileName()
        program = os.path.join(const.VIEWERS, "viewer_map.py")
        program = flexutils.Plugin.getProgram(program)
        self.runJob(program, args)

    # --------------------------- INFO functions -----------------------------
    def _summary(self):
        pass

    def _methods(self):
        return [
            "3D visualization of motion statistics",
        ]
