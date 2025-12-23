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


from pyworkflow import NEW
from pyworkflow.protocol.params import PointerParam
from pyworkflow.object import Boolean

from pwem.protocols import ProtAnalysis3D, ProtFlexBase


class ProtFlexAssociateSpace(ProtAnalysis3D, ProtFlexBase):
    """ Associate flexible spaces and info to a different set of particles """

    _label = 'associate flex space to particles'
    _devStatus = NEW
    OUTPUT_PREFIX = 'flexParticles'

    # --------------------------- DEFINE param functions ----------------------
    def _defineParams(self, form):
        form.addSection(label='General parameters')
        form.addParam('particlesFlex', PointerParam, label="Particles with flex info",
                      pointerClass='SetOfParticlesFlex', important=True,
                      help="Particles must have a flexibility information associated (Zernike3D, CryoDrgn...")
        form.addParam('particles', PointerParam, label="Particles to associate",
                      pointerClass="SetOfParticles", important=True,
                      help='Particles to associate flexibility information (space and other attributes)')

    # --------------------------- INSERT steps functions ----------------------
    def _insertAllSteps(self):
        self._insertFunctionStep(self._createOutputStep)

    # --------------------------- STEPS functions -----------------------------
    def _createOutputStep(self):
        particlesFlex = self.particlesFlex.get()
        particles = self.particles.get()

        outSet = self._createSetOfParticlesFlex()
        outSet.copyInfo(particlesFlex)
        outSet.setHasCTF(particlesFlex.hasCTF())

        for particleFlex, particle in zip(particlesFlex.iterItems(), particles.iterItems()):
            particleFlex.setLocation(particle.getLocation())
            outSet.append(particleFlex)

        # Save new output
        name = self.OUTPUT_PREFIX
        args = {}
        args[name] = outSet
        self._defineOutputs(**args)
        self._defineSourceRelation(particlesFlex, outSet)
        self._defineSourceRelation(particles, outSet)

    # --------------------------- INFO functions -----------------------------
    def _summary(self):
        summary = []
        if self.getOutputsSize() >= 1:
            summary.append("Particles successfully associated")
        else:
            summary.append("Output annotated classes not ready yet")
        return summary

    def _methods(self):
        return [
            "Associate flexiblity information to a different set of particles",
        ]

    # ----------------------- VALIDATE functions ----------------------------------------
    def validate(self):
        """ Try to find errors on define params. """
        errors = []
        return errors
