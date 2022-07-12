
# **************************************************************************
# *
# * Authors:     David Herreros Calero (dherreos@cnb.csic.es)
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
import os

import pyworkflow.protocol.params as params
from pyworkflow.object import Integer, String
from pyworkflow import VERSION_2_0

from pwem.protocols import ProtAnalysis3D
import pwem.emlib.metadata as md
from pwem.constants import ALIGN_PROJ

from xmipp3.convert import writeSetOfImages, imageToRow, coordinateToRow, setXmippAttributes, createItemMatrix

import flexutils.constants as const
import flexutils


class XmippProtReassignReferenceZernike3D(ProtAnalysis3D):
    """ Assignation of heterogeneity priors based on the Zernike3D basis. """
    _label = 'reassign reference - Zernike3D'
    _lastUpdateVersion = VERSION_2_0

    # --------------------------- DEFINE param functions --------------------------------------------
    def _defineParams(self, form):
        form.addSection(label='Input')
        form.addParam('inputParticles', params.PointerParam, label="Input particles", pointerClass='SetOfParticles')
        form.addParam('inmap', params.PointerParam, label="Particles map", pointerClass='Volume',
                      help="Map used for the computation of the Zernike3D coefficients associated "
                           "to the input particles",
                      condition="inputParticles and not hasattr(inputParticles,'refMap')")
        form.addParam('inmask', params.PointerParam, label="Particles mask", pointerClass='VolumeMask',
                      help="Mask used for the computation of the Zernike3D coefficients associated "
                           "to the input particles",
                      condition="inputParticles and not hasattr(inputParticles,'refMask')")
        form.addParam('inputVolume', params.PointerParam, label="New reference", pointerClass='Volume',
                      help="Particles will be reassigned so this map is their new reference")
        form.addParam('mask', params.PointerParam, label="Reference mask", pointerClass='VolumeMask',
                      help="Mask associated to the new reference map")
        form.addParam('L1', params.IntParam, label="Zernike degree", expertLevel=params.LEVEL_ADVANCED,
                      default=3,
                      help="Zernike polynomial degree for the new focused Zernike3D coefficients")
        form.addParam('L2', params.IntParam, label="Spherical harmonic degree", expertLevel=params.LEVEL_ADVANCED,
                      default=2,
                      help="Spherical harmonics degree for the new focused Zernike3D coefficients")
        # form.addParallelSection(threads=1, mpi=8)

    # --------------------------- INSERT steps functions --------------------------------------------
    def _insertAllSteps(self):
        self._insertFunctionStep(self.computeZernikeStep)
        self._insertFunctionStep(self.createOutputStep)

    # --------------------------- STEPS functions ---------------------------------------------------
    def computeZernikeStep(self):
        imgsFn = self._getExtraPath('inputParticles.xmd')
        particles = self.inputParticles.get()
        refMask = particles.refMask.get() if hasattr(particles, 'refMask') else self.inmask.get().getFileName()
        prevL1 = particles.L1.get()
        prevL2 = particles.L2.get()
        Rmax = particles.Rmax.get()

        newRefMap = self.inputVolume.get()
        newRefMask = self.mask.get().getFileName()
        L1 = self.L1.get()
        L2 = self.L2.get()

        z_clnm_vec = {}
        deformation_vec = {}
        for particle in particles.iterItems():
            z_clnm = np.fromstring(particle._xmipp_sphCoefficients.get(), sep=",")
            z_clnm_vec[particle.getObjId()] = z_clnm.reshape(-1)
            deformation_vec[particle.getObjId()] = particle._xmipp_sphDeformation.get()

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
            partRow.setValue(md.MDL_SPH_DEFORMATION, deformation_vec[idx])

        writeSetOfImages(particles, imgsFn, zernikeRow)

        # Write Zernike3D priors to file
        file_zclnm_r = self._getExtraPath('z_clnm_r,txt')
        with open(file_zclnm_r, 'w') as f:
            f.write(' '.join(map(str, [L1, L2, newRefMap.Rmax.get()])) + "\n")
            z_clnm = np.fromstring(newRefMap._xmipp_sphCoefficients.get(), sep=",")
            f.write(' '.join(map(str, z_clnm.reshape(-1))) + "\n")

        args = "--i %s --maski %s --maskr %s --zclnm_r %s --prevl1 %d --prevl2 %d --l1 %d --l2 %d --rmax %f" \
               % (imgsFn, refMask, newRefMask, file_zclnm_r, prevL1, prevL2, L1, L2, Rmax)
        program = os.path.join(const.XMIPP_SCRIPTS, "reassign_reference.py")
        program = flexutils.Plugin.getProgram(program)
        self.runJob(program, args)

    def createOutputStep(self):
        inputSet = self.inputParticles.get()
        partSet = self._createSetOfParticles()

        partSet.copyInfo(inputSet)
        partSet.setAlignmentProj()
        partSet.copyItems(inputSet,
                          updateItemCallback=self._updateParticle,
                          itemDataIterator=md.iterRows(self._getExtraPath("inputParticles_reassigned.xmd"), sortByLabel=md.MDL_ITEM_ID))

        partSet.L1 = Integer(self.L1.get())
        partSet.L2 = Integer(self.L2.get())
        partSet.Rmax = inputSet.Rmax
        partSet.refMask = String(self.mask.get().getFileName())
        partSet.refMap = String(self.inputVolume.get().getFileName())

        self._defineOutputs(outputParticles=partSet)
        self._defineTransformRelation(self.inputParticles, partSet)

    # --------------------------- UTILS functions --------------------------------------------
    def _updateParticle(self, item, row):
        setXmippAttributes(item, row, md.MDL_ANGLE_ROT, md.MDL_ANGLE_TILT,
                           md.MDL_ANGLE_PSI, md.MDL_SHIFT_X, md.MDL_SHIFT_Y,
                           md.MDL_FLIP, md.MDL_SPH_DEFORMATION,
                           md.MDL_SPH_COEFFICIENTS)
        createItemMatrix(item, row, align=ALIGN_PROJ)

    # ----------------------- VALIDATE functions ----------------------------------------
    def validate(self):
        """ Try to find errors on define params. """
        errors = []
        if not hasattr(self.inputParticles.get().getFirstItem(), "_xmipp_sphCoefficients"):
            errors.append("Particles must have Zernike3D coefficients associated to them")
        if not hasattr(self.inputVolume.get(), "_xmipp_sphCoefficients"):
            errors.append("New reference map must have Zernike3D coefficients associated to it")
        return errors





