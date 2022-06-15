
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
from pyworkflow.object import Integer, Float, String
from pyworkflow.utils.path import moveFile
from pyworkflow import VERSION_2_0

from pwem.protocols import ProtAnalysis3D
import pwem.emlib.metadata as md
from pwem.emlib.image import ImageHandler
from pwem.constants import ALIGN_PROJ

from xmipp3.convert import (writeSetOfParticles, createItemMatrix,
                            setXmippAttributes)
from xmipp3.base import writeInfoField, readInfoField

import flexutils.constants as const
import flexutils

class XmippProtHeterogeneityPriorsZernike3D(ProtAnalysis3D):
    """ Assignation of heterogeneity priors based on the Zernike3D basis. """
    _label = 'assign heterogeneity priors - Zernike3D'
    _lastUpdateVersion = VERSION_2_0

    # --------------------------- DEFINE param functions --------------------------------------------
    def _defineParams(self, form):
        form.addSection(label='Input')
        # form.addHidden(params.USE_GPU, params.BooleanParam, default=True,
        #                label="Use GPU for execution",
        #                help="This protocol has both CPU and GPU implementation.\
        #                    Select the one you want to use.")
        # form.addHidden(params.GPU_LIST, params.StringParam, default='0',
        #                expertLevel=params.LEVEL_ADVANCED,
        #                label="Choose GPU IDs",
        #                help="Add a list of GPU devices that can be used")
        form.addParam('inputParticles', params.PointerParam, label="Input particles", pointerClass='SetOfParticles')
        form.addParam('inputPriors', params.PointerParam, label="Input priors", pointerClass="SetOfVolumes",
                      help="A set of volumes with Zernike3D coefficients associated to be used as priors for "
                           "the input particles")
        form.addParam('inputVolume', params.PointerParam, label="Input volume", pointerClass='Volume',
                      condition="inputPriors and not hasattr(inputPriors,'refMap')")
        form.addParam('inputVolumeMask', params.PointerParam, label="Input volume mask", pointerClass='VolumeMask',
                      condition="inputPriors and not hasattr(inputPriors,'refMask')")
        form.addParam('targetResolution', params.FloatParam, label="Target resolution (A)", default=8.0,
                      help="In Angstroms, the images and the volume are rescaled so that this resolution is at "
                           "2/3 of the Fourier spectrum.")
        # form.addParam('maxShift', params.FloatParam, default=-1,
        #               label='Maximum shift (px)', expertLevel=params.LEVEL_ADVANCED,
        #               help='Maximum shift allowed in pixels')
        # form.addParam('maxAngular', params.FloatParam, default=5,
        #               label='Maximum angular change (degrees)', expertLevel=params.LEVEL_ADVANCED,
        #               help='Maximum angular change allowed (in degrees)')
        form.addParam('maxResolution', params.FloatParam, default=4.0,
                      label='Maximum resolution (A)', expertLevel=params.LEVEL_ADVANCED,
                      help='Maximum resolution (A)')
        # form.addParam('regularization', params.FloatParam, default=0.005, label='Regularization',
        #               expertLevel=params.LEVEL_ADVANCED,
        #               help='Penalization to deformations (higher values penalize more the deformation).')
        form.addParam('ignoreCTF', params.BooleanParam, default=False, label='Ignore CTF?',
                      expertLevel=params.LEVEL_ADVANCED,
                      help="If true, volume projection won't be subjected to CTF corrections")
        # form.addParam('optimizeAlignment', params.BooleanParam, default=True, label='Optimize alignment?',
        #              expertLevel=params.LEVEL_ADVANCED)
        form.addParallelSection(threads=1, mpi=8)

    def _createFilenameTemplates(self):
        """ Centralize how files are called """
        myDict = {
            'imgsFn': self._getExtraPath('input_particles.xmd'),
            'fnVol': self._getExtraPath('input_volume.vol'),
            'fnVolMask': self._getExtraPath('input_volume_mask.vol'),
            'fnOut': self._getExtraPath('sphDone.xmd'),
            'fnPriors': self._getExtraPath("priors.txt"),
            'fnOutDir': self._getExtraPath()
                 }
        self._updateFilenamesDict(myDict)

    # --------------------------- INSERT steps functions --------------------------------------------
    def _insertAllSteps(self):
        self._createFilenameTemplates()
        self._insertFunctionStep("convertStep")
        self._insertFunctionStep("alignmentStep")
        self._insertFunctionStep("createOutputStep")

    # --------------------------- STEPS functions ---------------------------------------------------
    def convertStep(self):
        imgsFn = self._getFileName('imgsFn')
        fnVol = self._getFileName('fnVol')
        fnVolMask = self._getFileName('fnVolMask')

        inputParticles = self.inputParticles.get()
        inputPriors = self.inputPriors.get()
        writeSetOfParticles(inputParticles, imgsFn)
        Xdim = inputParticles.getXDim()
        self.Ts = inputParticles.getSamplingRate()
        newTs = self.targetResolution.get() * 1.0 / 3.0
        self.newTs = max(self.Ts, newTs)
        self.newXdim = int(Xdim * self.Ts / newTs)
        writeInfoField(self._getExtraPath(), "sampling", md.MDL_SAMPLINGRATE, newTs)
        writeInfoField(self._getExtraPath(), "size", md.MDL_XSIZE, self.newXdim)
        if self.newXdim != Xdim:
            self.runJob("xmipp_image_resize",
                        "-i %s -o %s --save_metadata_stack %s --fourier %d" %
                        (imgsFn,
                         self._getExtraPath('scaled_particles.stk'),
                         self._getExtraPath('scaled_particles.xmd'),
                         self.newXdim), numberOfMpi=1)
            moveFile(self._getExtraPath('scaled_particles.xmd'), imgsFn)

        ih = ImageHandler()
        inputVolume = inputPriors.refMap.get() if hasattr(inputPriors, 'refMap') else self.inputVolume.get()
        ih.convert(inputVolume, fnVol)
        # Xdim = self.inputParticles.get().getFirstItem().getDim()[0]
        if Xdim != self.newXdim:
            self.runJob("xmipp_image_resize",
                        "-i %s --dim %d " % (fnVol, self.newXdim), numberOfMpi=1)
        inputMask = inputPriors.refMask.get() if hasattr(inputPriors, 'refMask') else self.inputVolumeMask.get()
        if inputMask:
            ih.convert(inputMask, fnVolMask)
            if Xdim != self.newXdim:
                self.runJob("xmipp_image_resize",
                            "-i %s --dim %d --interp nearest" % (fnVolMask, self.newXdim), numberOfMpi=1)

    def alignmentStep(self):
        inputPriors = self.inputPriors.get()
        inputParticles =self.inputParticles.get()
        imgsFn = self._getFileName('imgsFn')
        fnVol = self._getFileName('fnVol')
        fnOut = self._getFileName('fnOut')
        fnVolMask = self._getFileName('fnVolMask')
        fnOutDir = self._getFileName('fnOutDir')
        fnPriors = self._getFileName('fnPriors')
        Ts = readInfoField(self._getExtraPath(), "sampling", md.MDL_SAMPLINGRATE)

        Xdim = inputParticles.getXDim()
        self.Ts = inputParticles.getSamplingRate()
        newTs = self.targetResolution.get() * 1.0 / 3.0
        self.newTs = max(self.Ts, newTs)
        self.newXdim = int(Xdim * self.Ts / newTs)
        correctionFactor = self.newXdim / Xdim

        # Zernike3D parameters
        L1 = inputPriors.L1.get()
        L2 = inputPriors.L2.get()
        Rmax = correctionFactor * inputPriors.Rmax.get()

        # Write Zernike3D priors to file
        with open(fnPriors, 'w') as f:
            f.write(' '.join(map(str, [L1, L2, Rmax])) + "\n")
            for item in inputPriors.iterItems():
                z_clnm = np.fromstring(item._xmipp_sphCoefficients.get(), sep=",")
                z_clnm *= correctionFactor
                f.write(' '.join(map(str, z_clnm.reshape(-1))) + "\n")

        # Compute deformations
        def_file = self._getExtraPath("def_file.txt")
        args = "--i %s --z_clnm %s --o %s" % (fnVolMask, fnPriors, def_file)
        program = os.path.join(const.XMIPP_SCRIPTS, "compute_z_clnm_deformation.py")
        program = flexutils.Plugin.getProgram(program)
        self.runJob(program, args, numberOfMpi=1)
        deformations = np.loadtxt(def_file)

        # Write Zernike3D priors to file
        with open(fnPriors, 'w') as f:
            if deformations.size == 1:
                f.write(' '.join(map(str, [L1, L2, Rmax, deformations])) + "\n")
            else:
                f.write(' '.join(map(str, [L1, L2, Rmax] + deformations.tolist())) + "\n")
            for item in inputPriors.iterItems():
                z_clnm = np.fromstring(item._xmipp_sphCoefficients.get(), sep=",")
                z_clnm *= correctionFactor
                f.write(' '.join(map(str, z_clnm.reshape(-1))) + "\n")

        params = ' -i %s --ref %s -o %s ' \
                 '--l1 %d --l2 %d --sampling %f ' \
                 ' --max_resolution %f --odir %s --resume --regularization 0.0 --mask %s' \
                 ' --step 2 --blobr 2 --image_mode 1 --priors %s' %\
                 (imgsFn, fnVol, fnOut, L1, L2,
                  Ts, self.maxResolution, fnOutDir, fnVolMask, fnPriors)
        if not self.ignoreCTF.get():
            params += ' --useCTF'
        if self.inputParticles.get().isPhaseFlipped():
            params += ' --phaseFlipped'

        # if self.useGpu.get():
        #     params += ' --device %d' % self.getGpuList()[0]
        #     program = 'xmipp_cuda_angular_sph_alignment'
        #     self.runJob(program, params)
        # else:
        program = 'xmipp_forward_zernike_images_priors'
        self.runJob(program, params, numberOfMpi=self.numberOfMpi.get())


    def createOutputStep(self):
        Xdim = self.inputParticles.get().getXDim()
        self.Ts = self.inputParticles.get().getSamplingRate()
        newTs = self.targetResolution.get() * 1.0 /3.0
        self.newTs = max(self.Ts, newTs)
        self.newXdim = int(Xdim * self.Ts / newTs)
        fnOut = self._getFileName('fnOut')
        mdOut = md.MetaData(fnOut)

        # Zernike3D info
        inputPriors = self.inputPriors.get()
        L1 = inputPriors.L1.get()
        L2 = inputPriors.L2.get()
        Rmax = inputPriors.Rmax.get()
        inputVolume = inputPriors.refMap.get() if hasattr(inputPriors, 'refMap') else self.inputVolume.get()
        inputMask = inputPriors.refMask.get() if hasattr(inputPriors, 'refMask') else self.inputVolume.get()

        newMdOut = md.MetaData()
        i = 0
        for row in md.iterRows(mdOut):
            newRow = row
            if self.newTs != self.Ts:
                coeffs = mdOut.getValue(md.MDL_SPH_COEFFICIENTS, row.getObjId())
                correctionFactor = self.inputVolume.get().getDim()[0] / self.newXdim
                coeffs = [correctionFactor * coeff for coeff in coeffs]
                newRow.setValue(md.MDL_SPH_COEFFICIENTS, coeffs)
                shiftX = correctionFactor * mdOut.getValue(md.MDL_SHIFT_X, row.getObjId())
                shiftY = correctionFactor * mdOut.getValue(md.MDL_SHIFT_Y, row.getObjId())
                shiftZ = correctionFactor * mdOut.getValue(md.MDL_SHIFT_Z, row.getObjId())
                newRow.setValue(md.MDL_SHIFT_X, shiftX)
                newRow.setValue(md.MDL_SHIFT_Y, shiftY)
                newRow.setValue(md.MDL_SHIFT_Z, shiftZ)
            newRow.addToMd(newMdOut)
            i += 1
        newMdOut.write(fnOut)

        inputSet = self.inputParticles.get()
        partSet = self._createSetOfParticles()

        partSet.copyInfo(inputSet)
        partSet.setAlignmentProj()
        partSet.copyItems(inputSet,
                          updateItemCallback=self._updateParticle,
                          itemDataIterator=md.iterRows(fnOut, sortByLabel=md.MDL_ITEM_ID))
        partSet.L1 = Integer(L1)
        partSet.L2 = Integer(L2)
        partSet.Rmax = Float(Rmax)
        partSet.refMask = String(inputMask)
        partSet.refMap = String(inputVolume)

        self._defineOutputs(outputParticles=partSet)
        self._defineTransformRelation(self.inputParticles, partSet)


# --------------------------- UTILS functions --------------------------------------------
    def _updateParticle(self, item, row):
        setXmippAttributes(item, row, md.MDL_ANGLE_ROT, md.MDL_ANGLE_TILT,
                           md.MDL_ANGLE_PSI, md.MDL_SHIFT_X, md.MDL_SHIFT_Y,
                           md.MDL_FLIP, md.MDL_SPH_DEFORMATION,
                           md.MDL_SPH_COEFFICIENTS)
        createItemMatrix(item, row, align=ALIGN_PROJ)

    def getInputParticles(self):
        return self.inputParticles.get()

    # ----------------------- VALIDATE functions ----------------------------------------
    def validate(self):
        """ Try to find errors on define params. """
        errors = []
        if not hasattr(self.inputPriors.get().getFirstItem(), "_xmipp_sphCoefficients"):
            errors.append("Priors provided do not contain any Zernike3D prior that can "
                          "be used")
        return errors





