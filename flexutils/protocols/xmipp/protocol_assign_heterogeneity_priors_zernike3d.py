
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
from xmipp_metadata.metadata import XmippMetaData
from xmipp_metadata.image_handler import ImageHandler

import pyworkflow.protocol.params as params
from pyworkflow.object import Integer, Float, String, Boolean
from pyworkflow.utils.path import moveFile
from pyworkflow import VERSION_2_0

from pwem.protocols import ProtAnalysis3D, ProtFlexBase
import pwem.emlib.metadata as md
from pwem.constants import ALIGN_PROJ
from pwem.objects import ParticleFlex, SetOfParticlesFlex

from xmipp3.convert import (writeSetOfParticles, createItemMatrix,
                            setXmippAttributes, matrixFromGeometry)
import xmipp3

import flexutils.constants as const
import flexutils
from flexutils.utils import getXmippFileName


class XmippProtHeterogeneityPriorsZernike3D(ProtAnalysis3D, ProtFlexBase):
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
        form.addParam('inputPriors', params.PointerParam, label="Input priors", pointerClass="SetOfVolumesFlex",
                      help="A set of volumes with Zernike3D coefficients associated to be used as priors for "
                           "the input particles")
        form.addParam('boxSize', params.IntParam, default=128,
                      label='Downsample particles to this box size', expertLevel=params.LEVEL_ADVANCED,
                      help='In general, downsampling the particles will increase performance without compromising '
                           'the estimation the deformation field for each particle. Note that output particles will '
                           'have the original box size, and Zernike3D coefficients will be modified to work with the '
                           'original size images')
        # form.addParam('maxShift', params.FloatParam, default=-1,
        #               label='Maximum shift (px)', expertLevel=params.LEVEL_ADVANCED,
        #               help='Maximum shift allowed in pixels')
        # form.addParam('maxAngular', params.FloatParam, default=5,
        #               label='Maximum angular change (degrees)', expertLevel=params.LEVEL_ADVANCED,
        #               help='Maximum angular change allowed (in degrees)')
        form.addParam('maxResolution', params.FloatParam,
                      label='Maximum resolution (A)', expertLevel=params.LEVEL_ADVANCED,
                      allowsNull=True,
                      help='Filter the particles to this sampling rate. By default, no filter is '
                           'applied')
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
        # self.Ts = inputParticles.getSamplingRate()
        # newTs = self.targetResolution.get() * 1.0 / 3.0
        # self.newTs = max(self.Ts, newTs)
        # self.newXdim = int(Xdim * self.Ts / newTs)
        self.newXdim = self.boxSize.get()
        correctionFactor = self.newXdim / Xdim
        newTs = inputParticles.getSamplingRate() / correctionFactor
        np.savetxt(self._getExtraPath("sampling.txt"), [newTs])
        np.savetxt(self._getExtraPath("size.txt"), [self.newXdim])
        if self.newXdim != Xdim:
            self.runJob("xmipp_image_resize",
                        "-i %s -o %s --save_metadata_stack %s --fourier %d" %
                        (imgsFn,
                         self._getTmpPath('scaled_particles.stk'),
                         self._getExtraPath('scaled_particles.xmd'),
                         self.newXdim), numberOfMpi=1, env=xmipp3.Plugin.getEnviron())
            moveFile(self._getExtraPath('scaled_particles.xmd'), imgsFn)

        ih = ImageHandler()
        inputVolume = inputPriors.getFlexInfo().refMap.get()
        ih.convert(getXmippFileName(inputVolume), fnVol)
        # Xdim = self.inputParticles.get().getFirstItem().getDim()[0]
        if Xdim != self.newXdim:
            self.runJob("xmipp_image_resize",
                        "-i %s --dim %d " % (fnVol, self.newXdim), numberOfMpi=1,
                        env=xmipp3.Plugin.getEnviron())
        inputMask = inputPriors.getFlexInfo().refMask.get()
        if inputMask:
            ih.convert(getXmippFileName(inputMask), fnVolMask)
            if Xdim != self.newXdim:
                self.runJob("xmipp_image_resize",
                            "-i %s --dim %d --interp nearest" % (fnVolMask, self.newXdim), numberOfMpi=1,
                            env=xmipp3.Plugin.getEnviron())

    def alignmentStep(self):
        inputPriors = self.inputPriors.get()
        inputParticles =self.inputParticles.get()
        imgsFn = self._getFileName('imgsFn')
        fnVol = self._getFileName('fnVol')
        fnOut = self._getFileName('fnOut')
        fnVolMask = self._getFileName('fnVolMask')
        fnOutDir = self._getFileName('fnOutDir')
        fnPriors = self._getFileName('fnPriors')
        Ts = np.loadtxt(self._getExtraPath("sampling.txt"))
        maxResolution = self.maxResolution.get() if self.maxResolution.get() else Ts

        Xdim = inputParticles.getXDim()
        # self.Ts = inputParticles.getSamplingRate()
        # newTs = self.targetResolution.get() * 1.0 / 3.0
        # self.newTs = max(self.Ts, newTs)
        self.newXdim = self.boxSize.get()
        correctionFactor = self.newXdim / Xdim

        # Zernike3D parameters
        L1 = inputPriors.getFlexInfo().L1.get()
        L2 = inputPriors.getFlexInfo().L2.get()
        Rmax = correctionFactor * inputPriors.getFlexInfo().Rmax.get()

        # Write Zernike3D priors to file
        with open(fnPriors, 'w') as f:
            f.write(' '.join(map(str, [L1, L2, Rmax])) + "\n")
            for item in inputPriors.iterItems():
                z_clnm = item.getZFlex()
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
                z_clnm = item.getZFlex()
                z_clnm *= correctionFactor
                f.write(' '.join(map(str, z_clnm.reshape(-1))) + "\n")

        params = ' -i %s --ref %s -o %s ' \
                 '--l1 %d --l2 %d --sampling %f ' \
                 ' --max_resolution %f --odir %s --resume --regularization 0.0 --mask %s' \
                 ' --step 2 --blobr 2 --image_mode 1 --priors %s' %\
                 (imgsFn, fnVol, fnOut, L1, L2,
                  Ts, maxResolution, fnOutDir, fnVolMask, fnPriors)
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
        self.runJob(program, params, numberOfMpi=self.numberOfMpi.get(), env=xmipp3.Plugin.getEnviron())


    def createOutputStep(self):
        inputParticles = self.getInputParticles()
        Xdim = inputParticles.getXDim()
        self.newXdim = self.boxSize.get()
        correctionFactor = Xdim / self.newXdim
        fnOut = self._getFileName('fnOut')
        mdOut = XmippMetaData(fnOut)

        # Zernike3D info
        inputPriors = self.inputPriors.get()
        L1 = inputPriors.getFlexInfo().L1.get()
        L2 = inputPriors.getFlexInfo().L2.get()
        Rmax = inputPriors.getFlexInfo().Rmax.get()
        inputVolume = inputPriors.getFlexInfo().refMap.get()
        inputMask = inputPriors.getFlexInfo().refMask.get()

        coeffs = correctionFactor * np.asarray([np.fromstring(item, sep=',') for item in mdOut[:, "sphCoefficients"]])
        deformation = correctionFactor * mdOut[:, "sphDeformation"]
        shifts = correctionFactor * mdOut[:, ["shiftX", "shiftY", "shiftZ"]]
        angles = mdOut[:, ["angleRot", "angleTilt", "anglePsi"]]

        partSet = self._createSetOfParticlesFlex(progName=const.ZERNIKE3D)

        partSet.copyInfo(inputParticles)
        partSet.setHasCTF(inputParticles.hasCTF())
        partSet.setAlignmentProj()

        inverseTransform = partSet.getAlignment() == ALIGN_PROJ

        idx = 0
        for particle in inputParticles.iterItems():
            outParticle = ParticleFlex(progName=const.ZERNIKE3D)
            outParticle.copyInfo(particle)

            outParticle.setZFlex(coeffs[idx])
            outParticle.getFlexInfo().deformation = Float(deformation[idx])

            # Set new transformation matrix
            tr = matrixFromGeometry(shifts[idx], angles[idx], inverseTransform)
            outParticle.getTransform().setMatrix(tr)

            partSet.append(outParticle)

            idx += 1

        partSet.getFlexInfo().L1 = Integer(L1)
        partSet.getFlexInfo().L2 = Integer(L2)
        partSet.getFlexInfo().Rmax = Float(Rmax)
        partSet.getFlexInfo().refMask = String(inputMask)
        partSet.getFlexInfo().refMap = String(inputVolume)

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
        inputPriors = self.inputPriors.get()
        if inputPriors.getFlexInfo().getProgName() != const.ZERNIKE3D:
            errors.append("The flexibility information associated with the priors is not "
                          "coming from the Zernike3D algorithm. Please, provide a set of priors "
                          "with the correct flexibility information.")
        return errors





