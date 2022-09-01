
# **************************************************************************
# *
# * Authors:     Amaya Jimenez Moreno (ajimenez@cnb.csic.es)
# *              David Herreros Calero (dherreos@cnb.csic.es)
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

from xmipp3.convert import createItemMatrix, setXmippAttributes, writeSetOfImages, imageToRow, coordinateToRow
from xmipp3.base import writeInfoField, readInfoField
import xmipp3

import flexutils
import flexutils.constants as const


class XmippProtAngularAlignmentZernike3D(ProtAnalysis3D):
    """ Protocol for flexible angular alignment based on Zernike3D basis. """
    _label = 'angular align - Zernike3D'
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
        form.addParam('inputVolume', params.PointerParam, label="Input volume", pointerClass='Volume',
                      condition="inputParticles and not hasattr(inputParticles,'refMap')")
        form.addParam('inputVolumeMask', params.PointerParam, label="Input volume mask", pointerClass='VolumeMask',
                      condition="inputParticles and not hasattr(inputParticles,'refMask')")
        form.addParam('boxSize', params.IntParam, default=128,
                      label='Downsample particles to this box size', expertLevel=params.LEVEL_ADVANCED,
                      help='In general, downsampling the particles will increase performance without compromising '
                           'the estimation the deformation field for each particle. Note that output particles will '
                           'have the original box size, and Zernike3D coefficients will be modified to work with the '
                           'original size images')
        form.addParam('l1', params.IntParam, default=3,
                      label='Zernike Degree',
                      expertLevel=params.LEVEL_ADVANCED,
                      condition="inputParticles and not hasattr(inputParticles,'L1')",
                      help='Degree Zernike Polynomials of the deformation=1,2,3,...')
        form.addParam('l2', params.IntParam, default=2,
                      label='Harmonical Degree',
                      condition="inputParticles and not hasattr(inputParticles,'L2')",
                      expertLevel=params.LEVEL_ADVANCED,
                      help='Degree Spherical Harmonics of the deformation=1,2,3,...')
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
        form.addParam('regularization', params.FloatParam, default=0.005, label='Regularization',
                      expertLevel=params.LEVEL_ADVANCED,
                      help='Penalization to deformations (higher values penalize more the deformation).')
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
        fnPriors = self._getFileName('fnPriors')

        inputParticles = self.inputParticles.get()
        Xdim = inputParticles.getXDim()
        # self.Ts = inputParticles.getSamplingRate()
        # newTs = self.targetResolution.get() * 1.0 / 3.0
        # self.newTs = max(self.Ts, newTs)
        # self.newXdim = int(Xdim * self.Ts / newTs)
        self.newXdim = self.boxSize.get()
        correctionFactor = self.newXdim / Xdim
        newTs = inputParticles.getSamplingRate() / correctionFactor

        ih = ImageHandler()
        inputVolume = inputParticles.refMap.get() if hasattr(inputParticles, 'refMap') else self.inputVolume.get()
        ih.convert(inputVolume, fnVol)
        # Xdim = self.inputVolume.get().getDim()[0]
        if Xdim != self.newXdim:
            self.runJob("xmipp_image_resize",
                        "-i %s --dim %d " % (fnVol, self.newXdim), numberOfMpi=1, env=xmipp3.Plugin.getEnviron())
        inputMask = inputParticles.refMask.get() if hasattr(inputParticles, 'refMask') else self.inputVolumeMask.get()
        if inputMask:
            ih.convert(inputMask, fnVolMask)
            if Xdim != self.newXdim:
                self.runJob("xmipp_image_resize",
                            "-i %s --dim %d --interp nearest" % (fnVolMask, self.newXdim), numberOfMpi=1,
                            env=xmipp3.Plugin.getEnviron())

        if hasattr(inputParticles.getFirstItem(), '_xmipp_sphCoefficients'):
            L1 = inputParticles.L1.get()
            L2 = inputParticles.L2.get()
            Rmax = correctionFactor * inputParticles.Rmax.get()
            z_clnm_vec = {}

            with open(fnPriors, 'w') as f:
                f.write(' '.join(map(str, [L1, L2, Rmax])) + "\n")
                for particle in inputParticles.iterItems():
                    z_clnm = np.fromstring(particle._xmipp_sphCoefficients.get(), sep=",")
                    f.write(' '.join(map(str, z_clnm.reshape(-1))) + "\n")
                    z_clnm *= correctionFactor
                    z_clnm_vec[particle.getObjId()] = z_clnm.reshape(-1)

            # Compute deformations
            def_file = self._getExtraPath("def_file.txt")
            args = "--i %s --z_clnm %s --o %s" % (fnVolMask, fnPriors, def_file)
            program = os.path.join(const.XMIPP_SCRIPTS, "compute_z_clnm_deformation.py")
            program = flexutils.Plugin.getProgram(program)
            self.runJob(program, args, numberOfMpi=1)
            deformations = np.loadtxt(def_file)

        def zernikeRow(part, partRow, **kwargs):
            imageToRow(part, partRow, md.MDL_IMAGE, **kwargs)
            coord = part.getCoordinate()
            idx = part.getObjId()
            if coord is not None:
                coordinateToRow(coord, partRow, copyId=False)
            if part.hasMicId():
                partRow.setValue(md.MDL_MICROGRAPH_ID, int(part.getMicId()))
                partRow.setValue(md.MDL_MICROGRAPH, str(part.getMicId()))
            if hasattr(part, '_xmipp_sphCoefficients') and hasattr(part, '_xmipp_sphDeformation'):
                partRow.setValue(md.MDL_SPH_COEFFICIENTS, z_clnm_vec[idx].tolist())
                idx = list(z_clnm_vec.keys()).index(idx)
                partRow.setValue(md.MDL_SPH_DEFORMATION, deformations[idx])

        writeSetOfImages(inputParticles, imgsFn, zernikeRow)
        writeInfoField(self._getExtraPath(), "sampling", md.MDL_SAMPLINGRATE, newTs)
        writeInfoField(self._getExtraPath(), "size", md.MDL_XSIZE, self.newXdim)
        if self.newXdim != Xdim:
            params = "-i %s -o %s --save_metadata_stack %s --fourier %d" % \
                     (imgsFn,
                     self._getExtraPath('scaled_particles.stk'),
                     self._getExtraPath('scaled_particles.xmd'),
                     self.newXdim)
            if self.numberOfMpi.get() > 1:
                params += " --mpi_job_size %d" % int(inputParticles.getSize() / self.numberOfMpi.get())
            self.runJob("xmipp_image_resize", params, numberOfMpi=self.numberOfMpi.get(),
                        env=xmipp3.Plugin.getEnviron())
            moveFile(self._getExtraPath('scaled_particles.xmd'), imgsFn)

    def alignmentStep(self):
        inputParticles = self.inputParticles.get()
        imgsFn = self._getFileName('imgsFn')
        fnVol = self._getFileName('fnVol')
        fnOut = self._getFileName('fnOut')
        fnVolMask = self._getFileName('fnVolMask')
        fnOutDir = self._getFileName('fnOutDir')
        Ts = readInfoField(self._getExtraPath(), "sampling", md.MDL_SAMPLINGRATE)
        L1 = inputParticles.L1.get() if hasattr(inputParticles, 'L1') else self.l1.get()
        L2 = inputParticles.L2.get() if hasattr(inputParticles, 'L2') else self.l2.get()
        maxResolution = self.maxResolution.get() if self.maxResolution.get() else Ts
        params = ' -i %s --ref %s -o %s --optimizeDeformation ' \
                 '--l1 %d --l2 %d --sampling %f ' \
                 ' --max_resolution %f --odir %s --resume --regularization %f --mask %s' \
                 ' --step 2 --blobr 2 --image_mode 1' %\
                 (imgsFn, fnVol, fnOut, L1, L2,
                  Ts, maxResolution, fnOutDir, self.regularization.get(), fnVolMask)
        if not self.ignoreCTF.get():
            params += ' --useCTF'
        if self.inputParticles.get().isPhaseFlipped():
            params += ' --phaseFlipped'

        # if self.useGpu.get():
        #     params += ' --device %d' % self.getGpuList()[0]
        #     program = 'xmipp_cuda_angular_sph_alignment'
        #     self.runJob(program, params)
        # else:
        program = 'xmipp_forward_zernike_images'
        self.runJob(program, params, numberOfMpi=self.numberOfMpi.get(),
                    env=xmipp3.Plugin.getEnviron())


    def createOutputStep(self):
        inputParticles = self.inputParticles.get()
        Xdim = inputParticles.getXDim()
        # self.Ts = inputParticles.getSamplingRate()
        # newTs = self.targetResolution.get() * 1.0 / 3.0
        # self.newTs = max(self.Ts, newTs)
        # self.newXdim = int(Xdim * self.Ts / newTs)
        self.newXdim = self.boxSize.get()
        fnOut = self._getFileName('fnOut')
        mdOut = md.MetaData(fnOut)

        newMdOut = md.MetaData()
        i = 0
        for row in md.iterRows(mdOut):
            newRow = row
            if self.newXdim != Xdim:
                coeffs = mdOut.getValue(md.MDL_SPH_COEFFICIENTS, row.getObjId())
                correctionFactor = Xdim / self.newXdim
                deformation = mdOut.getValue(md.MDL_SPH_DEFORMATION, row.getObjId())
                coeffs = [correctionFactor * coeff for coeff in coeffs]
                newRow.setValue(md.MDL_SPH_COEFFICIENTS, coeffs)
                newRow.setValue(md.MDL_SPH_DEFORMATION, correctionFactor * deformation)
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
        inputMask = inputParticles.refMask.get() if hasattr(inputParticles, 'refMask') else self.inputVolumeMask.get().getFileName()
        inputVolume = inputParticles.refMap.get() if hasattr(inputParticles, 'refMap') else self.inputVolume.get().getFileName()

        partSet.copyInfo(inputSet)
        partSet.setAlignmentProj()
        partSet.copyItems(inputSet,
                          updateItemCallback=self._updateParticle,
                          itemDataIterator=md.iterRows(fnOut, sortByLabel=md.MDL_ITEM_ID))
        partSet.L1 = Integer(self.l1.get())
        partSet.L2 = Integer(self.l2.get())
        partSet.Rmax = Float(Xdim / 2)
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
        inputParticles = self.inputParticles.get()
        if not hasattr(inputParticles, 'L1') and hasattr(inputParticles, 'L2'):
            l1 = self.l1.get()
            l2 = self.l2.get()
            if (l1 - l2) < 0:
                errors.append('Zernike degree must be higher than '
                              'SPH degree.')
        return errors





