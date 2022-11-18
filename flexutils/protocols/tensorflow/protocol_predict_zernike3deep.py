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
import h5py

import pyworkflow.protocol.params as params
from pyworkflow.object import Integer, Float, String, CsvList
from pyworkflow.utils.path import moveFile
from pyworkflow import VERSION_2_0

from pwem.protocols import ProtAnalysis3D
import pwem.emlib.metadata as md
from pwem.emlib.image import ImageHandler
from pwem.constants import ALIGN_PROJ

from xmipp3.convert import createItemMatrix, setXmippAttributes, imageToRow, coordinateToRow, writeSetOfParticles
import xmipp3

import flexutils
import flexutils.constants as const
from flexutils.utils import getXmippFileName


class TensorflowProtPredictZernike3Deep(ProtAnalysis3D):
    """ Predict Zernike3D coefficents for a set of particles based on a trained
     Zernike3Deep network. """
    _label = 'predict - Zernike3Deep'
    _lastUpdateVersion = VERSION_2_0

    # --------------------------- DEFINE param functions --------------------------------------------
    def _defineParams(self, form):
        form.addSection(label='Input')
        form.addHidden(params.USE_GPU, params.BooleanParam, default=True,
                       label="Use GPU for execution",
                       help="This protocol has both CPU and GPU implementation.\
                             Select the one you want to use.")
        form.addHidden(params.GPU_LIST, params.StringParam, default='0',
                       expertLevel=params.LEVEL_ADVANCED,
                       label="Choose GPU IDs",
                       help="Add a list of GPU devices that can be used")
        form.addParam('inputParticles', params.PointerParam, label="Input particles to predict",
                      pointerClass='SetOfParticles')
        form.addParam('zernikeParticles', params.PointerParam, label="Zernike3Deep particles",
                      pointerClass='SetOfParticles',
                      help="Particles coming out from the protocol: 'angular align - Zernike3Deep'. "
                           "This particles store all the information needed to load the trained "
                           "Zernike3Deep network")
        form.addParam('boxSize', params.IntParam, default=128,
                      label='Downsample particles to this box size', expertLevel=params.LEVEL_ADVANCED,
                      help="Should match the boxSize applied during the 'angular align - Zernike3Deep' "
                           "execution")
        form.addParam('unStack', params.BooleanParam, default=True, label='Unstack images?',
                      expertLevel=params.LEVEL_ADVANCED,
                      help="If true, images stored in the metadata will be unstacked to save GPU "
                           "memory during the training steps. This will make the training slightly "
                           "slower.")
        form.addParallelSection(threads=4, mpi=0)

    def _createFilenameTemplates(self):
        """ Centralize how files are called """
        myDict = {
            'imgsFn': self._getExtraPath('input_particles.xmd'),
            'fnVol': self._getExtraPath('input_volume.vol'),
            'fnVolMask': self._getExtraPath('input_volume_mask.vol'),
            'fnOutDir': self._getExtraPath()
        }
        self._updateFilenamesDict(myDict)

    # --------------------------- INSERT steps functions --------------------------------------------
    def _insertAllSteps(self):
        self._createFilenameTemplates()
        self._insertFunctionStep(self.writeMetaDataStep)
        self._insertFunctionStep(self.convertMetaDataStep)
        self._insertFunctionStep(self.predictStep)
        self._insertFunctionStep(self.createOutputStep)

    # --------------------------- STEPS functions ---------------------------------------------------
    def writeMetaDataStep(self):
        imgsFn = self._getFileName('imgsFn')
        fnVol = self._getFileName('fnVol')
        fnVolMask = self._getFileName('fnVolMask')

        inputParticles = self.inputParticles.get()
        zernikeParticles = self.zernikeParticles.get()
        Xdim = inputParticles.getXDim()
        self.newXdim = self.boxSize.get()

        ih = ImageHandler()
        inputVolume = zernikeParticles.refMap.get()
        ih.convert(getXmippFileName(inputVolume), fnVol)
        if Xdim != self.newXdim:
            self.runJob("xmipp_image_resize",
                        "-i %s --dim %d " % (fnVol, self.newXdim), numberOfMpi=1, env=xmipp3.Plugin.getEnviron())

        inputMask = zernikeParticles.refMask.get()
        if inputMask:
            ih.convert(getXmippFileName(inputMask), fnVolMask)
            if Xdim != self.newXdim:
                self.runJob("xmipp_image_resize",
                            "-i %s --dim %d --interp nearest" % (fnVolMask, self.newXdim), numberOfMpi=1,
                            env=xmipp3.Plugin.getEnviron())

        writeSetOfParticles(inputParticles, imgsFn)

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

    def convertMetaDataStep(self):
        md_file = self._getFileName('imgsFn')
        out_path = self._getExtraPath('h5_metadata')
        if not os.path.isdir(out_path):
            os.mkdir(out_path)
        sr = self.inputParticles.get().getSamplingRate()
        unStack = self.unStack.get()
        volume = self._getFileName('fnVol')
        mask = self._getFileName('fnVolMask')
        thr = self.numberOfThreads.get()
        args = "--md_file %s --out_path %s --sr %f --volume %s --mask %s --thr %d" \
               % (md_file, out_path, sr, volume, mask, thr)
        if unStack:
            args += " --unStack"
        program = os.path.join(const.XMIPP_SCRIPTS, "md_to_h5.py")
        program = flexutils.Plugin.getProgram(program)
        self.runJob(program, args, numberOfMpi=1)

    def predictStep(self):
        zernikeParticles = self.zernikeParticles.get()
        h5_file = self._getExtraPath(os.path.join('h5_metadata', 'metadata.h5'))
        weigths_file = zernikeParticles.modelPath.get()
        L1 = zernikeParticles.L1.get()
        L2 = zernikeParticles.L2.get()
        args = "--h5_file %s --weigths_file %s --L1 %d --L2 %d" \
               % (h5_file, weigths_file, L1, L2)
        if self.useGpu.get():
            gpu_list = ','.join([str(elem) for elem in self.getGpuList()])
            args += " --gpu %s" % gpu_list
        program = os.path.join(const.TENSORFLOW_SCRIPTS, "predict_zernike3deep.py")
        program = flexutils.Plugin.getTensorflowProgram(program)
        self.runJob(program, args, numberOfMpi=1)

    def createOutputStep(self):
        inputParticles = self.inputParticles.get()
        zernikeParticles = self.zernikeParticles.get()
        Xdim = inputParticles.getXDim()
        self.newXdim = self.boxSize.get()
        model_path = zernikeParticles.modelPath.get()
        h5_file = self._getExtraPath(os.path.join('h5_metadata', 'metadata.h5'))
        with h5py.File(h5_file, 'r') as hf:
            zernike_space = np.asarray(hf.get('zernike_space'))

        inputSet = self.inputParticles.get()
        partSet = self._createSetOfParticles()
        inputMask = zernikeParticles.refMask.get()
        inputVolume = zernikeParticles.refMap.get()

        partSet.copyInfo(inputSet)
        partSet.setAlignmentProj()

        correctionFactor = Xdim / self.newXdim
        zernike_space = correctionFactor * zernike_space
        for idx, particle in enumerate(inputSet.iterItems()):
            # z = correctionFactor * zernike_space[idx]

            csv_z_space = CsvList()
            for c in zernike_space[idx]:
                csv_z_space.append(c)

            particle._xmipp_sphCoefficients = csv_z_space

            partSet.append(particle)

        partSet.L1 = Integer(zernikeParticles.L1.get())
        partSet.L2 = Integer(zernikeParticles.L2.get())
        partSet.Rmax = Float(Xdim / 2)
        partSet.refMask = String(inputMask)
        partSet.refMap = String(inputVolume)
        partSet.modelPath = String(model_path)

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
