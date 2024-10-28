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


import os
import re
from glob import glob

from xmipp_metadata.metadata import XmippMetaData
from xmipp_metadata.image_handler import ImageHandler

import pyworkflow.protocol.params as params
from pyworkflow.object import String, Boolean
from pyworkflow.utils.path import moveFile
from pyworkflow import VERSION_2_0

from pwem.protocols import ProtAnalysis3D
import pwem.emlib.metadata as md
from pwem.constants import ALIGN_PROJ, ALIGN_NONE
from pwem.objects import Volume

from xmipp3.convert import createItemMatrix, setXmippAttributes, writeSetOfParticles, \
    geometryFromMatrix, matrixFromGeometry
import xmipp3

import flexutils
from flexutils.utils import getXmippFileName


class TensorflowProtPredictReconSiren(ProtAnalysis3D):
    """ Predict particle pose (and map) with RecoSIREN neural network. """
    _label = 'predict - ReconSIREN'
    _lastUpdateVersion = VERSION_2_0

    # --------------------------- DEFINE param functions -----------------------
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
        group = form.addGroup("Data")
        group.addParam('inputParticles', params.PointerParam, label="Input particles to predict",
                       pointerClass='SetOfParticles')
        group.addParam('reconSirenProtocol', params.PointerParam, label="ReconSIREN trained network",
                       pointerClass='TensorflowProtAngularAlignmentReconSiren',
                       help="Previously executed 'angular align - ReconSIREN'. "
                            "This will allow to load the network trained in that protocol to be used during "
                            "the prediction")
        form.addSection(label='Output')
        form.addParam("filterDecoded", params.BooleanParam, default=False, label="Filter decoded map?",
                      help="If True, the map decoded after training the network will be convoluted with a Gaussian filter. "
                           "In general, this postprocessing is not needed unless 'Points step' parameter is set to a value "
                           "greater than 1")
        form.addParallelSection(threads=4, mpi=0)

    def _createFilenameTemplates(self):
        """ Centralize how files are called """
        myDict = {
            'imgsFn': self._getExtraPath('input_particles.xmd'),
            'fnVol': self._getExtraPath('volume.mrc'),
            'fnVolMask': self._getExtraPath('mask.mrc'),
            'fnOutDir': self._getExtraPath()
        }
        self._updateFilenamesDict(myDict)

    # --------------------------- INSERT steps functions -----------------------
    def _insertAllSteps(self):
        self._createFilenameTemplates()
        self._insertFunctionStep(self.writeMetaDataStep)
        self._insertFunctionStep(self.predictStep)
        self._insertFunctionStep(self.createOutputStep)

    # --------------------------- STEPS functions -----------------------
    def writeMetaDataStep(self):
        imgsFn = self._getFileName('imgsFn')
        fnVol = self._getFileName('fnVol')
        fnVolMask = self._getFileName('fnVolMask')

        inputParticles = self.inputParticles.get()
        reconSirenProtocol = self.reconSirenProtocol.get()
        Xdim = inputParticles.getXDim()
        self.newXdim = reconSirenProtocol.boxSize.get()

        if reconSirenProtocol.inputVolume.get():  # Map reference
            ih = ImageHandler()
            inputVolume = reconSirenProtocol.inputVolume.get().getFileName()
            ih.convert(getXmippFileName(inputVolume), fnVol)
            curr_vol_dim = ImageHandler(getXmippFileName(inputVolume)).getDimensions()[-1]
            if curr_vol_dim != self.newXdim:
                self.runJob("xmipp_image_resize",
                            "-i %s --dim %d " % (fnVol, self.newXdim), numberOfMpi=1, env=xmipp3.Plugin.getEnviron())

        if reconSirenProtocol.inputVolumeMask.get():  # Mask reference
            ih = ImageHandler()
            inputMask = reconSirenProtocol.inputVolumeMask.get().getFileName()
            if inputMask:
                ih.convert(getXmippFileName(inputMask), fnVolMask)
                curr_mask_dim = ImageHandler(getXmippFileName(inputMask)).getDimensions()[-1]
                if curr_mask_dim != self.newXdim:
                    self.runJob("xmipp_image_resize",
                                "-i %s --dim %d --interp nearest" % (fnVolMask, self.newXdim), numberOfMpi=1,
                                env=xmipp3.Plugin.getEnviron())
        else:
            ImageHandler().createCircularMask(fnVolMask, boxSize=self.newXdim, is3D=True)

        writeSetOfParticles(inputParticles, imgsFn, alignType=ALIGN_NONE)

        if reconSirenProtocol.considerCTF.get():
            # Wiener filter
            sr = inputParticles.getSamplingRate()
            corrected_stk = self._getTmpPath('corrected_particles.mrcs')
            args = "-i %s -o %s --save_metadata_stack --keep_input_columns --sampling_rate %f --wc -1.0" \
                   % (imgsFn, corrected_stk, sr)
            program = 'xmipp_ctf_correct_wiener2d'
            self.runJob(program, args, numberOfMpi=self.numberOfThreads.get(), env=xmipp3.Plugin.getEnviron())

        if self.newXdim != Xdim:
            params = "-i %s -o %s --save_metadata_stack %s --keep_input_columns --fourier %d" % \
                     (self._getTmpPath('corrected_particles.xmd'),
                      self._getTmpPath('scaled_particles.mrcs'),
                      self._getExtraPath('scaled_particles.xmd'),
                      self.newXdim)
            if self.numberOfMpi.get() > 1:
                params += " --mpi_job_size %d" % int(inputParticles.getSize() / self.numberOfMpi.get())
            self.runJob("xmipp_image_resize", params, numberOfMpi=self.numberOfMpi.get(),
                        env=xmipp3.Plugin.getEnviron())
            moveFile(self._getExtraPath('scaled_particles.xmd'), imgsFn)

        # Removing Xmipp Phantom config file
        self.runJob('rm', self._getTmpPath('corrected_particles.mrcs'))

    def predictStep(self):
        reconSirenProtocol = self.reconSirenProtocol.get()
        md_file = self._getFileName('imgsFn')
        weigths_file = glob(reconSirenProtocol._getExtraPath(os.path.join('network', 'reconsiren_model*')))[0]
        # pad = reconSirenProtocol.pad.get()
        self.newXdim = reconSirenProtocol.boxSize.get()
        correctionFactor = self.inputParticles.get().getXDim() / self.newXdim
        sr = correctionFactor * self.inputParticles.get().getSamplingRate()
        # applyCTF = reconSirenProtocol.applyCTF.get()
        nCandidates = reconSirenProtocol.nCandidates.get()
        args = "--md_file %s --weigths_file %s --pad 2 " \
               "--sr %f --apply_ctf 0 --n_candidates %d" \
               % (md_file, weigths_file, sr, nCandidates)

        if reconSirenProtocol.inputVolume.get():
            args += " --only_pose"

        # if reconSirenProtocol.ctfType.get() == 0:
        #     args += " --ctf_type apply"
        # elif reconSirenProtocol.ctfType.get() == 1:
        #     args += " --ctf_type wiener"

        if reconSirenProtocol.architecture.get() == 0:
            args += " --architecture convnn"
        elif reconSirenProtocol.architecture.get() == 1:
            args += " --architecture mlpnn"

        if self.filterDecoded.get():
            args += " --apply_filter"

        if self.useGpu.get():
            gpu_list = ','.join([str(elem) for elem in self.getGpuList()])
            args += " --gpu %s" % gpu_list

        program = flexutils.Plugin.getTensorflowProgram("predict_reconsiren.py", python=False)
        self.runJob(program, args, numberOfMpi=1)

    def createOutputStep(self):
        inputParticles = self.inputParticles.get()
        reconSirenProtocol = self.reconSirenProtocol.get()
        Xdim = inputParticles.getXDim()
        self.newXdim = reconSirenProtocol.boxSize.get()
        model_path = glob(reconSirenProtocol._getExtraPath(os.path.join('network', 'reconsiren_model*')))[0]
        md_file = self._getFileName('imgsFn')

        metadata = XmippMetaData(md_file)
        rot = metadata[:, 'angleRot']
        tilt = metadata[:, 'angleTilt']
        psi = metadata[:, 'anglePsi']
        shift_x = metadata[:, 'shiftX']
        shift_y = metadata[:, 'shiftY']

        inputSet = self.inputParticles.get()
        partSet = self._createSetOfParticles()
        partSet.setHasCTF(inputSet.hasCTF())

        partSet.copyInfo(inputSet)
        partSet.setAlignmentProj()

        correctionFactor = Xdim / self.newXdim

        inverseTransform = partSet.getAlignment() == ALIGN_PROJ

        idx = 0
        for particle in inputSet.iterItems():

            tr_ori = particle.getTransform().getMatrix()
            shifts, angles = geometryFromMatrix(tr_ori, inverseTransform)

            # Apply delta angles
            angles[0] = rot[idx]
            angles[1] = tilt[idx]
            angles[2] = psi[idx]

            # Apply delta shifts
            shifts[0] = correctionFactor * shift_x[idx]
            shifts[1] = correctionFactor * shift_y[idx]

            # Set new transformation matrix
            tr = matrixFromGeometry(shifts, angles, inverseTransform)
            particle.getTransform().setMatrix(tr)

            partSet.append(particle)

            idx += 1

        partSet.modelPath = String(model_path)

        if reconSirenProtocol.inputVolume.get():
            inputMask = reconSirenProtocol.inputVolumeMask.get().getFileName()
            inputVolume = reconSirenProtocol.inputVolume.get().getFileName()
            partSet.refMask = String(inputMask)
            partSet.refMap = String(inputVolume)

        outVol = Volume()
        outVol.setSamplingRate(inputParticles.getSamplingRate())
        outVol.setLocation(self._getExtraPath('decoded_map.mrc'))

        self._defineOutputs(outputParticles=partSet)
        self._defineTransformRelation(self.inputParticles, partSet)

        if reconSirenProtocol.inputVolume.get() is None:
            ImageHandler().scaleSplines(self._getExtraPath('decoded_map.mrc'),
                                        self._getExtraPath('decoded_map.mrc'),
                                        finalDimension=inputParticles.getXDim(), overwrite=True)

            self._defineOutputs(outputVolume=outVol)
            self._defineTransformRelation(self.inputParticles, outVol)

    # --------------------------- UTILS functions -----------------------
    def _updateParticle(self, item, row):
        setXmippAttributes(item, row, md.MDL_ANGLE_ROT, md.MDL_ANGLE_TILT,
                           md.MDL_ANGLE_PSI, md.MDL_SHIFT_X, md.MDL_SHIFT_Y,
                           md.MDL_FLIP)
        createItemMatrix(item, row, align=ALIGN_PROJ)

    def getInputParticles(self):
        return self.inputParticles.get()

    def readPDB(self, fnIn):
        with open(fnIn) as f:
            lines = f.readlines()
        return lines

    def PDB2List(self, lines):
        newlines = []
        for line in lines:
            eval = re.search(r'^ATOM\s+\d+\s+/N|CA|C|O/\s+', line) if self.onlyBackbone.get() else line.startswith(
                "ATOM ")
            if eval:
                try:
                    x = float(line[30:38])
                    y = float(line[38:46])
                    z = float(line[46:54])
                    val = float(line[54:60])
                    newline = [x, y, z, val]
                    newlines.append(newline)
                except:
                    pass
        return newlines

    # ----------------------- VALIDATE functions -----------------------
    def validate(self):
        """ Try to find errors on define params. """
        return []
