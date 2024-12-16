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

from xmipp_metadata.metadata import XmippMetaData
from xmipp_metadata.image_handler import ImageHandler

import pyworkflow.protocol.params as params
from pyworkflow.object import String, Boolean
from pyworkflow.utils.path import moveFile
from pyworkflow import VERSION_2_0

from pwem.protocols import ProtAnalysis3D
import pwem.emlib.metadata as md
from pwem.constants import ALIGN_PROJ
from pwem.objects import Volume

from xmipp3.convert import createItemMatrix, setXmippAttributes, writeSetOfParticles, \
    geometryFromMatrix, matrixFromGeometry
import xmipp3

import flexutils
from flexutils.utils import getXmippFileName


class TensorflowProtPredictHomoSiren(ProtAnalysis3D):
    """ Predict particle poses with map decoding with the HomoSIREN network. """
    _label = 'predict - HomoSIREN'
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
        group.addParam('homoSirenProtocol', params.PointerParam, label="HomoSIREN trained network",
                       pointerClass='TensorflowProtAngularAlignmentHomoSiren',
                       help="Previously executed 'angular align - HomoSIREN'. "
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
        homoSirenProtocol = self.homoSirenProtocol.get()
        Xdim = inputParticles.getXDim()
        self.newXdim = homoSirenProtocol.boxSize.get()

        if homoSirenProtocol.inputVolume.get():  # Map reference
            ih = ImageHandler()
            inputVolume = homoSirenProtocol.inputVolume.get().getFileName()
            ih.convert(getXmippFileName(inputVolume), fnVol)
            curr_vol_dim = ImageHandler(getXmippFileName(inputVolume)).getDimensions()[-1]
            if curr_vol_dim != self.newXdim:
                self.runJob("xmipp_image_resize",
                            "-i %s --dim %d " % (fnVol, self.newXdim), numberOfMpi=1, env=xmipp3.Plugin.getEnviron())

        if homoSirenProtocol.inputVolumeMask.get():  # Mask reference
            ih = ImageHandler()
            inputMask = homoSirenProtocol.inputVolumeMask.get().getFileName()
            if inputMask:
                ih.convert(getXmippFileName(inputMask), fnVolMask)
                curr_mask_dim = ImageHandler(getXmippFileName(inputMask)).getDimensions()[-1]
                if curr_mask_dim != self.newXdim:
                    self.runJob("xmipp_image_resize",
                                "-i %s --dim %d --interp nearest" % (fnVolMask, self.newXdim), numberOfMpi=1,
                                env=xmipp3.Plugin.getEnviron())
        else:
            ImageHandler().createCircularMask(fnVolMask, boxSize=self.newXdim, is3D=True)

        writeSetOfParticles(inputParticles, imgsFn)

        if self.newXdim != Xdim:
            params = "-i %s -o %s --save_metadata_stack %s --fourier %d" % \
                     (imgsFn,
                      self._getTmpPath('scaled_particles.stk'),
                      self._getExtraPath('scaled_particles.xmd'),
                      self.newXdim)
            if self.numberOfMpi.get() > 1:
                params += " --mpi_job_size %d" % int(inputParticles.getSize() / self.numberOfMpi.get())
            self.runJob("xmipp_image_resize", params, numberOfMpi=self.numberOfMpi.get(),
                        env=xmipp3.Plugin.getEnviron())
            moveFile(self._getExtraPath('scaled_particles.xmd'), imgsFn)

    def predictStep(self):
        homoSirenProtocol = self.homoSirenProtocol.get()
        md_file = self._getFileName('imgsFn')
        weigths_file = homoSirenProtocol._getExtraPath(os.path.join('network', 'homo_siren_model.h5'))
        pad = homoSirenProtocol.pad.get()
        self.newXdim = homoSirenProtocol.boxSize.get()
        correctionFactor = self.inputParticles.get().getXDim() / self.newXdim
        sr = correctionFactor * self.inputParticles.get().getSamplingRate()
        applyCTF = homoSirenProtocol.applyCTF.get()
        args = "--md_file %s --weigths_file %s --pad %d " \
               "--sr %f --apply_ctf %d" \
               % (md_file, weigths_file, pad, sr, applyCTF)

        if homoSirenProtocol.ctfType.get() == 0:
            args += " --ctf_type apply"
        elif homoSirenProtocol.ctfType.get() == 1:
            args += " --ctf_type wiener"

        if homoSirenProtocol.architecture.get() == 0:
            args += " --architecture convnn"
        elif homoSirenProtocol.architecture.get() == 1:
            args += " --architecture mlpnn"

        if self.filterDecoded.get():
            args += " --apply_filter"

        if self.useGpu.get():
            gpu_list = ','.join([str(elem) for elem in self.getGpuList()])
            args += " --gpu %s" % gpu_list

        program = flexutils.Plugin.getTensorflowProgram("predict_homo_siren.py", python=False)
        self.runJob(program, args, numberOfMpi=1)

    def createOutputStep(self):
        inputParticles = self.inputParticles.get()
        homoSirenProtocol = self.homoSirenProtocol.get()
        Xdim = inputParticles.getXDim()
        self.newXdim = homoSirenProtocol.boxSize.get()
        model_path = homoSirenProtocol._getExtraPath(os.path.join('network', 'homo_siren_model.h5'))
        md_file = self._getFileName('imgsFn')

        metadata = XmippMetaData(md_file)
        delta_rot = metadata[:, 'delta_angle_rot']
        delta_tilt = metadata[:, 'delta_angle_tilt']
        delta_psi = metadata[:, 'delta_angle_psi']
        delta_shift_x = metadata[:, 'delta_shift_x']
        delta_shift_y = metadata[:, 'delta_shift_y']

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
            angles[0] += delta_rot[idx]
            angles[1] += delta_tilt[idx]
            angles[2] += delta_psi[idx]

            # Apply delta shifts
            shifts[0] += correctionFactor * delta_shift_x[idx]
            shifts[1] += correctionFactor * delta_shift_y[idx]

            # if refinePose:
            #     # Apply delta angles
            #     angles[0] += delta_rot[idx]
            #     angles[1] += delta_tilt[idx]
            #     angles[2] += delta_psi[idx]
            #
            #     # Apply delta shifts
            #     shifts[0] += correctionFactor * delta_shift_x[idx]
            #     shifts[1] += correctionFactor * delta_shift_y[idx]
            # else:
            #     # Assign delta angles
            #     angles[0] = delta_rot[idx]
            #     angles[1] = delta_tilt[idx]
            #     angles[2] = delta_psi[idx]
            #
            #     # Assign delta shifts
            #     shifts[0] = correctionFactor * delta_shift_x[idx]
            #     shifts[1] = correctionFactor * delta_shift_y[idx]

            # Set new transformation matrix
            tr = matrixFromGeometry(shifts, angles, inverseTransform)
            particle.getTransform().setMatrix(tr)

            partSet.append(particle)

            idx += 1

        partSet.modelPath = String(model_path)

        if homoSirenProtocol.inputVolume.get():
            inputMask = homoSirenProtocol.inputVolumeMask.get().getFileName()
            inputVolume = homoSirenProtocol.inputVolume.get().getFileName()
            partSet.refMask = String(inputMask)
            partSet.refMap = String(inputVolume)

        outVol = Volume()
        outVol.setSamplingRate(inputParticles.getSamplingRate())
        outVol.setLocation(self._getExtraPath('decoded_map.mrc'))

        ImageHandler().scaleSplines(self._getExtraPath('decoded_map.mrc'),
                                    self._getExtraPath('decoded_map.mrc'),
                                    finalDimension=inputParticles.getXDim(), overwrite=True)

        self._defineOutputs(outputParticles=partSet)
        self._defineTransformRelation(self.inputParticles, partSet)

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
