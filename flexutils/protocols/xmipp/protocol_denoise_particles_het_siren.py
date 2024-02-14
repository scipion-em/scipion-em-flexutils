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
import numpy as np

from xmipp_metadata.metadata import XmippMetaData
from xmipp_metadata.image_handler import ImageHandler

import pyworkflow.protocol.params as params
from pyworkflow.object import String, Integer
from pyworkflow.utils.path import moveFile
from pyworkflow import VERSION_2_0
import pyworkflow.utils as pwutils

from pwem.protocols import ProtAnalysis3D, ProtFlexBase
import pwem.emlib.metadata as md
from pwem.constants import ALIGN_PROJ
from pwem.objects import ParticleFlex

from xmipp3.convert import createItemMatrix, setXmippAttributes, writeSetOfParticles, \
    geometryFromMatrix, matrixFromGeometry
import xmipp3

import flexutils
import flexutils.constants as const
from flexutils.utils import getXmippFileName


class TensorflowProtDenoiseParticlesHetSiren(ProtAnalysis3D, ProtFlexBase):
    """ Denoise particles with the HetSIREN network. """
    _label = 'denoise particles - HetSIREN'
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
        group.addParam('hetSirenProtocol', params.PointerParam, label="HetSIREN trained network",
                       pointerClass='TensorflowProtAngularAlignmentHetSiren',
                       help="Previously executed 'angular align - HetSIREN'. "
                            "This will allow to load the network trained in that protocol to be used during "
                            "the prediction")
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
        hetSirenProtocol = self.hetSirenProtocol.get()
        Xdim = inputParticles.getXDim()
        self.newXdim = hetSirenProtocol.boxSize.get()
        self.vol_mask_dim = hetSirenProtocol.outSize.get() if hetSirenProtocol.outSize.get() is not None else self.newXdim

        if hetSirenProtocol.inputVolume.get():  # Map reference
            ih = ImageHandler()
            inputVolume = hetSirenProtocol.inputVolume.get().getFileName()
            ih.convert(getXmippFileName(inputVolume), fnVol)
            curr_vol_dim = ImageHandler(getXmippFileName(inputVolume)).getDimensions()[-1]
            if curr_vol_dim != self.vol_mask_dim:
                self.runJob("xmipp_image_resize",
                            "-i %s --dim %d " % (fnVol, self.vol_mask_dim), numberOfMpi=1, env=xmipp3.Plugin.getEnviron())

        if hetSirenProtocol.inputVolumeMask.get():  # Mask reference
            ih = ImageHandler()
            inputMask = hetSirenProtocol.inputVolumeMask.get().getFileName()
            if inputMask:
                ih.convert(getXmippFileName(inputMask), fnVolMask)
                curr_mask_dim = ImageHandler(getXmippFileName(inputMask)).getDimensions()[-1]
                if curr_mask_dim != self.vol_mask_dim:
                    self.runJob("xmipp_image_resize",
                                "-i %s --dim %d --interp nearest" % (fnVolMask, self.vol_mask_dim), numberOfMpi=1,
                                env=xmipp3.Plugin.getEnviron())
        else:
            ImageHandler().createCircularMask(fnVolMask, boxSize=self.vol_mask_dim, is3D=True)

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
        hetSirenProtocol = self.hetSirenProtocol.get()
        md_file = self._getFileName('imgsFn')
        weigths_file = hetSirenProtocol._getExtraPath(os.path.join('network', 'het_siren_model.h5'))
        pad = hetSirenProtocol.pad.get()
        self.newXdim = hetSirenProtocol.boxSize.get()
        Xdim = self.inputParticles.get().getXDim()
        correctionFactor = Xdim / self.newXdim
        sr = correctionFactor * self.inputParticles.get().getSamplingRate()
        applyCTF = hetSirenProtocol.ctfType.get()
        hetDim = hetSirenProtocol.hetDim.get()
        trainSize = hetSirenProtocol.trainSize.get() if hetSirenProtocol.trainSize.get() else self.newXdim
        outSize = hetSirenProtocol.outSize.get() if hetSirenProtocol.outSize.get() else self.newXdim
        args = "--md_file %s --weigths_file %s --pad %d " \
               "--sr %f --apply_ctf %d --het_dim %d --trainSize %d --outSize %d" \
               % (md_file, weigths_file, pad, sr, applyCTF, hetDim, trainSize, outSize)

        if hetSirenProtocol.ctfType.get() == 0:
            args += " --ctf_type apply"
        elif hetSirenProtocol.ctfType.get() == 1:
            args += " --ctf_type wiener"

        if hetSirenProtocol.architecture.get() == 0:
            args += " --architecture deepconv"
        elif hetSirenProtocol.architecture.get() == 1:
            args += " --architecture convnn"
        elif hetSirenProtocol.architecture.get() == 2:
            args += " --architecture mlpnn"

        if hetSirenProtocol.refinePose.get():
            args += " --refine_pose"

        if self.useGpu.get():
            gpu_list = ','.join([str(elem) for elem in self.getGpuList()])
            args += " --gpu %s" % gpu_list

        program = flexutils.Plugin.getTensorflowProgram("predict_particles_het_siren.py", python=False)
        self.runJob(program, args, numberOfMpi=1)

        # Scale particles
        ImageHandler().scaleSplines(self._getExtraPath('decoded_particles.mrcs'),
                                    self._getExtraPath('decoded_particles.mrcs'),
                                    finalDimension=Xdim, overwrite=True, isStack=True)

    def createOutputStep(self):
        inputParticles = self.inputParticles.get()
        hetSirenProtocol = self.hetSirenProtocol.get()
        Xdim = inputParticles.getXDim()
        self.newXdim = hetSirenProtocol.boxSize.get()
        trainSize = hetSirenProtocol.trainSize.get()
        outSize = hetSirenProtocol.outSize.get()
        model_path = hetSirenProtocol._getExtraPath(os.path.join('network', 'het_siren_model.h5'))
        md_file = self._getFileName('imgsFn')

        metadata = XmippMetaData(md_file)
        latent_space = np.asarray([np.fromstring(item, sep=',') for item in metadata[:, 'latent_space']])
        delta_rot = metadata[:, 'delta_angle_rot']
        delta_tilt = metadata[:, 'delta_angle_tilt']
        delta_psi = metadata[:, 'delta_angle_psi']
        delta_shift_x = metadata[:, 'delta_shift_x']
        delta_shift_y = metadata[:, 'delta_shift_y']
        denoised_images_path = metadata[:, "image"]

        inputSet = self.inputParticles.get()
        partSet = self._createSetOfParticlesFlex(progName=const.HETSIREN)

        partSet.copyInfo(inputSet)
        partSet.setAlignmentProj()

        correctionFactor = Xdim / self.newXdim

        inverseTransform = partSet.getAlignment() == ALIGN_PROJ

        idx = 0
        for particle in inputSet.iterItems():
            outParticle = ParticleFlex(progName=const.HETSIREN)
            outParticle.copyInfo(particle)

            index, _ = denoised_images_path[idx].split("@")
            outParticle.setLocation((int(index), self._getExtraPath("decoded_particles.mrcs")))

            outParticle.setZFlex(latent_space[idx])

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
            outParticle.getTransform().setMatrix(tr)

            partSet.append(outParticle)

            idx += 1

        partSet.getFlexInfo().modelPath = String(model_path)
        partSet.getFlexInfo().coordStep = Integer(hetSirenProtocol.step.get())
        partSet.getFlexInfo().outSize = Integer(outSize)
        partSet.getFlexInfo().trainSize = Integer(trainSize)

        if hetSirenProtocol.inputVolume.get():
            inputMask = hetSirenProtocol.inputVolumeMask.get().getFileName()
            inputVolume = hetSirenProtocol.inputVolume.get().getFileName()
            partSet.getFlexInfo().refMask = String(inputMask)
            partSet.getFlexInfo().refMap = String(inputVolume)

        if hetSirenProtocol.architecture.get() == 0:
            partSet.getFlexInfo().architecture = String("deepconv")
        elif hetSirenProtocol.architecture.get() == 1:
            partSet.getFlexInfo().architecture = String("convnn")
        elif hetSirenProtocol.architecture.get() == 2:
            partSet.getFlexInfo().architecture = String("mlpnn")

        if hetSirenProtocol.ctfType.get() == 0:
            partSet.getFlexInfo().ctfType = String("apply")
        elif hetSirenProtocol.ctfType.get() == 1:
            partSet.getFlexInfo().ctfType = String("wiener")

        partSet.getFlexInfo().pad = Integer(hetSirenProtocol.pad.get())

        self._defineOutputs(outputParticles=partSet)
        self._defineTransformRelation(self.inputParticles, partSet)

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
