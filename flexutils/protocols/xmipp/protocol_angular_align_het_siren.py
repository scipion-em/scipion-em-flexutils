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

import pyworkflow.protocol.params as params
from pyworkflow.object import String, Integer, CsvList
from pyworkflow.utils.path import moveFile
from pyworkflow import VERSION_2_0

from pwem.protocols import ProtAnalysis3D
import pwem.emlib.metadata as md
from pwem.emlib.image import ImageHandler
from pwem.constants import ALIGN_PROJ
from pwem.objects import Volume

from xmipp3.convert import createItemMatrix, setXmippAttributes, writeSetOfParticles, \
    geometryFromMatrix, matrixFromGeometry
import xmipp3

import flexutils
from flexutils.protocols import ProtFlexBase
from flexutils.objects import ParticleFlex
import flexutils.constants as const
from flexutils.utils import getXmippFileName


class TensorflowProtAngularAlignmentHetSiren(ProtAnalysis3D, ProtFlexBase):
    """ Protocol for angular alignment with heterogeneous reconstruction with the HetSIREN algorithm."""
    _label = 'angular align - HetSIREN'
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
        group.addParam('inputParticles', params.PointerParam, label="Input particles", pointerClass='SetOfParticles')
        group.addParam('inputVolume', params.PointerParam, allowsNull=True,
                       label="Starting volume", pointerClass='Volume',
                       help="If provided, the HomoSIREN network will learn to refine with the new learned angles. "
                            "Otherwise, the network will learn the reconstruction of the map from scratch")
        group.addParam('inputVolumeMask', params.PointerParam,
                       label="Reconsctruction mask", pointerClass='VolumeMask', allowsNull=True,
                       help="If provided, the pose refinement and reconstruction learned by HomoSIREN will be focused "
                            "in the region delimited by the mask. Otherise, a sphere inscribed in the volume box will "
                            "be used")
        group.addParam('boxSize', params.IntParam, default=128,
                       label='Downsample particles to this box size', expertLevel=params.LEVEL_ADVANCED,
                       help='In general, downsampling the particles will increase performance without compromising '
                            'the estimation the deformation field for each particle. Note that output particles will '
                            'have the original box size, and Zernike3D coefficients will be modified to work with the '
                            'original size images')
        group = form.addGroup("Latent Space")
        group.addParam('hetDim', params.IntParam, default=10, label='Latent space dimension',
                       expertLevel=params.LEVEL_ADVANCED,
                       help="Dimension of the HetSIREN bottleneck (latent space dimension)")
        group = form.addGroup("CTF Parameters (Advanced)",
                              expertLevel=params.LEVEL_ADVANCED)
        group.addParam('applyCTF', params.BooleanParam, default=True, label='Apply CTF?',
                       expertLevel=params.LEVEL_ADVANCED,
                       help="If true, volume projection will be subjected to CTF corrections")
        group.addParam('ctfType', params.EnumParam, choices=['Apply', 'Wiener'],
                       default=0, label="CTF correction type",
                       display=params.EnumParam.DISPLAY_HLIST,
                       condition="applyCTF", expertLevel=params.LEVEL_ADVANCED,
                       help="* *Apply*: CTF is applied to the projection generated from the reference map\n"
                            "* *Wiener*: input particle is CTF corrected by a Wiener fiter")
        group.addParam("pad", params.IntParam, default=2,
                       label="Padding factor",
                       condition="applyCTF", expertLevel=params.LEVEL_ADVANCED,
                       help="Determines the padding factor to be applied before computing "
                            "the Fourier Transform of the images to increase the frequency "
                            "content")
        form.addSection(label='Network')
        form.addParam('architecture', params.EnumParam, choices=['ConvNN', 'MPLNN'],
                      expertLevel=params.LEVEL_ADVANCED,
                      default=0, label="Network architecture", display=params.EnumParam.DISPLAY_HLIST,
                      help="* *ConvNN*: convolutional neural network\n"
                           "* *MLPNN*: multiperceptron neural network")
        form.addParam('fineTune', params.BooleanParam, default=False, label="Fine tune previous network?",
                      help="If True, a previously trained deepPose network will be fine tuned based on the "
                           "new input parameters. Note that when this option is set, the input particles "
                           "must have a trained deepPose network associated (i.e. particles must come from "
                           "a **'angular align - deepPose'** protocol.")
        form.addParam('netProtocol', params.PointerParam, label="Previously trained network",
                      allowsNull=True,
                      pointerClass='TensorflowProtAngularAlignmentHomoSiren',
                      condition="fineTune")
        form.addParam('refinePose', params.BooleanParam, default=True, label="Refine pose?",
                      help="If True, the neural network will be also trained to refine the angular "
                           "and shift assignation of the particles to make it more consistent with the "
                           "heterogeneity estimation. Otherwise, only heterogeneity information will be "
                           "estimated.")
        form.addParam('epochs', params.IntParam, default=50, label='Number of training epochs',
                      help="When training in refinenment mode, the number of epochs might be decreased to "
                           "improve performance. For ab initio, we recommend around 25 - 50 epochs to reach "
                           "a meaningful local minima.")
        form.addParam('batch_size', params.IntParam, default=16, label='Number of images in batch',
                      help="Number of images that will be used simultaneously for every training step. "
                           "We do not recommend to change this value unless you experience memory errors. "
                           "In this case, value should be decreased.")
        form.addParam('split_train', params.FloatParam, default=1.0, label='Traning dataset fraction',
                      help="This value (between 0 and 1) determines the fraction of images that will "
                           "be used to train the network.")
        form.addParam('step', params.IntParam, default=1, label='Points step',
                      help="How many points (voxels) to skip during the training computations. "
                           "A value of 1 means that all point within the mask provided in the input "
                           "will be used. A value of 2 implies that half of the point will be skipped "
                           "to increase the performance.")
        form.addSection(label='Cost function')
        form.addParam('costFunction', params.EnumParam, choices=['MSE', 'Correlation', 'FPC'],
                      default=0, label="Cost function type", display=params.EnumParam.DISPLAY_HLIST,
                      help="Determine the cost function to be minimized during the neural network training. Both, "
                           "**Correlation** and **FPC** will yield similar results. However, **Corr * FPC** "
                           "allows excluding high frequency information by masking in the Fourier space. This might "
                           "help preveting overfitting in scenarios with low Signal to Noise ratios at the expense "
                           "of slightly increasing computation time.\n"
                           "Recommended cost function is **MSE** unlsess a reference map has been specified. "
                           "If a volume is going to be refined, **Correlation** or **FPC** will provide more accurate "
                           "results")
        form.addParam('maskRadius', params.FloatParam, default=0.85, label="Mask radius (%)",
                      condition="costFunction==1",
                      help="Determine the radius (in percentage) of the circular mask to be applied to the Fourier "
                           "Transform of the images. A value of 1 implies that the circular mask is inscribed to the "
                           "bounding box the Fourier Transform.")
        form.addParam("smoothMask", params.BooleanParam, default=True, label="Smooth mask?",
                      condition="costFunction==1",
                      help="If True, the mask applied to the Fourier Transform of the particle images will have a smooth"
                           "vanishing transition.")
        form.addParam("l1Reg", params.FloatParam, default=0.5, label="L1 loss regularization",
                      help="Determines the weight of the L1 map minimization in the cost function. L1 is moslty used to "
                           "decrease the amount of noise in the map learned by the network. We do not recommend to touch "
                           "this parameter")
        form.addSection(label='Output')
        form.addParam("filterDecoded", params.BooleanParam, default=False, label="Filter decoded map?",
                      help="If True, the maps decoded after training the network will be convoluted with a Gaussian filter. "
                           "In general, this postprocessing is not needed unless 'Points step' parameter is set to a value "
                           "greater than 1")
        form.addParam("numVol", params.IntParam, default=20, label="Number of decoded maps",
                      help="Determines in how many regions the trained latent space will be splitted by "
                           "KMeans, allowing to decode a state based on the representative of each cluster. "
                           "This provides and initial summary/exploration of the trained landscape")
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
        self._insertFunctionStep(self.trainingStep)
        self._insertFunctionStep(self.predictStep)
        self._insertFunctionStep(self.createOutputStep)

    # --------------------------- STEPS functions -----------------------
    def writeMetaDataStep(self):
        imgsFn = self._getFileName('imgsFn')
        fnVol = self._getFileName('fnVol')
        fnVolMask = self._getFileName('fnVolMask')

        inputParticles = self.inputParticles.get()
        Xdim = inputParticles.getXDim()
        self.newXdim = self.boxSize.get()

        if self.inputVolume.get():  # Map reference
            ih = ImageHandler()
            inputVolume = self.inputVolume.get().getFileName()
            ih.convert(getXmippFileName(inputVolume), fnVol)
            if Xdim != self.newXdim:
                self.runJob("xmipp_image_resize",
                            "-i %s --dim %d " % (fnVol, self.newXdim), numberOfMpi=1, env=xmipp3.Plugin.getEnviron())

        if self.inputVolumeMask.get():  # Mask reference
            ih = ImageHandler()
            inputMask = self.inputVolumeMask.get().getFileName()
            if inputMask:
                ih.convert(getXmippFileName(inputMask), fnVolMask)
                if Xdim != self.newXdim:
                    self.runJob("xmipp_image_resize",
                                "-i %s --dim %d --interp nearest" % (fnVolMask, self.newXdim), numberOfMpi=1,
                                env=xmipp3.Plugin.getEnviron())
        else:
            ih = ImageHandler()
            ih.createEmptyImage(fnVolMask, xDim=self.newXdim, yDim=self.newXdim,
                                zDim=self.newXdim, nDim=1)
            ih.createCircularMask(int(0.5 * self.newXdim), fnVolMask, fnVolMask)

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

    def trainingStep(self):
        md_file = self._getFileName('imgsFn')
        out_path = self._getExtraPath('network')
        if not os.path.isdir(out_path):
            os.mkdir(out_path)
        pad = self.pad.get()
        batch_size = self.batch_size.get()
        step = self.step.get()
        split_train = self.split_train.get()
        epochs = self.epochs.get()
        l1Reg = self.l1Reg.get()
        hetDim = self.hetDim.get()
        correctionFactor = self.inputParticles.get().getXDim() / self.newXdim
        sr = correctionFactor * self.inputParticles.get().getSamplingRate()
        applyCTF = self.applyCTF.get()
        args = "--md_file %s --out_path %s --batch_size %d " \
               "--shuffle --split_train %f --epochs %d --pad %d --refine_pose " \
               "--sr %f --apply_ctf %d --step %d --l1_reg %f --het_dim %d" \
               % (md_file, out_path, batch_size, split_train, epochs, pad, sr, applyCTF, step,
                  l1Reg, hetDim)

        if self.costFunction.get() == 0:
            args += " --cost mse"
        elif self.costFunction.get() == 1:
            args += " --cost corr"
        elif self.costFunction.get() == 2:
            args += " --cost corr-fpc --radius_mask %f" % self.maskRadius.get()
            if self.smoothMask.get():
                args += " --smooth_mask"

        if self.architecture.get() == 0:
            args += " --architecture convnn"
        elif self.architecture.get() == 1:
            args += " --architecture mlpnn"

        if self.ctfType.get() == 0:
            args += " --ctf_type apply"
        elif self.ctfType.get() == 1:
            args += " --ctf_type wiener"

        if self.refinePose.get():
            args += " --refine_pose"

        if self.fineTune.get():
            netProtocol = self.netProtocol.get()
            modelPath = netProtocol._getExtraPath(os.path.join('network', 'het_siren_model'))
            args += " --weigths_file %s" % modelPath

        if self.useGpu.get():
            gpu_list = ','.join([str(elem) for elem in self.getGpuList()])
            args += " --gpu %s" % gpu_list

        program = flexutils.Plugin.getTensorflowProgram("train_het_siren.py", python=False)
        self.runJob(program, args, numberOfMpi=1)

    def predictStep(self):
        md_file = self._getFileName('imgsFn')
        weigths_file = self._getExtraPath(os.path.join('network', 'het_siren_model'))
        pad = self.pad.get()
        hetDim = self.hetDim.get()
        numVol = self.numVol.get()
        correctionFactor = self.inputParticles.get().getXDim() / self.newXdim
        sr = correctionFactor * self.inputParticles.get().getSamplingRate()
        applyCTF = self.applyCTF.get()
        args = "--md_file %s --weigths_file %s --pad %d --refine_pose --sr %f " \
               "--apply_ctf %d --het_dim %d --num_vol %d" \
               % (md_file, weigths_file, pad, sr, applyCTF, hetDim, numVol)

        if self.ctfType.get() == 0:
            args += " --ctf_type apply"
        elif self.ctfType.get() == 1:
            args += " --ctf_type wiener"

        if self.architecture.get() == 0:
            args += " --architecture convnn"
        elif self.architecture.get() == 1:
            args += " --architecture mlpnn"

        if self.refinePose.get():
            args += " --refine_pose"

        if self.filterDecoded.get():
            args += " --apply_filter"

        if self.useGpu.get():
            gpu_list = ','.join([str(elem) for elem in self.getGpuList()])
            args += " --gpu %s" % gpu_list

        program = flexutils.Plugin.getTensorflowProgram("predict_het_siren.py", python=False)
        self.runJob(program, args, numberOfMpi=1)

    def createOutputStep(self):
        inputParticles = self.inputParticles.get()
        Xdim = inputParticles.getXDim()
        self.newXdim = self.boxSize.get()
        model_path = self._getExtraPath(os.path.join('network', 'het_siren_model'))
        md_file = self._getFileName('imgsFn')

        metadata = XmippMetaData(md_file)
        latent_space = np.asarray([np.fromstring(item, sep=',') for item in metadata[:, 'latent_space']])
        delta_rot = metadata[:, 'delta_angle_rot']
        delta_tilt = metadata[:, 'delta_angle_tilt']
        delta_psi = metadata[:, 'delta_angle_psi']
        delta_shift_x = metadata[:, 'delta_shift_x']
        delta_shift_y = metadata[:, 'delta_shift_y']

        inputSet = self.inputParticles.get()
        partSet = self._createSetOfParticlesFlex(progName=const.HETSIREN)

        partSet.copyInfo(inputSet)
        partSet.setAlignmentProj()

        correctionFactor = Xdim / self.newXdim

        inverseTransform = partSet.getAlignment() == ALIGN_PROJ

        for idx, particle in enumerate(inputSet.iterItems()):

            outParticle = ParticleFlex(progName=const.HETSIREN)
            outParticle.copyInfo(particle)

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

            # Set new transformation matrix
            tr = matrixFromGeometry(shifts, angles, inverseTransform)
            outParticle.getTransform().setMatrix(tr)

            partSet.append(outParticle)

        partSet.getFlexInfo().modelPath = String(model_path)
        partSet.getFlexInfo().coordStep = Integer(self.step.get())

        if self.inputVolume.get():
            inputMask = self.inputVolumeMask.get().getFileName()
            inputVolume = self.inputVolume.get().getFileName()
            partSet.getFlexInfo().refMask = String(inputMask)
            partSet.getFlexInfo().refMap = String(inputVolume)

        if self.architecture.get() == 0:
            partSet.getFlexInfo().architecture = String("convnn")
        elif self.architecture.get() == 1:
            partSet.getFlexInfo().architecture = String("mlpnn")

        if self.ctfType.get() == 0:
            partSet.getFlexInfo().ctfType = String("apply")
        elif self.ctfType.get() == 1:
            partSet.getFlexInfo().ctfType = String("wiener")

        partSet.getFlexInfo().pad = Integer(self.pad.get())

        outVols = self._createSetOfVolumes()
        outVols.setSamplingRate(inputParticles.getSamplingRate() / correctionFactor)
        for idx in range(self.numVol.get()):
            outVol = Volume()
            outVol.setSamplingRate(inputParticles.getSamplingRate() / correctionFactor)
            outVol.setLocation(self._getExtraPath('decoded_map_class_%d.mrc' % (idx + 1)))
            outVols.append(outVol)

            ImageHandler().scaleSplines(self._getExtraPath('decoded_map_class_%d.mrc' % (idx + 1)),
                                        self._getExtraPath('decoded_map_class_%d.mrc' % (idx + 1)), 1,
                                        finalDimension=inputParticles.getXDim())

        self._defineOutputs(outputParticles=partSet)
        self._defineTransformRelation(self.inputParticles, partSet)

        self._defineOutputs(outputVolumes=outVols)
        self._defineTransformRelation(self.inputParticles, outVols)

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
        errors = []
        return errors