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

import numpy as np
from joblib.testing import param
from xmipp_metadata.metadata import XmippMetaData
from xmipp_metadata.image_handler import ImageHandler

import pyworkflow.protocol.params as params
from pyworkflow.object import String, Boolean, Float
from pyworkflow.utils.path import moveFile
from pyworkflow import VERSION_2_0

from pwem.protocols import ProtAnalysis3D
import pwem.emlib.metadata as md
from pwem.constants import ALIGN_PROJ, ALIGN_NONE
from pwem.objects import Volume, SetOfAverages, Transform, Class3D

from xmipp3.convert import createItemMatrix, setXmippAttributes, writeSetOfParticles, \
    geometryFromMatrix, matrixFromGeometry
import xmipp3
import xmippLib

import flexutils
from flexutils.utils import getXmippFileName


class TensorflowProtAngularAlignmentReconSiren(ProtAnalysis3D):
    """ Ab initio reconstruction and global assignation with ReconSIREN neural network."""
    _label = 'angular align - ReconSIREN'
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
                       help="If provided, ReconSIREN network will switch to global assignation based on this map. "
                            "Otherwise, the network will perform ab initio reconstruction")
        group.addParam('inputVolumeMask', params.PointerParam,
                       label="Reconstruction mask", pointerClass='VolumeMask', allowsNull=True,
                       help="If provided, the pose refinement and reconstruction learned by ReconSIREN will be focused "
                            "in the region delimited by the mask. Otherwise, a sphere inscribed in the volume box will "
                            "be used")
        group.addParam('boxSize', params.IntParam, default=64,
                       label='Downsample particles to this box size', expertLevel=params.LEVEL_ADVANCED,
                       help='In general, downsampling the particles will increase performance without compromising '
                            'the estimation the deformation field for each particle. Note that output particles will '
                            'have the original box size, and Zernike3D coefficients will be modified to work with the '
                            'original size images')
        group.addParam("refinement", params.BooleanParam, default=False, label="Refine current alignment?",
                       condition="inputParticles and inputParticles.hasAlignment()")
        group.addParam("useHet", params.BooleanParam, default=False, label="Consider heterogeneity?")
        group = form.addGroup("Symmetry")
        group.addParam('symmetry', params.StringParam, default="c1", label='Symmetry group',
                       help="Determine the symmetry group to be considered when training the network")
        group = form.addGroup("CTF Parameters (Advanced)",
                              expertLevel=params.LEVEL_ADVANCED)
        group.addParam('considerCTF', params.BooleanParam, default=True, label='Consider CTF?',
                       expertLevel=params.LEVEL_ADVANCED,
                       help="Leave to Yes if the input particles have not undergone any CTF correction process. When using "
                            "2D classes, this parameter must be set to No to avoid unwanted results.")
        # group.addParam('ctfType', params.EnumParam, choices=['Apply', 'Wiener'],
        #                default=0, label="CTF correction type",
        #                display=params.EnumParam.DISPLAY_HLIST,
        #                condition="applyCTF", expertLevel=params.LEVEL_ADVANCED,
        #                help="* *Apply*: CTF is applied to the projection generated from the reference map\n"
        #                     "* *Wiener*: input particle is CTF corrected by a Wiener fiter")
        # group.addParam("pad", params.IntParam, default=4,
        #                label="Padding factor",
        #                condition="applyCTF", expertLevel=params.LEVEL_ADVANCED,
        #                help="Determines the padding factor to be applied before computing "
        #                     "the Fourier Transform of the images to increase the frequency "
        #                     "content")
        form.addSection(label='Network')
        form.addParam('fineTune', params.BooleanParam, default=False, label="Fine tune previous network?",
                      help="If True, a previously trained deepPose network will be fine tuned based on the "
                           "new input parameters. Note that when this option is set, the input particles "
                           "must have a trained deepPose network associated (i.e. particles must come from "
                           "a **'angular align - deepPose'** protocol.")
        form.addParam('netProtocol', params.PointerParam, label="Previously trained network",
                      allowsNull=True,
                      pointerClass='TensorflowProtAngularAlignmentReconSiren',
                      condition="fineTune")
        group = form.addGroup("Network hyperparameters")
        group.addParam('nCandidates', params.IntParam, default=6,
                       expertLevel=params.LEVEL_ADVANCED,
                       label="Number of multi-head encoders?",
                       help="Number of multi-head encoders determining the number of candidates from which to predict "
                            "an image pose and shifts. The more candidates available, the finer the search will be, "
                            "leading to better estimation of the poses and the shifts. However, the training times will "
                            "also increase.")
        group.addParam('architecture', params.EnumParam, choices=['ConvNN', 'MPLNN'],
                       expertLevel=params.LEVEL_ADVANCED,
                       default=0, label="Network architecture", display=params.EnumParam.DISPLAY_HLIST,
                       help="* *ConvNN*: convolutional neural network\n"
                            "* *MLPNN*: multiperceptron neural network")
        group.addParam('stopType', params.EnumParam, choices=['Samples', 'Manual'],
                       default=1, label="How to compute total epochs?",
                       display=params.EnumParam.DISPLAY_HLIST,
                       help="* *Samples*: Epochs will be obtained from the total number of samples "
                            "the network will see\n"
                            "* *Epochs*: Total number of epochs is provided manually")
        group.addParam('epochs', params.IntParam, default=500, condition="stopType==1",
                       label='Number of training epochs')
        group.addParam('maxSamples', params.IntParam, default=1000000, condition="stopType==0",
                       label="Samples",
                       help='Maximum number of samples seen during network training')
        group.addParam('batch_size', params.IntParam, default=64, label='Number of images in batch',
                       help="Number of images that will be used simultaneously for every training step. "
                            "We do not recommend to change this value unless you experience memory errors. "
                            "In this case, value should be decreased.")
        group.addParam('xla', params.BooleanParam, default=True, label="Allow XLA compilation?",
                       help="When XLA compilation is allowed, extra optimizations are applied during neural network "
                            "training increasing the training performance. However, XLA will only work with compatible "
                            "GPUs. If any error is experienced, set to No.")
        group.addParam('tensorboard', params.BooleanParam, default=True, label="Allow Tensorboard visualization?",
                       help="Tensorboard visualization provides a complete real-time report to supervise the training "
                            "of the neural network. However, for very large networks RAM requirements to save the "
                            "Tensorboard logs might overflow. If your process unexpectedly finishes when saving the "
                            "network callbacks, please, set this option to NO and restart the training.")
        group = form.addGroup("Extra network parameters")
        group.addParam('split_train', params.FloatParam, default=1.0, label='Traning dataset fraction',
                       help="This value (between 0 and 1) determines the fraction of images that will "
                            "be used to train the network.")
        group = form.addGroup("Logger")
        group.addParam('debugMode', params.BooleanParam, default=False, label='Debugging mode',
                       help="If you experience any error during the training execution, we recommend setting "
                            "this parameter to True followed by a restart of this protocol to generate a more "
                            "informative logging file.")
        form.addSection(label='Cost function')
        form.addParam("l1Reg", params.FloatParam, default=0.1, label="L1 loss regularization",
                      help="Determines the weight of the L1 map minimization in the cost function. L1 is moslty used to "
                           "decrease the amount of noise in the map learned by the network. We do not recommend to touch "
                           "this parameter")
        form.addParam("tvReg", params.FloatParam, default=0.1, label="Total variation loss regularization",
                      help="Determines the weight of the TV map minimization in the cost function. TV is moslty used to "
                           "promote densitiy smoothness while focusing on the preservation of edges present in "
                           "the CryoEM map.")
        form.addParam("mseReg", params.FloatParam, default=0.1, label="MSE loss regularization",
                      help="Determines the weight of the MSE variation map minimization in the cost function. "
                           "MSE is moslty used to promote density smoothnes while focusing on ensuring a continuous "
                           "transition of the density values")
        form.addParam("onlyPos", params.BooleanParam, default=False, label="Exclude negative values?",
                      condition="not isinstance(inputParticles, SetOfAverages)")
        form.addParam("udLambda", params.FloatParam, default=1e-6, label="Uniform distribution regularization",
                      expertLevel=params.LEVEL_ADVANCED,
                      help="Determines how much the network will enforce particles to have an uniform angular "
                           "distribution")  # default=0.000001
        form.addParam("unLambda", params.FloatParam, default=0.0001, label="Unit norm regularization",
                      expertLevel=params.LEVEL_ADVANCED,
                      help="Determines whether to enforce a 6D representation with unit norm.")
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
            'fnOutDir': self._getExtraPath(),
            'fnSymMatrices': self._getExtraPath("sym_matrices.npy")
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
            curr_vol_dim = ImageHandler(getXmippFileName(inputVolume)).getDimensions()[-1]
            if curr_vol_dim != self.newXdim:
                sr_vol = self.inputVolume.get().getSamplingRate()
                freq = sr_vol / (2. * (Xdim / self.newXdim) * sr_vol)
                params = "-i %s --fourier low_pass %f" % \
                         (fnVol, freq)
                self.runJob("xmipp_transform_filter", params, numberOfMpi=self.numberOfMpi.get(),
                            env=xmipp3.Plugin.getEnviron())

                self.runJob("xmipp_image_resize",
                            "-i %s --fourier %d " % (fnVol, self.newXdim), numberOfMpi=1, env=xmipp3.Plugin.getEnviron())

        if self.inputVolumeMask.get():  # Mask reference
            ih = ImageHandler()
            inputMask = self.inputVolumeMask.get().getFileName()
            if inputMask:
                ih.convert(getXmippFileName(inputMask), fnVolMask)
                curr_mask_dim = ImageHandler(getXmippFileName(inputMask)).getDimensions()[-1]
                if curr_mask_dim != self.newXdim:
                    self.runJob("xmipp_image_resize",
                                "-i %s --dim %d --interp nearest" % (fnVolMask, self.newXdim), numberOfMpi=1,
                                env=xmipp3.Plugin.getEnviron())
        else:
            ImageHandler().createCircularMask(fnVolMask, boxSize=self.newXdim, is3D=True)

        if self.refinement.get():
            writeSetOfParticles(inputParticles, imgsFn, alignType=ALIGN_PROJ)
        else:
            writeSetOfParticles(inputParticles, imgsFn, alignType=ALIGN_NONE)

        sr = inputParticles.getSamplingRate()
        if self.considerCTF.get():
            # Wiener filter
            corrected_stk = self._getTmpPath('corrected_particles.xmd')
            args = "-i %s -o %s --save_metadata_stack %s --keep_input_columns --sampling_rate %f --wc -1.0" \
                   % (imgsFn, self._getTmpPath('corrected_particles.mrcs'), corrected_stk, sr)
            program = 'xmipp_ctf_correct_wiener2d'
            self.runJob(program, args, numberOfMpi=self.numberOfThreads.get(), env=xmipp3.Plugin.getEnviron())
        else:
            corrected_stk = imgsFn

        if self.newXdim != Xdim:
            freq = sr / (2. * (Xdim / self.newXdim) * sr)
            params = "-i %s -o %s --save_metadata_stack %s --keep_input_columns --fourier low_pass %f" % \
                     (corrected_stk,
                      self._getTmpPath('scaled_particles.mrcs'),
                      self._getExtraPath('scaled_particles.xmd'),
                      freq)
            self.runJob("xmipp_transform_filter", params, numberOfMpi=self.numberOfMpi.get(),
                        env=xmipp3.Plugin.getEnviron())

            params = "-i %s --fourier %d" % \
                     (self._getTmpPath('scaled_particles.mrcs'),
                      self.newXdim)
            if self.numberOfMpi.get() > 1:
                params += " --mpi_job_size %d" % int(inputParticles.getSize() / self.numberOfMpi.get())
            self.runJob("xmipp_image_resize", params, numberOfMpi=self.numberOfMpi.get(),
                        env=xmipp3.Plugin.getEnviron())

            # Update shifts based on new scale
            if self.refinement.get():
                md = XmippMetaData(self._getExtraPath('scaled_particles.xmd'))
                md[:, ["shiftX", "shiftY"]] *= (self.newXdim / Xdim)
                md.write(self._getExtraPath('scaled_particles.xmd'), overwrite=True)

            moveFile(self._getExtraPath('scaled_particles.xmd'), imgsFn)

        # Removing Xmipp Phantom config file
        if self.considerCTF.get():
            self.runJob('rm', self._getTmpPath('corrected_particles.mrcs'))

        # Symmetry
        SL = xmippLib.SymList()
        listSymmetryMatrices = np.asarray(SL.getSymmetryMatrices(self.symmetry.get()))
        np.save(self._getFileName("fnSymMatrices"), listSymmetryMatrices)


    def trainingStep(self):
        md_file = self._getFileName('imgsFn')
        out_path = self._getExtraPath('network')
        if not os.path.isdir(out_path):
            os.mkdir(out_path)
        # pad = self.pad.get()
        batch_size = self.batch_size.get()
        split_train = self.split_train.get()
        nCandidates = self.nCandidates.get()
        l1Reg = self.l1Reg.get()
        tvReg = self.tvReg.get()
        mseReg = self.mseReg.get()
        udLambda = self.udLambda.get()
        unLambda = self.unLambda.get()
        self.newXdim = self.boxSize.get()
        correctionFactor = self.inputParticles.get().getXDim() / self.newXdim
        sr = correctionFactor * self.inputParticles.get().getSamplingRate()
        onlyPos = self.onlyPos.get() if not isinstance(self.inputParticles.get(), SetOfAverages) else True
        # applyCTF = self.applyCTF.get()
        xla = self.xla.get()
        tensorboard = self.tensorboard.get()
        args = "--md_file %s --out_path %s --batch_size %d " \
               "--shuffle --split_train %f --pad 2 --n_candidates %d " \
               "--sr %f --l1_reg %f --tv_reg %f --mse_reg %f --ud_lambda %f --un_lambda %f " \
               % (md_file, out_path, batch_size, split_train, nCandidates, sr,
                  l1Reg, tvReg, mseReg, udLambda, unLambda)

        if self.inputVolume.get() and not self.refinement.get():
            args += "--only_pose "

        if self.stopType.get() == 0:
            args += " --max_samples_seen %d" % self.maxSamples.get()
        else:
            args += " --epochs %d" % self.epochs.get()

        if self.architecture.get() == 0:
            args += " --architecture convnn"
        elif self.architecture.get() == 1:
            args += " --architecture mlpnn"

        # if self.ctfType.get() == 0:
        #     args += " --ctf_type apply"
        # elif self.ctfType.get() == 1:
        #     args += " --ctf_type wiener"

        if onlyPos:
            args += " --only_pos"

        if self.useHet:
            args += " --heterogeneous"

        if self.fineTune.get():
            netProtocol = self.netProtocol.get()
            modelPath = glob(netProtocol._getExtraPath(os.path.join('network', 'reconsiren_model*')))
            args += " --weigths_file %s" % modelPath

        if xla:
            args += " --jit_compile"

        if tensorboard:
            args += " --tensorboard"

        if self.useGpu.get():
            gpu_list = ','.join([str(elem) for elem in self.getGpuList()])
            args += " --gpu %s" % gpu_list

        if self.debugMode.get():
            log_level = 0
        else:
            log_level = 2

        program = flexutils.Plugin.getTensorflowProgram("train_reconsiren.py", python=False,
                                                        log_level=log_level)
        self.runJob(program, args, numberOfMpi=1)

    def predictStep(self):
        md_file = self._getFileName('imgsFn')
        weigths_file = glob(self._getExtraPath(os.path.join('network', 'reconsiren_model*')))[0]
        # pad = self.pad.get()
        self.newXdim = self.boxSize.get()
        correctionFactor = self.inputParticles.get().getXDim() / self.newXdim
        sr = correctionFactor * self.inputParticles.get().getSamplingRate()
        onlyPos = self.onlyPos.get() if not isinstance(self.inputParticles.get(), SetOfAverages) else True
        # applyCTF = self.applyCTF.get()
        nCandidates = self.nCandidates.get()
        args = "--md_file %s --weigths_file %s --pad 2 --n_candidates %d --sr %f " \
               % (md_file, weigths_file, nCandidates, sr)

        if self.inputVolume.get() and not self.refinement.get():
            args += "--only_pose "

        # if self.ctfType.get() == 0:
        #     args += " --ctf_type apply"
        # elif self.ctfType.get() == 1:
        #     args += " --ctf_type wiener"

        if self.architecture.get() == 0:
            args += " --architecture convnn"
        elif self.architecture.get() == 1:
            args += " --architecture mlpnn"

        if self.filterDecoded.get():
            args += " --apply_filter"

        if onlyPos:
            args += " --only_pos"

        if self.useHet:
            args += " --heterogeneous"

        if self.useGpu.get():
            gpu_list = ','.join([str(elem) for elem in self.getGpuList()])
            args += " --gpu %s" % gpu_list

        program = flexutils.Plugin.getTensorflowProgram("predict_reconsiren.py", python=False)
        self.runJob(program, args, numberOfMpi=1)

    def createOutputStep(self):
        inputParticles = self.inputParticles.get()
        Xdim = inputParticles.getXDim()
        self.newXdim = self.boxSize.get()
        model_path = glob(self._getExtraPath(os.path.join('network', 'reconsiren_model*')))[0]
        md_file = self._getFileName('imgsFn')

        metadata = XmippMetaData(md_file)
        rot = metadata[:, 'angleRot']
        tilt = metadata[:, 'angleTilt']
        psi = metadata[:, 'anglePsi']
        shift_x = metadata[:, 'shiftX']
        shift_y = metadata[:, 'shiftY']
        loss_cons = metadata[:, "reproj_cons_error"]
        if self.useHet.get():
            loss_het = metadata[:, "reproj_het_error"]

        inputSet = self.inputParticles.get()
        partSet = self._createSetOfParticles()

        partSet.copyInfo(inputSet)
        partSet.setHasCTF(inputSet.hasCTF())
        partSet.setAlignmentProj()

        correctionFactor = Xdim / self.newXdim

        inverseTransform = partSet.getAlignment() == ALIGN_PROJ

        idx = 0
        for particle in inputSet.iterItems():
            shifts, angles = np.asarray([0, 0, 0]), np.asarray([0, 0, 0])

            # Apply delta angles
            angles[0] = rot[idx]
            angles[1] = tilt[idx]
            angles[2] = psi[idx]

            # Apply delta shifts
            shifts[0] = correctionFactor * shift_x[idx]
            shifts[1] = correctionFactor * shift_y[idx]

            # Set new transformation matrix
            tr_matrix = matrixFromGeometry(shifts, angles, inverseTransform)
            tr = Transform()
            tr.setMatrix(tr_matrix)
            particle.setTransform(tr)

            particle.reproj_cons_error = Float(loss_cons[idx])
            if self.useHet.get():
                particle.reproj_het_error = Float(loss_het[idx])
                particle.class_agreement = Float(np.abs(loss_het[idx] - loss_cons[idx]) / loss_cons[idx])

            partSet.append(particle)

            idx += 1

        partSet.modelPath = String(model_path)
        partSet.considerCTF = Boolean(self.considerCTF.get())
        partSet.onlyPos = Boolean(self.onlyPos.get() if not isinstance(inputParticles, SetOfAverages) else True)
        partSet.useHet = Boolean(self.useHet.get())

        if self.inputVolume.get():
            inputVolume = self.inputVolume.get().getFileName()
            partSet.refMap = String(inputVolume)

        if self.inputVolumeMask.get():
            inputMask = self.inputVolumeMask.get().getFileName()
            partSet.refMask = String(inputMask)

        if self.architecture.get() == 0:
            partSet.architecture = String("convnn")
        elif self.architecture.get() == 1:
            partSet.architecture = String("mlpnn")

        # if self.ctfType.get() == 0:
        #     partSet.ctfType = String("apply")
        # elif self.ctfType.get() == 1:
        #     partSet.ctfType = String("wiener")

        # partSet.pad = Integer(self.pad.get())

        self._defineOutputs(outputParticles=partSet)
        self._defineTransformRelation(self.inputParticles, partSet)

        if self.inputVolume.get() is None:
            outVol = Volume()
            outVol.setSamplingRate(inputParticles.getSamplingRate())
            outVol.setLocation(self._getExtraPath('decoded_map.mrc'))

            ImageHandler().scaleSplines(self._getExtraPath('decoded_map.mrc'),
                                        self._getExtraPath('decoded_map.mrc'),
                                        finalDimension=inputParticles.getXDim(), overwrite=True)

            # Set correct sampling rate in volume header
            ImageHandler().setSamplingRate(self._getExtraPath('decoded_map.mrc'),
                                           inputParticles.getSamplingRate())

            self._defineOutputs(outputVolume=outVol)
            self._defineTransformRelation(self.inputParticles, outVol)

        if self.useHet.get():
            labels = metadata[:, "cluster_labels"].astype(int)
            unique_labels = np.unique(labels).astype(int)
            partIds = list(inputParticles.getIdSet())

            hetClasses = self._createSetOfClasses3D(inputParticles)
            hetVols = self._createSetOfVolumes()
            hetVols.setSamplingRate(inputParticles.getSamplingRate())
            for label in unique_labels:
                vol = self._getExtraPath(f'decoded_map_{label:02}.mrc')
                hetVol = Volume()
                hetVol.setSamplingRate(inputParticles.getSamplingRate())
                hetVol.setLocation(vol)
                hetVols.append(hetVol)

                ImageHandler().scaleSplines(vol, vol, finalDimension=inputParticles.getXDim(), overwrite=True)

                # Set correct sampling rate in volume header
                ImageHandler().setSamplingRate(vol, inputParticles.getSamplingRate())

                cls = Class3D()
                cls.copyInfo(inputParticles)
                cls.setObjId(int(label + 1))
                cls.setHasCTF(inputParticles.hasCTF())
                cls.setAcquisition(inputParticles.getAcquisition())
                cls.setRepresentative(hetVol)
                cls.reproj_cons_error = Float(0.0)
                cls.reproj_het_error = Float(0.0)
                cls.class_agreement = Float(0.0)
                hetClasses.append(cls)
                enabledClass = hetClasses[cls.getObjId()]
                enabledClass.enableAppend()
                mean_reproj_cons_error = 0.0
                mean_reproj_het_error = 0.0
                mean_class_agreement = 0.0
                for itemId in np.argwhere(labels == label)[..., 0]:
                    item = inputParticles[partIds[int(itemId)]]
                    item.reproj_cons_error = Float(loss_cons[itemId])
                    item.reproj_het_error = Float(loss_het[itemId])
                    item.class_agreement = Float(1. / np.abs(loss_het[itemId] + loss_cons[itemId]))
                    mean_reproj_cons_error += loss_cons[itemId]
                    mean_reproj_het_error += loss_het[itemId]
                    # mean_class_agreement += 1. / np.abs(loss_het[itemId] + loss_cons[itemId])
                    enabledClass.append(item)
                mean_reproj_cons_error /= len(np.argwhere(labels == label)[..., 0])
                mean_reproj_het_error /= len(np.argwhere(labels == label)[..., 0])
                # mean_class_agreement /= len(np.argwhere(labels == label)[..., 0])
                mean_class_agreement = 1. / np.abs(mean_reproj_cons_error + mean_reproj_het_error)
                enabledClass.reproj_cons_error = Float(mean_reproj_cons_error)
                enabledClass.reproj_het_error = Float(mean_reproj_het_error)
                enabledClass.class_agreement = Float(mean_class_agreement)
                hetClasses.update(enabledClass)

            self._defineOutputs(hetVolumes=hetVols)
            self._defineOutputs(hetClasses=hetClasses)
            self._defineTransformRelation(self.inputParticles, hetVols)
            self._defineTransformRelation(self.inputParticles, hetClasses)


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

    # --------------------------- INFO functions -----------------------------
    def _summary(self):
        summary = []
        logFile = os.path.abspath(self._getLogsPath()) + "/run.stdout"
        with open(logFile, "r") as fi:
            for ln in fi:
                if ln.startswith("GPU memory has"):
                    summary.append(ln)
                    break
        return summary

    # ----------------------- VALIDATE functions -----------------------
    def validate(self):
        """ Try to find errors on define params. """
        errors = []

        mask = self.inputVolumeMask.get()
        if mask is not None:
            data = ImageHandler(mask.getFileName()).getData()
            if not np.all(np.logical_and(data >= 0, data <= 1)):
                errors.append("Mask provided is not binary. Please, provide a binary mask")

        return errors
