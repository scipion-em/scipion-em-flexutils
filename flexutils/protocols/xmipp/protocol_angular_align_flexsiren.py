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
import re
from glob import glob
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

from xmipp3.convert import createItemMatrix, setXmippAttributes, writeSetOfParticles, \
    geometryFromMatrix, matrixFromGeometry
import xmipp3

import flexutils
import flexutils.constants as const
from flexutils.utils import getXmippFileName, coordsToMap, saveMap
from flexutils.protocols.xmipp.utils.pdb_parser import AtomicModelParser


class TensorflowProtAngularAlignmentFlexSIREN(ProtAnalysis3D, ProtFlexBase):
    """ Protocol for flexible angular alignment with the FlexSIREN algortihm. """
    _label = 'flexible align - FlexSIREN'
    _lastUpdateVersion = VERSION_2_0
    _subset = ["bb", "all"]

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
        group = form.addGroup("Data")
        group.addParam('inputParticles', params.PointerParam, label="Input particles",
                       pointerClass='SetOfParticles,SetOfParticlesFlex')
        group.addParam('referenceType', params.EnumParam, choices=['Volume', 'Structure'],
                       default=0, label="Reference type", display=params.EnumParam.DISPLAY_HLIST,
                       help="Determine which type of reference will be used to compute the motions. "
                            "In general, Structure will lead to faster and more accurate estimations "
                            "if available.")
        group.addParam('inputVolume', params.PointerParam,
                       label="Input volume", pointerClass='Volume')
        group.addParam('inputVolumeMask', params.PointerParam,
                       label="Input volume mask", pointerClass='VolumeMask',
                       help="Two different type of mask could be provided:\n"
                            "     * Macromolecular mask: A binary (non-smooth) mask telling where the protein is in "
                            "the volume. The tighter the mask is to the protein the better.\n"
                            "     * Rigid region mask: An id mask determining a set of rigid regions. Rigid regions "
                            "will move coordinately, which may be useful to prevent overfitting compared o the protein "
                            "mask cases (per point displacement)."
                       )
        group.addParam('inputStruct', params.PointerParam, condition="referenceType==1",
                       label="Input structure", pointerClass='AtomStruct',
                       help="Reference structure should be aligned within Scipion to the map reconstructed "
                            "from the input particles. This will ensure that the structure coordinates are "
                            "properly placed in the expected reference frame.")
        group.addParam("atomSubset", params.EnumParam, label="Atoms considered",
                       choices=['Backbone', 'Full'], default=0, condition="referenceType==1",
                       help="Atoms to be considered for the computation of the normal modes. Options include: \n"
                            "\t **Backbone**: Use protein backbone only\n"
                            "\t **Full**: Use all the atomic structure")
        group.addParam('boxSize', params.IntParam, default=128,
                       label='Downsample particles to this box size', expertLevel=params.LEVEL_ADVANCED,
                       help='In general, downsampling the particles will increase performance without compromising '
                            'the estimation the deformation field for each particle. Note that output particles will '
                            'have the original box size, and FlexSIREN coefficients will be modified to work with the '
                            'original size images')
        group = form.addGroup("FlexSIREN Parameters (Advanced)",
                              expertLevel=params.LEVEL_ADVANCED)
        group.addParam('latDim', params.IntParam, default=8,
                       label='Latent space dimension',
                       expertLevel=params.LEVEL_ADVANCED)
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
        form.addParam('fineTune', params.BooleanParam, default=False, label="Fine tune previous network?",
                      help="If True, a previously trained deepPose network will be fine tuned based on the "
                           "new input parameters. Note that when this option is set, the input particles "
                           "must have a trained deepPose network associated (i.e. particles must come from "
                           "a **'angular align - deepPose'** protocol.")
        form.addParam('netProtocol', params.PointerParam, label="Previously trained network",
                      allowsNull=True,
                      pointerClass='TensorflowProtAngularAlignmentFlexSIREN',
                      condition="fineTune")
        group = form.addGroup("Network hyperparameters")
        group.addParam('architecture', params.EnumParam, choices=['ConvNN', 'MPLNN'],
                       expertLevel=params.LEVEL_ADVANCED,
                       default=1, label="Network architecture", display=params.EnumParam.DISPLAY_HLIST,
                       help="* *ConvNN*: convolutional neural network\n"
                            "* *MLPNN*: multiperceptron neural network")
        group.addParam('stopType', params.EnumParam, choices=['Samples', 'Manual'],
                       default=1, label="How to compute total epochs?",
                       display=params.EnumParam.DISPLAY_HLIST,
                       help="* *Samples*: Epochs will be obtained from the total number of samples "
                            "the network will see\n"
                            "* *Epochs*: Total number of epochs is provided manually")
        group.addParam('epochs', params.IntParam, default=20, condition="stopType==1",
                       label='Number of training epochs')
        group.addParam('maxSamples', params.IntParam, default=1000000, condition="stopType==0",
                       label="Samples",
                       help='Maximum number of samples seen during network training')
        group.addParam('batch_size', params.IntParam, default=8, label='Number of images in batch',
                       help="Number of images that will be used simultaneously for every training step. "
                            "We do not recommend to change this value unless you experience memory errors. "
                            "In this case, value should be decreased.")
        group.addParam('lr', params.FloatParam, default=1e-5, label='Learning rate',
                       help="The learning rate determines how fast the network will train based on the "
                            "seen samples. The larger the value, the faster the network although divergence "
                            "might occur. We recommend decreasing the learning rate value if this happens.")
        group.addParam('xla', params.BooleanParam, default=True, label="Allow XLA compilation?",
                       help="When XLA compilation is allowed, extra optimizations are applied during neural network "
                            "training increasing the training performance. However, XLA will only work with compatible "
                            "GPUs. If any error is experienced, set to No.")
        group.addParam('tensorboard', params.BooleanParam, default=True, label="Allow Tensorboard visualization?",
                       help="Tensorboard visualization provides a complete real-time report to supervides the training "
                            "of the neural network. However, for very large networks RAM requirements to save the "
                            "Tensorboard logs might overflow. If your process unexpectedly finishes when saving the "
                            "network callbacks, please, set this option to NO and restart the training.")
        group = form.addGroup("Extra network parameters")
        group.addParam('refinePose', params.BooleanParam, default=True, label="Refine pose?",
                       help="If True, the neural network will be also trained to refine the angular "
                            "and shift assignation of the particles to make it more consistent with the "
                            "flexibility estimation. Otherwise, only heterogeneity information will be "
                            "estimated.")
        group.addParam('split_train', params.FloatParam, default=1.0, label='Traning dataset fraction',
                       help="This value (between 0 and 1) determines the fraction of images that will "
                            "be used to train the network.")
        group.addParam('step', params.IntParam, default=1, label='Points step', condition="referenceType==0",
                       help="How many points (voxels) to skip during the training computations. "
                            "A value of 1 means that all point within the mask provided in the input "
                            "will be used. A value of 2 implies that half of the point will be skipped "
                            "to increase the performance.")
        group = form.addGroup("Disentanglement")
        group.addParam('disPose', params.BooleanParam, default=True, label='Pose disentanglement?',
                       help="If True, the neural network will be also trained to disentangle the pose information "
                            "from the conformational landscape.")
        group.addParam('poseReg', params.FloatParam, default=0.001, label='Pose disentanglement factor',
                       expertLevel=params.LEVEL_ADVANCED, condition="disPose",
                       help="Pose disentanglement factor to be considered while computing the cost function")
        group.addParam('disCTF', params.BooleanParam, default=True, label='CTF disentanglement?',
                       help="If True, the neural network will be also trained to disentangle the CTF information "
                            "from the conformational landscape.")
        group.addParam('ctfReg', params.FloatParam, default=0.001, label='CTF disentanglement factor',
                       expertLevel=params.LEVEL_ADVANCED, condition="disCTF",
                       help="CTF disentanglement factor to be considered while computing the cost function")
        group = form.addGroup("Logger")
        group.addParam('debugMode', params.BooleanParam, default=False, label='Debugging mode',
                       help="If you experience any error during the training execution, we recommend setting "
                            "this parameter to True followed by a restart of this protocol to generate a more "
                            "informative logging file.")
        form.addSection(label='Cost function')
        form.addParam('costFunction', params.EnumParam,
                      choices=['Correlation', 'Fourier Phase Correlation', 'MSE', 'MAE'],
                      default=0, label="Cost function type", display=params.EnumParam.DISPLAY_HLIST,
                      help="Determine the cost function to be minimized during the neural network training. Both, "
                           "Correlation and Fourier Phase Correlation will yield similar results. However, Fourier "
                           "Shell Correlation allows excluding high frequency information by masking in the Fourier "
                           "space. This might help preveting overfitting in scenarios with low Signal to Noise ratios "
                           "at the expense of slightly increasing computation time. MSE and MAE focuses more on "
                           "voxel/pixel values, which might lead to more accurate results (although it may result in "
                           "overfitting). Compared to MSE, MAE will give a lower weight to unwanted values, so it "
                           "might be a little more robust results when SNR is low.")
        form.addParam('maskRadius', params.FloatParam, default=0.85, label="Mask radius (%)",
                      condition="costFunction==1",
                      help="Determine the radius (in percentage) of the circular mask to be applied to the Fourier "
                           "Transform of the images. A value of 1 implies that the circular mask is inscribed to the "
                           "bounding box the Fourier Transform.")
        form.addParam("smoothMask", params.BooleanParam, default=True, label="Smooth mask?",
                      condition="costFunction==1",
                      help="If True, the mask applied to the Fourier Transform of the particle images will have a smooth"
                           "vanishing transition.")
        form.addParam("regNorm", params.FloatParam, default=0.0001, label="FlexSIREN coefficient norm regularization",
                      help="Regularization factor determining how big FlexSIREN coefficients are allowed to be."
                           "The larger the value, the smaller the coefficients that will be found (leading to a "
                           "larger restriction of the motions). We do not recommend touching this value unless motions "
                           "observed are greatly dampened.")
        form.addParam("regBond", params.FloatParam, default=0.01, label="Bond loss regularization",
                      condition="referenceType==1",
                      help="Regularization factor determining how stiff bond distances will be when deforming the "
                           "atomic model")
        form.addParam("regAngle", params.FloatParam, default=0.001, label="Hedra loss regularization",
                      condition="referenceType==1",
                      help="Regularization factor determining how stiff hedra angles distances will be when deforming "
                           "the atomic model")
        form.addParam("regClashes", params.FloatParam, default=0.001, label="Clashes regularization",
                      allowsNull=True,
                      condition="referenceType==1",
                      help="Regularization factor determining how the importance of bonded and non-bonded clashes in "
                           "the cost function. NOTE: Clashes will only be used if Tensorflow version is (>= 2.15.0). "
                           "If this is not the case, you may update the plugin to update Tensorflow to the last "
                           "compatible version of your system. If clashes are not to be included in the cost, leave "
                           "this field empty.")
        form.addParallelSection(threads=0, mpi=4)

    def _createFilenameTemplates(self):
        """ Centralize how files are called """
        myDict = {
            'imgsFn': self._getExtraPath('input_particles.xmd'),
            'fnVol': self._getExtraPath('volume.mrc'),
            'fnVolMask': self._getExtraPath('mask.mrc'),
            'fnStruct': self._getExtraPath('structure.txt'),
            'fnBond': self._getExtraPath("bonds.txt"),
            'fnDihedral': self._getExtraPath("dihedrals.txt"),
            'fnCA': self._getExtraPath("ca_indices.txt"),
            'fnOutDir': self._getExtraPath()
        }
        self._updateFilenamesDict(myDict)

    # --------------------------- INSERT steps functions --------------------------------------------
    def _insertAllSteps(self):
        self._createFilenameTemplates()
        self._insertFunctionStep(self.writeMetaDataStep)
        self._insertFunctionStep(self.trainingStep)
        self._insertFunctionStep(self.predictStep)
        self._insertFunctionStep(self.createOutputStep)

    # --------------------------- STEPS functions ---------------------------------------------------
    def writeMetaDataStep(self):
        imgsFn = self._getFileName('imgsFn')
        fnVol = self._getFileName('fnVol')
        fnVolMask = self._getFileName('fnVolMask')
        md_file = self._getFileName('imgsFn')

        inputParticles = self.inputParticles.get()
        Xdim = inputParticles.getXDim()
        self.newXdim = self.boxSize.get()
        correctionFactor = Xdim / self.newXdim
        i_sr = 1. / (correctionFactor * inputParticles.getSamplingRate())

        # Map reference
        ih = ImageHandler()
        inputVolume = self.inputVolume.get().getFileName()
        ih.convert(getXmippFileName(inputVolume), fnVol)
        curr_vol_dim = ImageHandler(getXmippFileName(inputVolume)).getDimensions()[-1]
        if curr_vol_dim != self.newXdim:
            self.runJob("xmipp_image_resize",
                        "-i %s --fourier %d " % (fnVol, self.newXdim), numberOfMpi=1, env=xmipp3.Plugin.getEnviron())
        ih.setSamplingRate(fnVol, inputParticles.getSamplingRate())

        inputMask = self.inputVolumeMask.get().getFileName()
        if inputMask:
            ih.convert(getXmippFileName(inputMask), fnVolMask)
            curr_mask_dim = ImageHandler(getXmippFileName(inputMask)).getDimensions()[-1]
            if curr_mask_dim != self.newXdim:
                self.runJob("xmipp_image_resize",
                            "-i %s --dim %d --interp nearest" % (fnVolMask, self.newXdim), numberOfMpi=1,
                            env=xmipp3.Plugin.getEnviron())

        if self.referenceType.get() == 1:  # Structure reference
            inputVolume = self.inputVolume.get().getFileName()
            structure_file = self._getFileName('fnStruct')
            bonds_file = self._getFileName('fnBond')
            dihedrals_file = self._getFileName('fnDihedral')
            ca_file = self._getFileName('fnCA')
            parser = AtomicModelParser(self.inputStruct.get().getFileName(), self._subset[self.atomSubset.get()])
            pdb_coordinates = parser.get_atom_coordinates()
            covalent = parser.get_covalent_bonds()
            dihedrals = parser.get_dihedral_angles()
            ca_indices = parser.get_ca_indices()
            pdb_coordinates *= i_sr
            ih = ImageHandler(getXmippFileName(inputVolume))
            vol = ih.getData()
            factor = 0.5 * ih.getDimensions()[-1]
            pdb_indices = np.round(pdb_coordinates + factor).astype(int)
            values = vol[pdb_indices[:, 2], pdb_indices[:, 1], pdb_indices[:, 0]]
            pdb_coordinates = np.c_[pdb_coordinates, values]
            np.savetxt(structure_file, pdb_coordinates)
            np.savetxt(bonds_file, covalent)
            np.savetxt(dihedrals_file, dihedrals)
            np.savetxt(ca_file, ca_indices)

        # Write particles
        writeSetOfParticles(inputParticles, imgsFn)

        # Write extra attributes (if needed)
        md = XmippMetaData(md_file)
        if hasattr(inputParticles.getFirstItem(), "_xmipp_subtomo_labels"):
            labels = np.asarray([int(particle._xmipp_subtomo_labels) for particle in inputParticles.iterItems()])
            md[:, "subtomo_labels"] = labels
        md.write(md_file, overwrite=True)

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

    def trainingStep(self):
        md_file = self._getFileName('imgsFn')
        out_path = self._getExtraPath('network')
        if not os.path.isdir(out_path):
            os.mkdir(out_path)
        latDim = self.latDim.get()
        pad = self.pad.get()
        batch_size = self.batch_size.get()
        step = self.step.get()
        split_train = self.split_train.get()
        lr = self.lr.get()
        regNorm = self.regNorm.get()
        correctionFactor = self.inputParticles.get().getXDim() / self.boxSize.get()
        sr = correctionFactor * self.inputParticles.get().getSamplingRate()
        applyCTF = self.applyCTF.get()
        xla = self.xla.get()
        tensorboard = self.tensorboard.get()
        args = "--md_file %s --out_path %s --lat_dim %d --batch_size %d " \
               "--shuffle --split_train %f --pad %d --sr %f --apply_ctf %d --lr %f --regNorm %f" \
               % (md_file, out_path, latDim, batch_size, split_train, pad, sr,
                  applyCTF, lr, regNorm)

        if self.stopType.get() == 0:
            args += " --max_samples_seen %d" % self.maxSamples.get()
        else:
            args += " --epochs %d" % self.epochs.get()

        if self.referenceType.get() == 0:
            args += " --step %d" % step
        else:
            regClashes = self.regClashes.get()
            args += " --step 1 --regBond %f --regAngle %f" % (self.regBond.get(), self.regAngle.get())

            if regClashes is not None:
                args += " --regClashes %f" % regClashes

        if self.costFunction.get() == 0:
            args += " --cost corr"
        elif self.costFunction.get() == 1:
            args += " --cost fpc --radius_mask %f" % self.maskRadius.get()
            if self.smoothMask.get():
                args += " --smooth_mask"
        elif self.costFunction.get() == 2:
            args += " --cost mse"
        elif self.costFunction.get() == 3:
            args += " --cost mae"

        if self.refinePose.get():
            args += " --refine_pose"

        if self.fineTune.get():
            netProtocol = self.netProtocol.get()
            modelPath = glob(netProtocol._getExtraPath(os.path.join('network', 'flexsiren_model*')))[0]
            args += " --weigths_file %s" % modelPath

        if self.architecture.get() == 0:
            args += " --architecture convnn"
        elif self.architecture.get() == 1:
            args += " --architecture mlpnn"

        if self.ctfType.get() == 0:
            args += " --ctf_type apply"
        elif self.ctfType.get() == 1:
            args += " --ctf_type wiener"

        if self.disPose.get():
            args += " --pose_reg %f" % self.poseReg.get()

        if self.disCTF.get():
            args += " --ctf_reg %f" % self.ctfReg.get()

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

        program = flexutils.Plugin.getTensorflowProgram("train_flexsiren.py", python=False,
                                                        log_level=log_level)
        self.runJob(program, args, numberOfMpi=1)

    def predictStep(self):
        md_file = self._getFileName('imgsFn')
        weigths_file = glob(self._getExtraPath(os.path.join('network', 'flexsiren_model*')))[0]
        latDim = self.latDim.get()
        pad = self.pad.get()
        correctionFactor = self.inputParticles.get().getXDim() / self.boxSize.get()
        sr = correctionFactor * self.inputParticles.get().getSamplingRate()
        applyCTF = self.applyCTF.get()
        args = "--md_file %s --weigths_file %s --lat_dim %d --pad %d --sr %f " \
               "--apply_ctf %d" \
               % (md_file, weigths_file, latDim, pad, sr, applyCTF)

        if self.refinePose.get():
            args += " --refine_pose"

        if self.ctfType.get() == 0:
            args += " --ctf_type apply"
        elif self.ctfType.get() == 1:
            args += " --ctf_type wiener"

        if self.architecture.get() == 0:
            args += " --architecture convnn"
        elif self.architecture.get() == 1:
            args += " --architecture mlpnn"

        if self.disPose.get():
            args += " --pose_reg %f" % self.poseReg.get()

        if self.disCTF.get():
            args += " --ctf_reg %f" % self.ctfReg.get()

        if self.useGpu.get():
            gpu_list = ','.join([str(elem) for elem in self.getGpuList()])
            args += " --gpu %s" % gpu_list

        program = flexutils.Plugin.getTensorflowProgram("predict_flexsiren.py", python=False)
        self.runJob(program, args, numberOfMpi=1)

    def createOutputStep(self):
        inputParticles = self.inputParticles.get()
        Xdim = inputParticles.getXDim()
        self.newXdim = self.boxSize.get()
        model_path = glob(self._getExtraPath(os.path.join('network', 'flexsiren_model*')))[0]
        md_file = self._getFileName('imgsFn')

        metadata = XmippMetaData(md_file)
        z_space = np.asarray([np.fromstring(item, sep=',') for item in metadata[:, 'zCoefficients']])
        b_coeff = np.asarray([np.fromstring(item, sep=',') for item in metadata[:, 'bCoefficients']])

        if self.refinePose.get():
            delta_rot = metadata[:, 'delta_angle_rot']
            delta_tilt = metadata[:, 'delta_angle_tilt']
            delta_psi = metadata[:, 'delta_angle_psi']
            delta_shift_x = metadata[:, 'delta_shift_x']
            delta_shift_y = metadata[:, 'delta_shift_y']

        inputSet = self.inputParticles.get()
        partSet = self._createSetOfParticlesFlex(progName=const.FLEXSIREN, suffix="1")
        partSet_Zernike_like = self._createSetOfParticlesFlex(progName=const.ZERNIKE3D, suffix="2")

        partSet.copyInfo(inputSet)
        partSet_Zernike_like.copyInfo(inputSet)
        partSet.setHasCTF(inputSet.hasCTF())
        partSet_Zernike_like.setHasCTF(inputSet.hasCTF())
        partSet.getFlexInfo().setProgName(const.FLEXSIREN)
        partSet_Zernike_like.getFlexInfo().setProgName(const.ZERNIKE3D)
        partSet.setAlignmentProj()
        partSet_Zernike_like.setAlignmentProj()

        correctionFactor = Xdim / self.newXdim

        inverseTransform = partSet.getAlignment() == ALIGN_PROJ

        idx = 0
        for particle in inputSet.iterItems():

            outParticle = ParticleFlex(progName=const.FLEXSIREN)
            outParticle_Zernike_like = ParticleFlex(progName=const.ZERNIKE3D)
            outParticle.copyInfo(particle)
            outParticle_Zernike_like.copyInfo(particle)
            outParticle.getFlexInfo().setProgName(const.FLEXSIREN)
            outParticle_Zernike_like.getFlexInfo().setProgName(const.ZERNIKE3D)

            outParticle.setZFlex(z_space[idx])
            outParticle_Zernike_like.setZFlex(0.5 * Xdim * b_coeff[idx])

            if self.refinePose.get():
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
                outParticle_Zernike_like.getTransform().setMatrix(tr)

            idx += 1

            partSet.append(outParticle)
            partSet_Zernike_like.append(outParticle_Zernike_like)

        partSet.getFlexInfo().latDim = Integer(self.latDim.get())
        partSet.getFlexInfo().pad = Integer(self.pad.get())
        partSet.getFlexInfo().modelPath = String(model_path)

        inputMask = self.inputVolumeMask.get().getFileName()
        inputVolume = self.inputVolume.get().getFileName()
        partSet.getFlexInfo().refMask = String(inputMask)
        partSet.getFlexInfo().refMap = String(inputVolume)
        partSet.getFlexInfo().disPose = Boolean(self.disPose.get())
        partSet.getFlexInfo().disCTF = Boolean(self.disCTF.get())

        if self.refinePose.get():
            partSet.getFlexInfo().refPose = Boolean(True)
        else:
            partSet.getFlexInfo().refPose = Boolean(False)

        if self.architecture.get() == 0:
            partSet.getFlexInfo().architecture = String("convnn")
        elif self.architecture.get() == 1:
            partSet.getFlexInfo().architecture = String("mlpnn")

        if self.ctfType.get() == 0:
            partSet.getFlexInfo().ctfType = String("apply")
        elif self.ctfType.get() == 1:
            partSet.getFlexInfo().ctfType = String("wiener")

        partSet_Zernike_like.getFlexInfo().L1 = Integer(7)
        partSet_Zernike_like.getFlexInfo().L2 = Integer(7)
        partSet_Zernike_like.getFlexInfo().Rmax = Float(Xdim / 2)
        partSet_Zernike_like.getFlexInfo().refMask = String(inputMask)
        partSet_Zernike_like.getFlexInfo().refMap = String(inputVolume)

        self._defineOutputs(outputParticles=partSet, outputParticlesZernike3D=partSet_Zernike_like)
        self._defineTransformRelation(self.inputParticles, partSet)
        self._defineTransformRelation(self.inputParticles, partSet_Zernike_like)

    # --------------------------- UTILS functions --------------------------------------------
    def _updateParticle(self, item, row):
        setXmippAttributes(item, row, md.MDL_ANGLE_ROT, md.MDL_ANGLE_TILT,
                           md.MDL_ANGLE_PSI, md.MDL_SHIFT_X, md.MDL_SHIFT_Y,
                           md.MDL_FLIP, md.MDL_SPH_DEFORMATION,
                           md.MDL_SPH_COEFFICIENTS)
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

    # ----------------------- VALIDATE functions ----------------------------------------
    def validate(self):
        """ Try to find errors on define params. """
        errors = []

        mask = self.inputVolumeMask.get()
        if mask is not None:
            data = ImageHandler(mask.getFileName()).getData()
            if not np.all(np.logical_and(data >= 0, data <= 1)):
                errors.append("Mask provided is not binary. Please, provide a binary mask")

        return errors

    def _warnings(self):
        warnings = []

        num_particles = self.inputParticles.get().getSize()
        split_train = self.split_train.get()
        num_particles_train = int(split_train * num_particles)

        if num_particles_train > 1000000:
            warnings.append("The dataset you are using to train is quite large, which may lead to long training times. "
                            "If this is intended, you may ignore this warning. Otherwise, it is recommended to modify "
                            "the form parameter \"Traning dataset fraction\" to a lower value so that the number of "
                            "particles to train is smaller than 500k. In this way, the network will learn faster and "
                            "posteriorly the trained network will use the complete dataset in the prediction step to "
                            "reduce execution times.")

        return warnings
