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
from pyworkflow.utils.path import makePath
from pyworkflow import VERSION_2_0

from pwem.protocols import ProtAnalysis3D

import flexutils
from flexutils.protocols import ProtFlexBase


class TensorflowProtTrainFlexConsensus(ProtAnalysis3D, ProtFlexBase):
    """ Protocol to train a FlexConsensus network """
    _label = 'train - FlexConsensus'
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
        group = form.addGroup("Data")
        group.addParam('inputSets', params.MultiPointerParam,
                       label="Input particles", pointerClass='SetOfParticlesFlex')
        group = form.addGroup("Latent Space")
        group.addParam('latDim', params.IntParam, default=10, label='Latent space dimension',
                       expertLevel=params.LEVEL_ADVANCED,
                       help="Dimension of the FlexConsensus bottleneck (latent space dimension)")
        form.addSection(label='Network')
        form.addParam('epochs', params.IntParam, default=100, label='Number of training epochs')
        form.addParam('batch_size', params.IntParam, default=64, label='Number of images in batch',
                      help="Number of images that will be used simultaneously for every training step. "
                           "We do not recommend to change this value unless you experience memory errors. "
                           "In this case, value should be decreased.")
        form.addParam('split_train', params.FloatParam, default=1.0, label='Traning dataset fraction',
                      help="This value (between 0 and 1) determines the fraction of images that will "
                           "be used to train the network.")
        form.addParallelSection(threads=4, mpi=0)

    # --------------------------- INSERT steps functions --------------------------------------------
    def _insertAllSteps(self):
        self._insertFunctionStep(self.convertInputStep)
        self._insertFunctionStep(self.trainingStep)

    # --------------------------- STEPS functions ---------------------------------------------------
    def convertInputStep(self):
        data_path = self._getExtraPath("data")
        if not os.path.isdir(data_path):
            makePath(data_path)

        idx = 0
        for pointer_set in self.inputSets:
            particle_set = pointer_set.get()

            # Get data filename
            progName = particle_set.getFlexInfo().getProgName()
            data_file = progName + f"_{idx}.txt"

            # Read flexible space form particles
            z_flex = []
            for particle in particle_set.iterItems():
                z_flex.append(particle.getZFlex())
            z_flex = np.vstack(z_flex)

            # Save flexible space
            np.savetxt(os.path.join(data_path, data_file), z_flex)

            idx += 1

    def trainingStep(self):
        data_path = self._getExtraPath("data")
        out_path = self._getExtraPath()
        batch_size = self.batch_size.get()
        split_train = self.split_train.get()
        epochs = self.epochs.get()
        lat_dim = self.latDim.get()
        args = "--data_path %s --out_path %s --lat_dim %d --batch_size %d " \
               "--shuffle --split_train %f --epochs %d" \
               % (data_path, out_path, lat_dim, batch_size, split_train, epochs)

        if self.useGpu.get():
            gpu_list = ','.join([str(elem) for elem in self.getGpuList()])
            args += " --gpu %s" % gpu_list

        program = flexutils.Plugin.getTensorflowProgram("train_flex_consensus.py", python=False)
        self.runJob(program, args, numberOfMpi=1)

    # --------------------------- UTILS functions --------------------------------------------

    # ----------------------- VALIDATE functions ----------------------------------------
