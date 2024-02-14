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
from pyworkflow.object import Boolean
from pyworkflow.utils.path import makePath
from pyworkflow import VERSION_2_0

from pwem.protocols import ProtAnalysis3D, ProtFlexBase
from pwem.objects import ParticleFlex, SetOfParticlesFlex

import flexutils
from flexutils.utils import getOutputSuffix


class TensorflowProtInteractiveFlexConsensus(ProtAnalysis3D, ProtFlexBase):
    """ Protocol to filter particles based on a FlexConsensus network interactively """
    _label = 'interactive consensus - FlexConsensus'
    _lastUpdateVersion = VERSION_2_0
    OUTPUT_PREFIX = 'consensusParticles'

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
        group.addParam('inputSet', params.PointerParam,
                       label="Input particles", pointerClass='SetOfParticlesFlex')
        group.addParam('flexConsensusProtocol', params.PointerParam, label="FlexConsensus trained network",
                       pointerClass='TensorflowProtTrainFlexConsensus',
                       help="Previously executed 'train - FlexConsensus'. "
                            "This will allow to load the network trained in that protocol to be used during "
                            "the prediction")
        form.addParallelSection(threads=4, mpi=0)

    # --------------------------- INSERT steps functions --------------------------------------------
    def _insertAllSteps(self):
        self._insertFunctionStep(self.convertInputStep)
        self._insertFunctionStep(self.predictStep)

    # --------------------------- STEPS functions ---------------------------------------------------
    def convertInputStep(self):
        particles = self.inputSet.get()
        data_path = self._getExtraPath("data")
        if not os.path.isdir(data_path):
            makePath(data_path)

        # Get data filename
        progName = particles.getFlexInfo().getProgName()
        data_file = progName + "_1.txt"

        # Read flexible space form particles
        z_flex = []
        for particle in particles.iterItems():
            z_flex.append(particle.getZFlex())
        z_flex = np.vstack(z_flex)

        # Save flexible space
        np.savetxt(os.path.join(data_path, data_file), z_flex)

    def predictStep(self):
        flexConsensusProtocol = self.flexConsensusProtocol.get()
        data_path = self._getExtraPath("data")
        out_path = self._getExtraPath()
        lat_dim = flexConsensusProtocol.latDim.get()
        weigths_file = flexConsensusProtocol._getExtraPath(os.path.join('network', 'flex_consensus_model.h5'))
        args = "--data_path %s --out_path %s --weigths_file %s --lat_dim %d" \
               % (data_path, out_path, weigths_file, lat_dim)

        if self.useGpu.get():
            gpu_list = ','.join([str(elem) for elem in self.getGpuList()])
            args += " --gpu %s" % gpu_list

        program = flexutils.Plugin.getTensorflowProgram("predict_flex_consensus.py", python=False)
        self.runJob(program, args, numberOfMpi=1)

    def _createOutput(self):
        inputSet = self.inputSet.get()
        selected_idx = np.loadtxt(self._getExtraPath("selected_idx.txt"))

        suffix = getOutputSuffix(self, SetOfParticlesFlex)
        partSet = self._createSetOfParticlesFlex(suffix, progName=inputSet.getFlexInfo().getProgName())

        partSet.copyInfo(inputSet)
        partSet.setHasCTF(inputSet.hasCTF())
        partSet.setAlignmentProj()

        idx = 0
        for particle in inputSet.iterItems():
            if idx in selected_idx:
                outParticle = ParticleFlex(progName=inputSet.getFlexInfo().getProgName())
                outParticle.copyInfo(particle)

                partSet.append(outParticle)

            idx += 1

        # Save new output
        name = self.OUTPUT_PREFIX + suffix
        args = {}
        args[name] = partSet
        self._defineOutputs(**args)
        self._defineSourceRelation(self.inputSet, partSet)

    # --------------------------- UTILS functions --------------------------------------------

    # ----------------------- VALIDATE functions ----------------------------------------
    def validate(self):
        """ Try to find errors on define params. """
        errors = []

        flexConsensusSets = self.flexConsensusProtocol.get().inputSets
        in_lat_dim = self.inputSet.get().getFirstItem().getZFlex().size
        dim_match = False

        for particle_set in flexConsensusSets:
            lat_dim = particle_set.get().getFirstItem().getZFlex().size
            if lat_dim == in_lat_dim:
                dim_match = True
                break

        if not dim_match:
            errors.append("The input particles' flexible information does not match the flexible space "
                          "dimension of the provided FlexConsensus network. Please, provide a set of particles "
                          "computed with some of the following programs:\n")
            progNames = []
            for particle_set in flexConsensusSets:
                progName = particle_set.get().getFlexInfo().getProgName()
                if progName not in progNames:
                    progNames.append(progName)
                    errors.append(f"     -{progName}")

        return errors
