# **************************************************************************
# *
# * Authors:     David Herreros Calero (dherreros@cnb.csic.es)
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
import numpy as np

from pyworkflow import NEW
from pyworkflow.object import CsvList, Boolean
from pyworkflow.protocol import LEVEL_ADVANCED
from pyworkflow.protocol.params import PointerParam, EnumParam, IntParam, BooleanParam, FloatParam, StringParam, \
                                       GPU_LIST, USE_GPU

from pwem.protocols import ProtAnalysis3D, ProtFlexBase
from pwem.objects import ParticleFlex

import flexutils
import flexutils.constants as const


class ProtFlexDimRedSpace(ProtAnalysis3D, ProtFlexBase):
    """ Dimensionality reduction of spaces based on different methods """

    _label = 'dimred space'
    _devStatus = NEW
    OUTPUT_PREFIX = 'outputParticles'

    # --------------------------- DEFINE param functions ----------------------
    def _defineParams(self, form):
        form.addSection(label='General parameters')
        form.addHidden(USE_GPU, BooleanParam, default=True,
                       condition="mode==0 or mode==2",
                       label="Use GPU for execution",
                       help="This protocol has both CPU and GPU implementation.\
                                     Select the one you want to use.")
        form.addHidden(GPU_LIST, StringParam, default='0',
                       expertLevel=LEVEL_ADVANCED,
                       condition="mode==2",
                       label="Choose GPU IDs",
                       help="Add a list of GPU devices that can be used")
        form.addParam('particles', PointerParam, label="Input particles",
                      pointerClass='SetOfParticlesFlex', important=True,
                      help="Particles must have a flexibility information associated (Zernike3D, CryoDrgn...")
        form.addParam('mode', EnumParam, choices=['UMAP', 'PCA', 'deepElastic'],
                      default=0, display=EnumParam.DISPLAY_HLIST,
                      label="Dimensionality reduction method",
                      help="\t * UMAP: usually leads to more meaningfull spaces, although execution "
                           "is higher\n"
                           "\t * PCA: faster but less meaningfull spaces \n"
                           "\t * deepElastic: Variational autencoder based dimred method. This method learns and "
                           "embedding that tries to keep as best as possible the internal clustering structure of "
                           "the original N-D space \n"
                           "UMAP, PCA, and cryoExplode are only computed the first time the are used. Afterwards, they "
                           "will be reused to increase performance.")
        form.addParam('epochs_umap', IntParam, label="Number of UMAP epochs",
                      default=10, condition="mode==0",
                      help="Increasing the number of epochs will lead to more accurate UMAP spaces at the cost "
                           "of larger execution times")
        form.addParam('clusters', IntParam, label="Initial number of clusters", default=10,
                      condition="mode==2",
                      expertLevel=LEVEL_ADVANCED,
                      help="The N-D space will be splitted in the number of cluster specified so the network "
                           "can learn the best way to keep their strucutre in the reduced space.")
        form.addParam('init_power', FloatParam, label="Initial explosion power", default=10.0,
                      condition="mode==2",
                      expertLevel=LEVEL_ADVANCED,
                      help="The initial power to scatter the landscape. This help the network to learn appropiately "
                           "the initial clustering")
        form.addParam('end_power', FloatParam, label="Final explosion power", default=1.0,
                      condition="mode==2",
                      expertLevel=LEVEL_ADVANCED,
                      help="The final power to scatter the landscape. This will determine how close the clusters will "
                           "be in the final embedding.")
        form.addParam('vae_sigma', FloatParam, label="Variational autoencoder sigma", default=1.0,
                      condition="mode==2",
                      expertLevel=LEVEL_ADVANCED,
                      help="Larger values of sigma will enfoce continuity of the final embedding. If set to zero, a "
                           "non-variational autoencoder will be trained.")
        form.addParam('loss_lambda', FloatParam, label="Cosine mapping lambda", default=1.0,
                      condition="mode==2",
                      expertLevel=LEVEL_ADVANCED,
                      help="If 0.0, cosine distance mapping will not be considered in the lost function. In general, "
                           "adding the cosine distance mapping to the cost function will lead to more discriminative "
                           "embeddings, at the expense of having possible visual artefacts. By default it is set to "
                           "1.0 consider it in the cost function.")
        form.addParam('dimensions', IntParam, default=3,
                      label="Reduced landscape space dimensions?",
                      help="Determine the number of dimensions the reduced landscape will have. It must be a value "
                           "larger than three and smaller or equal to the number of dimension in the original space.")
        form.addParallelSection(threads=4, mpi=0)


    # --------------------------- INSERT steps functions ----------------------
    def _insertAllSteps(self):
        self._insertFunctionStep(self.computeReducedSpace)
        self._insertFunctionStep(self.createOutputStep)

    def createOutputStep(self):
        file_coords = self._getExtraPath("red_coords.txt")
        red_space = np.loadtxt(file_coords)

        inputSet = self.particles.get()
        progName = inputSet.getFlexInfo().getProgName()
        partSet = self._createSetOfParticlesFlex(progName=progName)

        partSet.copyInfo(inputSet)
        partSet.setHasCTF(inputSet.hasCTF())
        partSet.setAlignmentProj()

        idx = 0
        for particle in inputSet.iterItems():
            outParticle = ParticleFlex(progName=progName)
            outParticle.copyInfo(particle)

            outParticle.setZRed(red_space[idx])

            partSet.append(outParticle)

            idx += 1

        if self.mode.get() == 0:
            partSet.getFlexInfo().setAttr("umap_weights", self._getExtraPath("trained_umap"))

        self._defineOutputs(outputParticles=partSet)
        self._defineTransformRelation(self.particles, partSet)

    # --------------------------- STEPS functions -----------------------------
    def computeReducedSpace(self):
        particles = self.particles.get()
        self.num_vol = 0

        # ********* Get Z space *********
        z_space = []
        for particle in particles.iterItems():
            z_space.append(particle.getZFlex())
        z_space = np.asarray(z_space)
        # ********************

        # Generate files to call command line
        file_z_space = self._getExtraPath("z_space.txt")
        np.savetxt(file_z_space, z_space)

        # Compute reduced space
        file_coords = self._getExtraPath("red_coords.txt")
        mode = self.mode.get()
        if mode == 0:
            args = "--input %s --umap --output %s --n_epochs %d --n_components %d " \
                   % (file_z_space, file_coords, self.epochs_umap.get(), self.dimensions.get())

            if self.useGpu.get():
                gpu_list = ','.join([str(elem) for elem in self.getGpuList()])
                args += " --gpu %s" % gpu_list

            program = os.path.join(const.XMIPP_SCRIPTS, "dimensionality_reduction.py")
            program = flexutils.Plugin.getProgram(program)
            self.runJob(program, args)
        elif mode == 1:
            args = "--input %s --pca --n_components %d --output %s" \
                   % (file_z_space, self.dimensions.get(), file_coords)
            program = os.path.join(const.XMIPP_SCRIPTS, "dimensionality_reduction.py")
            program = flexutils.Plugin.getProgram(program)
            self.runJob(program, args)
        elif mode == 2:
            args = "--space %s --output %s --split_train 1 --clusters %d --init_power %f " \
                   "--end_power %f --vae_sigma %f --lat_dim %d --loss_lambda %f" \
                    % (file_z_space, file_coords, self.clusters.get(), self.init_power.get(),
                       self.end_power.get(), self.vae_sigma.get(),
                       self.dimensions.get(), self.loss_lambda.get())

            if self.useGpu.get():
                gpu_list = ','.join([str(elem) for elem in self.getGpuList()])
                args += " --gpu %s" % gpu_list

            program = flexutils.Plugin.getTensorflowProgram("train_deep_elastic.py", python=False)
            self.runJob(program, args, numberOfMpi=1)

    # --------------------------- INFO functions -----------------------------
    def _summary(self):
        summary = []
        if self.getOutputsSize() >= 1:
            for _, outParticles in self.iterOutputAttributes():
                summary.append("Output *%s*:" % outParticles.getNameId().split('.')[1])
        else:
            summary.append("Output particles not ready yet")
        return summary

    def _methods(self):
        return [
            "Dimensionality reduction of spaces based on different methods",
        ]

    def _validate(self):
        errors = []

        # Check number of reduced dimensions
        particles = self.particles.get()
        original_dimensions = len(particles.getFirstItem().getZFlex())
        dimensions = self.dimensions.get()
        if dimensions < 3:
            errors.append(f"The number of dimensions in the reduced space must be larger than 3. Currently "
                          f"it is set to {dimensions}")
        if dimensions > original_dimensions:
            errors.append(f"The number of dimensions in the reduced space must be smalller than or equal to "
                          f"the number of dimensions in the original space (original dimensions "
                          f"= {original_dimensions}. Currently set to {dimensions}."
                          "it is set to {dimensions}")

        return errors

