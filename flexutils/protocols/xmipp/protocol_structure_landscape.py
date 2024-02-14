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
from pyworkflow.object import CsvList
from pyworkflow.protocol import LEVEL_ADVANCED
from pyworkflow.protocol.params import PointerParam, EnumParam, IntParam, BooleanParam, FloatParam, StringParam, \
                                       GPU_LIST, USE_GPU

from pwem.protocols import ProtAnalysis3D, ProtFlexBase
from pwem.objects import ParticleFlex, SetOfParticlesFlex

import flexutils
import flexutils.constants as const


class XmippProtStructureLanscapes(ProtAnalysis3D, ProtFlexBase):
    """ Reduced structure based conformational landscape """

    _label = 'structure landscape'
    _devStatus = NEW
    OUTPUT_PREFIX = 'outputParticles'
    DIMENSIONS = [2, 3]
    SAVE = ["structures", "residuals"]

    # --------------------------- DEFINE param functions ----------------------
    def _defineParams(self, form):
        form.addSection(label='General parameters')
        form.addHidden(USE_GPU, BooleanParam, default=True,
                       condition="mode==2",
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
                      help="Particles must have a Zernike3D flexibility information associated")
        form.addParam('structure', PointerParam, label="Input structures",
                      pointerClass='AtomStruct',
                      help="This structure must be traced (or at least fitted) to the reference "
                           "map used to compute the input particles' Zernike3D coefficients.")
        form.addParam('save', EnumParam, choices=['Structures', 'Residuals'],
                      default=0, display=EnumParam.DISPLAY_HLIST,
                      label="Space based on: ",
                      help="\t * Structures : the real atomic coordinates for all the generated structures "
                           "will be used to get the new conformational space.\n"
                           "\t * Residuals: the deformation field associated with all the generated strucutres "
                           "will be used to get the new conformational landscape.\n")
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
        form.addParam('nb_umap', IntParam, label="UMAP neighbors",
                      default=5, condition="mode==0",
                      help="Number of neighbors to associate to each point in the space when computing "
                           "the UMAP space. The higher the number of neighbors, the more predominant "
                           "global in the original space features will be")
        form.addParam('epochs_umap', IntParam, label="Number of UMAP epochs",
                      default=1000, condition="mode==0",
                      help="Increasing the number of epochs will lead to more accurate UMAP spaces at the cost "
                           "of larger execution times")
        form.addParam('densmap_umap', BooleanParam, label="Compute DENSMAP?",
                      default=False, condition="mode==0",
                      help="DENSMAP will try to bring densities in the UMAP space closer to each other. Execution time "
                           "will increase when computing a DENSMAP")
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
        form.addParam('dimensions', EnumParam, choices=['2D', '3D'],
                      default=0, display=EnumParam.DISPLAY_HLIST,
                      label="Landscape space dimensions?",
                      help="Determine if the original landscape will be reduced to have "
                           "2 or 3 dimensions.")
        form.addParallelSection(threads=4, mpi=0)


    # --------------------------- INSERT steps functions ----------------------
    def _insertAllSteps(self):
        self._insertFunctionStep(self.computeAtomTheshold)
        self._insertFunctionStep(self.computeReducedSpace)
        self._insertFunctionStep(self.createOutputStep)

    def createOutputStep(self):
        file_coords = self._getExtraPath("red_coords.txt")
        red_space = np.loadtxt(file_coords)

        inputSet = self.particles.get()
        partSet = self._createSetOfParticlesFlex(progName=const.ZERNIKE3D)
        partSet.copyInfo(inputSet)
        partSet.setAlignmentProj()

        idx = 0
        for particle in inputSet.iterItems():
            outParticle = ParticleFlex(progName=const.ZERNIKE3D)
            outParticle.copyInfo(particle)

            outParticle.setZRed(red_space[idx])

            partSet.append(outParticle)

            idx += 1

        self._defineOutputs(outputParticles=partSet)
        self._defineTransformRelation(self.particles, partSet)

    # --------------------------- STEPS functions -----------------------------
    def computeAtomTheshold(self):
        particles = self.particles.get()
        structure = self.structure.get()
        l1 = particles.getFlexInfo().L1.get()
        l2 = particles.getFlexInfo().L2.get()

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
        args = "--structure %s --z_space %s --out_path %s --sr %f --bxsize %d --L1 %d " \
               "--L2 %d --thr %d" \
               % (structure.getFileName(), file_z_space, self._getExtraPath(),
                  particles.getSamplingRate(), particles.getFirstItem().getXDim(),
                  l1, l2, self.numberOfThreads.get())
        program = os.path.join(const.XMIPP_SCRIPTS, "structure_space.py")
        program = flexutils.Plugin.getProgram(program)
        self.runJob(program, args)

    def computeReducedSpace(self):
        particles = self.particles.get()
        structure = self.structure.get()
        l1 = particles.getFlexInfo().L1.get()
        l2 = particles.getFlexInfo().L2.get()

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
        mode = self.mode.get()
        if mode == 0:
            args = "--structure %s --z_space %s --out_path %s --sr %f --bxsize %d --L1 %d " \
                   "--L2 %d --thr %d --umap --n_neighbors %d --n_epochs %d --n_components %d " \
                   "--save %s" \
                   % (structure.getFileName(), file_z_space, self._getExtraPath(),
                      particles.getSamplingRate(), particles.getFirstItem().getXDim(),
                      l1, l2, self.numberOfThreads.get(), self.nb_umap.get(),
                      self.epochs_umap.get(), self.DIMENSIONS[self.dimensions.get()],
                      self.SAVE[self.save.get()])
            if self.densmap_umap.get():
                args += " --densmap"
            program = os.path.join(const.XMIPP_SCRIPTS, "structure_space.py")
            program = flexutils.Plugin.getProgram(program)
            self.runJob(program, args)
        elif mode == 1:
            args = "--structure %s --z_space %s --out_path %s --sr %f --bxsize %d --L1 %d " \
                   "--L2 %d --thr %d --pca --n_components %d --save %s" \
                   % (structure.getFileName(), file_z_space, self._getExtraPath(),
                      particles.getSamplingRate(), particles.getFirstItem().getXDim(),
                      l1, l2, self.numberOfThreads.get(), self.DIMENSIONS[self.dimensions.get()],
                      self.SAVE[self.save.get()])
            program = os.path.join(const.XMIPP_SCRIPTS, "structure_space.py")
            program = flexutils.Plugin.getProgram(program)
            self.runJob(program, args)
        elif mode == 2:
            args = "--space %s --output %s --split_train 1 --clusters %d --init_power %f " \
                   "--end_power %f --vae_sigma %f --lat_dim %d --loss_lambda %f" \
                    % (file_z_space, self._getExtraPath("red_coords.txt"), self.clusters.get(),
                       self.init_power.get(), self.end_power.get(), self.vae_sigma.get(),
                       self.DIMENSIONS[self.dimensions.get()], self.loss_lambda.get())

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

    # ----------------------- VALIDATE functions ----------------------------------------
    def validate(self):
        """ Try to find errors on define params. """
        errors = []
        particles = self.particles.get()
        if isinstance(particles, SetOfParticlesFlex):
            if particles.getFlexInfo().getProgName() != const.ZERNIKE3D:
                errors.append("The flexibility information associated with the particles is not "
                              "coming from the Zernike3D algorithm. Please, provide a set of particles "
                              "with the correct flexibility information.")
        return errors