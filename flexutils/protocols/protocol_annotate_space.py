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
import re
import shutil

import numpy as np
from glob import glob
from sklearn.neighbors import KDTree
from xmipp_metadata.image_handler import ImageHandler

from pyworkflow import NEW
from pyworkflow.protocol import LEVEL_ADVANCED
from pyworkflow.protocol.params import (PointerParam, IntParam, MultiPointerParam, BooleanParam, StringParam,
                                        USE_GPU, GPU_LIST)
import pyworkflow.utils as pwutils
from pyworkflow.utils.properties import Message
from pyworkflow.gui.dialog import askYesNo
from pyworkflow.object import Boolean, Integer

from pwem.protocols import ProtAnalysis3D, ProtFlexBase
from pwem.objects import SetOfVolumesFlex, VolumeFlex

import flexutils
from flexutils.utils import getOutputSuffix, computeNormRows
import flexutils.constants as const

import xmipp3


class ProtFlexAnnotateSpace(ProtAnalysis3D, ProtFlexBase):
    """ Interactive annotation of conformational spaces """

    _label = 'annotate space'
    _devStatus = NEW
    OUTPUT_PREFIX = 'flexible3DClasses'

    # --------------------------- DEFINE param functions ----------------------
    def _defineParams(self, form):
        form.addSection(label='General parameters')
        form.addHidden(USE_GPU, BooleanParam, default=True,
                       label="Use GPU for execution",
                       help="This protocol has both CPU and GPU implementation.\
                                     Select the one you want to use.")
        form.addHidden(GPU_LIST, StringParam, default='0',
                       expertLevel=LEVEL_ADVANCED,
                       label="Choose GPU IDs",
                       help="Add a list of GPU devices that can be used")
        form.addParam('particles', PointerParam, label="Particles to annotate",
                      pointerClass='SetOfParticlesFlex', important=True,
                      help="Particles must have a flexibility information associated (Zernike3D, CryoDrgn...")
        form.addParam('priors', MultiPointerParam, label="Priors", allowsNull=True,
                      pointerClass="SetOfVolumesFlex, VolumeFlex",
                      condition="particles and particles.getFlexInfo().getProgName() == 'Zernike3D'",
                      help='Volumes with Zernike3D coefficients associated (computed using '
                           '"Refernce map" as reference) to add as prior information to the Zernike3D '
                           'space')
        form.addParam('boxSize', IntParam, label="Box size",
                      condition="particles and particles.getFlexInfo().getProgName() == 'CryoDRGN'",
                      help="Volumes generated from the CryoDrgn network will be resampled to the "
                           "chosen box size (only for the visualization).")
        form.addParam('neighbors', IntParam, label="Number of particles to associate to selections",
                      default=5000, expertLevel=LEVEL_ADVANCED)



    # --------------------------- INSERT steps functions ----------------------
    def _insertAllSteps(self):
        self._insertFunctionStep(self.launchVolumeSlicer, interactive=True)

    def _createOutput(self):
        particles = self.particles.get()
        partIds = list(particles.getIdSet())
        neighbors = self.neighbors.get()
        num_part = particles.getSize()
        sr = particles.getSamplingRate()
        progName = particles.getFlexInfo().getProgName()

        # Get right imports
        if progName == const.NMA:
            createFn = self._createSetOfClassesStructFlex
            from pwem.objects import ClassStructFlex as Class
            from pwem.objects import AtomStructFlex as Rep
            from pwem.objects import SetOfClassesStructFlex as SetOfClasses
        else:
            createFn = self._createSetOfClassesFlex
            from pwem.objects import ClassFlex as Class
            from pwem.objects import VolumeFlex as Rep
            from pwem.objects import SetOfClassesFlex as SetOfClasses

        # Create SetOfFlexClasses
        suffix = getOutputSuffix(self, SetOfClasses)
        flexClasses = createFn(self.particles, suffix, progName=progName)

        # Folder to save decoded volumes
        suffix_int = int(suffix)
        save_volume_path = self._getExtraPath(os.path.join(f"Output_Volumes_{suffix_int}", "class_{:d}.mrc"))
        if not os.path.isdir(self._getExtraPath(f"Output_Volumes_{suffix_int}")):
            os.mkdir(self._getExtraPath(f"Output_Volumes_{suffix_int}"))

        # ****** Generate representative volumes *******
        z_rep = []
        for file in glob(self._getExtraPath('saved_selections*')):
            with open(file) as f:
                line = f.readline()
                z_rep.append(np.fromstring(line, dtype=float, sep=' '))
        z_rep = np.stack(z_rep)
        z_rep = z_rep if z_rep.ndim == 2 else z_rep[None, ...]

        if particles.getFlexInfo().getProgName() == const.ZERNIKE3D:
            reference = particles.getFlexInfo().refMap.get()
            representatives_paths = [reference for _ in range(z_rep.shape[0])]

        elif particles.getFlexInfo().getProgName() == const.CRYODRGN:
            from cryodrgn.utils import generateVolumes
            representatives_paths = []
            generateVolumes(z_rep, particles.getFlexInfo()._cryodrgnWeights.get(),
                            particles.getFlexInfo()._cryodrgnConfig.get(), self._getExtraPath(),
                            downsample=self.boxSize.get(), apix=particles.getSamplingRate())
            for idx in range(z_rep.shape[0]):
                ImageHandler().scaleSplines(self._getExtraPath('vol_{:03d}.mrc'.format(idx)),
                                            save_volume_path.format(idx),
                                            finalDimension=particles.getXDim(), overwrite=True)
                representatives_paths.append(save_volume_path.format(idx))

        elif particles.getFlexInfo().getProgName() == const.HETSIREN:
            from flexutils.utils import generateVolumesHetSIREN
            representatives_paths = []
            gpu_ids = ','.join([str(elem) for elem in self.getGpuList()])
            generateVolumesHetSIREN(particles.getFlexInfo().modelPath.get(), z_rep,
                                    self._getExtraPath(), step=particles.getFlexInfo().coordStep.get(),
                                    architecture=particles.getFlexInfo().architecture.get(),
                                    disPose=particles.getFlexInfo().disPose.get(),
                                    disCTF=particles.getFlexInfo().disCTF.get(), gpu=gpu_ids)
            for idx in range(z_rep.shape[0]):
                ImageHandler().scaleSplines(self._getExtraPath('decoded_map_class_{:02d}.mrc'.format(idx + 1)),
                                            save_volume_path.format(idx),
                                            finalDimension=particles.getXDim(), overwrite=True)
                representatives_paths.append(save_volume_path.format(idx))

        elif particles.getFlexInfo().getProgName() == const.FLEXSIREN:
            from flexutils.utils import generateVolumesFlexSIREN
            representatives_paths = []
            gpu_ids = ','.join([str(elem) for elem in self.getGpuList()])
            generateVolumesFlexSIREN(particles.getFlexInfo().modelPath.get(), z_rep,
                                     self._getExtraPath(), step=1,
                                     architecture=particles.getFlexInfo().architecture.get(), gpu=gpu_ids)
            for idx in range(z_rep.shape[0]):
                ImageHandler().scaleSplines(self._getExtraPath('decoded_map_class_{:02d}.mrc'.format(idx + 1)),
                                            save_volume_path.format(idx),
                                            finalDimension=particles.getXDim(), overwrite=True)
                representatives_paths.append(save_volume_path.format(idx))

        elif particles.getFlexInfo().getProgName() == const.NMA:
            reference = particles.getFlexInfo().refStruct.get()
            subset = particles.getFlexInfo().atomSubset
            representatives_paths = [reference for _ in range(z_rep.shape[0])]

        elif particles.getFlexInfo().getProgName() == const.CRYOSPARCFLEX:
            import cryosparc2
            from cryosparc2.utils import generateFlexVolumes
            representatives_paths = []
            csGPU = self.getGpuList()[0] if self.usesGpu() else 0
            flexGeneratorJob = generateFlexVolumes(z_rep,
                                                   particles.getFlexInfo().getAttr("projectId"),
                                                   particles.getFlexInfo().getAttr("workSpaceId"),
                                                   particles.getFlexInfo().getAttr("trainJobId"),
                                                   gpu=csGPU)
            flexGeneratorJob = str(flexGeneratorJob.get())
            for idx in range(z_rep.shape[0]):
                volume_path = os.path.join(particles.getFlexInfo().getAttr("projectPath"), flexGeneratorJob,
                                           flexGeneratorJob + "_series_000",
                                           flexGeneratorJob + "_series_000_frame_{:03d}.mrc".format(idx))
                ImageHandler().scaleSplines(volume_path, save_volume_path.format(idx),
                                            finalDimension=particles.getXDim(), overwrite=True)
                representatives_paths.append(save_volume_path.format(idx))

        # Read selected coefficients
        clInx = 1
        for file in glob(self._getExtraPath('saved_selections*')):
            z_space_vw = []
            with open(file) as f:
                lines = f.readlines()
                for line in lines:
                    z_space_vw.append(np.fromstring(line, dtype=float, sep=' '))
            z_space_vw = np.asarray(z_space_vw)
            # z_space_vw = np.asarray(z_space_vw[self.num_vol:])

            if z_space_vw.ndim < 2:
                z_space_vw = z_space_vw[None, ...]

            if "_cluster" in file:
                # Read space
                z_space = z_space_vw[1:]
                z_space_vw = z_space_vw[0][None, ...]
            else:
                # Read space
                z_space = np.loadtxt(self._getExtraPath("z_space.txt"))

            # Create KDTree
            kdtree = KDTree(np.loadtxt(self._getExtraPath("z_space.txt")))

            # Populate SetOfClasses3D with KMean particles
            for z_idx in range(z_space_vw.shape[0]):
                if "_cluster" in file:
                    if z_space.shape[0] > 0:
                        _, currIds = kdtree.query(z_space, k=1)
                        currIds = np.squeeze(np.asarray(currIds)).astype(int)
                    else:
                        currIds = np.array([])
                else:
                    _, currIds = kdtree.query(z_space_vw[z_idx].reshape(1, -1), k=neighbors + 10)
                    currIds = currIds[0]

                if currIds.ndim and currIds.size:
                    newClass = Class()
                    newClass.copyInfo(particles)
                    newClass.setHasCTF(particles.hasCTF())
                    newClass.setAcquisition(particles.getAcquisition())
                    representative = Rep(progName=progName)
                    if hasattr(representative, "setSamplingRate"):
                        representative.setSamplingRate(sr)

                    # ****** Fill representative information *******
                    if particles.getFlexInfo().getProgName() == const.ZERNIKE3D:
                        representative.setLocation(representatives_paths[clInx - 1])

                    elif particles.getFlexInfo().getProgName() == const.CRYODRGN:
                        representative.setLocation(representatives_paths[clInx - 1])

                    elif particles.getFlexInfo().getProgName() == const.HETSIREN:
                        representative.setLocation(representatives_paths[clInx - 1])

                    elif particles.getFlexInfo().getProgName() == const.FLEXSIREN:
                        representative.setLocation(representatives_paths[clInx - 1])

                    elif particles.getFlexInfo().getProgName() == const.NMA:
                        representative.getFlexInfo().atomSubset = subset
                        representative.setLocation(representatives_paths[clInx - 1])

                    elif particles.getFlexInfo().getProgName() == const.CRYOSPARCFLEX:
                        representative.setLocation(representatives_paths[clInx - 1])

                    representative.setZFlex(z_space_vw[z_idx])
                    representative.getFlexInfo().copyInfo(particles.getFlexInfo())
                    # ********************

                    newClass.setRepresentative(representative)

                    flexClasses.append(newClass)

                    enabledClass = flexClasses[newClass.getObjId()]
                    enabledClass.enableAppend()


                    if "_cluster" in file:
                        for itemId in currIds:
                            item = particles[partIds[itemId]]
                            item._xmipp_subtomo_labels = Integer(clInx)
                            enabledClass.append(item)
                    else:
                        for idx in range(neighbors):
                            itemId = currIds[idx]
                            while itemId >= num_part:
                                currIds = np.delete(currIds, idx)
                                itemId = currIds[idx]
                            item = particles[partIds[itemId]]
                            item._xmipp_subtomo_labels = Integer(clInx)
                            enabledClass.append(item)

                    flexClasses.update(enabledClass)
                    clInx += 1

        # Save new output
        name = self.OUTPUT_PREFIX + "_" + suffix
        args = {}
        args[name] = flexClasses
        self._defineOutputs(**args)
        self._defineSourceRelation(particles, flexClasses)

    # --------------------------- STEPS functions -----------------------------
    def launchVolumeSlicer(self):
        particles = self.particles.get()
        self.num_vol = 0

        # ********* Get Z space *********
        z_space = []
        for particle in particles.iterItems():
            z_space.append(particle.getZFlex())
        z_space = np.asarray(z_space)

        if particles.getFlexInfo().getProgName() == const.ZERNIKE3D:
            reference = particles.getFlexInfo().refMap.get()
            mask = particles.getFlexInfo().refMask.get()
            volumes = self.priors

            # Copy original reference and mask to extra
            ih = ImageHandler()
            ih.convert(reference, self._getExtraPath("reference_original.mrc"), overwrite=True)
            ih.convert(mask, self._getExtraPath("mask_reference_original.mrc"), overwrite=True)

            # Resize reference map to increase real time conformation inspection performance
            inputFile = reference
            outFile = self._getExtraPath('reference.mrc')
            if not os.path.isfile(outFile):
                if pwutils.getExt(inputFile) == ".mrc":
                    inputFile += ":mrc"
                self.runJob("xmipp_image_resize",
                            "-i %s -o %s --dim %d " % (inputFile,
                                                       outFile,
                                                       64), numberOfMpi=1, env=xmipp3.Plugin.getEnviron())

            # Resize mask
            inputFile = mask
            outFile = self._getExtraPath('mask.mrc')
            if not os.path.isfile(outFile):
                if pwutils.getExt(inputFile) == ".mrc":
                    inputFile += ":mrc"
                self.runJob("xmipp_image_resize",
                            "-i %s -o %s --dim %d " % (inputFile,
                                                       outFile,
                                                       64), numberOfMpi=1, env=xmipp3.Plugin.getEnviron())
                self.runJob("xmipp_transform_threshold",
                            "-i %s -o %s --select below 0.01 "
                            "--substitute binarize " % (outFile, outFile), numberOfMpi=1, env=xmipp3.Plugin.getEnviron())

            z_space_vol = []
            if volumes:
                for pointer in volumes:
                    item = pointer.get()
                    if isinstance(item, VolumeFlex):
                        z_space_vol.append(item.getZFlex())
                    elif isinstance(item, SetOfVolumesFlex):
                        for volume in item.iterItems():
                            z_space_vol.append(volume.getZFlex())
            z_space_vol = np.asarray(z_space_vol)
            file_z_vol = self._getExtraPath("z_space_vol.txt")
            np.savetxt(file_z_vol, z_space_vol)

            # Resize coefficients
            # z_space = (64 / ImageHandler().read(reference).getDimensions()[0]) * z_space
        # ********************

        # Generate files to call command line
        file_z_space = self._getExtraPath("z_space.txt")
        file_interp_val = self._getExtraPath("interp_val.txt")
        np.savetxt(file_z_space, z_space)

        # Compute/Read UMAP or PCA
        file_coords = self._getExtraPath("red_coords.txt")
        red_space = []
        for particle in particles.iterItems():
            red_space.append(particle.getZRed())
        red_space = np.asarray(red_space)
        np.savetxt(file_coords, red_space)

        # ********* Get interpolation value for coloring the space *********
        if particles.getFlexInfo().getProgName() == const.ZERNIKE3D:
            interp_val = computeNormRows(z_space)
        else:
            interp_val = np.ones([1, z_space.shape[0]])
        # *********

        # Generate files to call command line
        np.savetxt(file_interp_val, interp_val)
        path = os.path.abspath(self._getExtraPath())

        # ********* Run viewer *********
        needsPackages = None

        args = "--data %s --z_space %s --interp_val %s --path %s --sr %f " \
                %(file_coords, file_z_space, file_interp_val, path, particles.getSamplingRate())

        if self.usesGpu():
            args += "--useGPU %s " % (','.join([str(elem) for elem in self.getGpuList()]))

        if particles.getFlexInfo().getProgName() == const.ZERNIKE3D:
            L1 = particles.getFlexInfo().L1.get()
            L2 = particles.getFlexInfo().L2.get()
            args += "--L1 %d --L2 %d --boxsize 64 --mode Zernike3D" \
                   % (L1, L2)
            if volumes:
                args += "--z_space_vol %s" % file_z_vol

        elif particles.getFlexInfo().getProgName() == const.CRYODRGN:
            import cryodrgn
            args += "--weights %s --config %s --boxsize %d --mode CryoDrgn --env_name %s" \
                   % (particles.getFlexInfo()._cryodrgnWeights.get(),
                      particles.getFlexInfo()._cryodrgnConfig.get(), self.boxSize.get(),
                      cryodrgn.Plugin.getCryoDrgnEnvActivation().split(" ")[-1])

        elif particles.getFlexInfo().getProgName() == const.HETSIREN:
            args += "--weights %s --step %d --architecture %s --mode HetSIREN --env_name flexutils-tensorflow" \
                   % (particles.getFlexInfo().modelPath.get(),
                      particles.getFlexInfo().coordStep.get(),
                      particles.getFlexInfo().architecture.get())

            if particles.getFlexInfo().disPose.get():
                args += " --pose_reg 1.0"
            else:
                args += " --pose_reg 0.0"

            if particles.getFlexInfo().disCTF.get():
                args += " --ctf_reg 1.0"
            else:
                args += " --ctf_reg 0.0"

        elif particles.getFlexInfo().getProgName() == const.FLEXSIREN:
            args += "--weights %s --architecture %s --mode FlexSIREN --env_name flexutils-tensorflow" \
                   % (particles.getFlexInfo().modelPath.get(),
                      particles.getFlexInfo().architecture.get())

        elif particles.getFlexInfo().getProgName() == const.NMA:
            args += "--weights %s --boxsize %d --mode NMA --env_name flexutils-tensorflow" \
                   % (particles.getFlexInfo().modelPath.get(), particles.getXDim())

        elif particles.getFlexInfo().getProgName() == const.CRYOSPARCFLEX:
            args += ("--projectId %s --workSpaceId %s --trainJobId %s --projectPath %s --mode 3DFlex "
                     "--env_name scipion3") \
                   % (particles.getFlexInfo().getAttr("projectId"),
                      particles.getFlexInfo().getAttr("workSpaceId"),
                      particles.getFlexInfo().getAttr("trainJobId"),
                      particles.getFlexInfo().getAttr("projectPath"))

        if hasattr(particles.getFlexInfo(), "umap_weights"):
            args += " --reduce umap --umap_weights %s" % particles.getFlexInfo().getAttr("umap_weights")
        else:
            args += " --reduce pca"

        env = pwutils.Environ(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = ''
        env["NAPARI_ASYNC"] = "1"

        program = "viewer_interactive_3d.py"
        program = flexutils.Plugin.getProgram(program, needsPackages=needsPackages, chimera=True)
        self.runJob(program, args, env=env)

        # *********

        if len(glob(self._getExtraPath("saved_selections*"))) > 0 and \
           askYesNo(Message.TITLE_SAVE_OUTPUT, Message.LABEL_SAVE_OUTPUT, None):
            self._createOutput()

    # --------------------------- OUTPUT functions -----------------------------
    def deleteOutput(self, output):
        attrName = self.findAttributeName(output)
        output_id = attrName.split("_")[-1]
        volumes_path = self._getExtraPath(f"Output_Volumes_{output_id}")
        shutil.rmtree(volumes_path)
        super().deleteOutput(output)

    def allowsDelete(self, obj):
        return True

    # --------------------------- INFO functions -----------------------------
    def _summary(self):
        summary = []
        if self.getOutputsSize() >= 1:
            for _, outClasses in self.iterOutputAttributes():
                summary.append("Output *%s*:" % outClasses.getNameId().split('.')[1])
                summary.append("    * Total annotated classes: *%s*" % outClasses.getSize())
        else:
            summary.append("Output annotated classes not ready yet")
        return summary

    def _methods(self):
        return [
            "Interactive annotation of conformational spaces",
        ]

    # ----------------------- VALIDATE functions ----------------------------------------
    def validate(self):
        """ Try to find errors on define params. """
        errors = []
        particles = self.particles.get()
        if particles.getFirstItem().getZRed().size == 0:
            errors.append("Particles do not have any dimensionality reduced version of the flexibility information "
                          "they store. Please, used the dimensionality reduction protocol available "
                          "in Flexutils Plugin to generate a valid set of particles")
        if particles.getSize() < self.neighbors.get():
            errors.append("Number of particles to be associated with each selected state is larger than the "
                          "total number of particles in the dataset. Please, provide a smaller value "
                          "(Advanced parameter)")

        # Check CryoDRGN boxsize parameter is set as it is mandatory
        if particles.getFlexInfo().getProgName() == 'CryoDRGN':
            if self.boxSize.get() is None:
                errors.append("Boxsize parameter needs to be set to an integer value smaller than or equal "
                              "to the boxsize used internally to train the CryoDRGN network")
            elif self.boxSize.get() % 2 != 0:
                errors.append("Boxsize parameter needs to be an even value")

        return errors
