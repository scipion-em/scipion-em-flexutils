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
from xmipp_metadata.image_handler import ImageHandler

from pyworkflow import BETA
from pyworkflow.protocol.params import PointerParam, IntParam, MultiPointerParam
import pyworkflow.utils as pwutils
from pyworkflow.utils.properties import Message
from pyworkflow.gui.dialog import askYesNo

from pwem.protocols import ProtAnalysis3D

import flexutils
from flexutils.utils import getOutputSuffix, computeNormRows
from flexutils.protocols import ProtFlexBase
from flexutils.objects import SetOfVolumesFlex, VolumeFlex
import flexutils.constants as const

import xmipp3


class ProtFlexClusterSpace(ProtAnalysis3D, ProtFlexBase):
    """ Interactive clustering of conformational spaces """

    _label = 'cluster space'
    _devStatus = BETA
    OUTPUT_PREFIX = 'flexible3DClasses'

    # --------------------------- DEFINE param functions ----------------------
    def _defineParams(self, form):
        form.addSection(label='General parameters')
        form.addParam('particles', PointerParam, label="Particles to annotate",
                      pointerClass='SetOfParticlesFlex', important=True,
                      help="Particles must have a flexibility information associated (Zernike3D, CryoDrgn...")
        form.addHidden('volumes', PointerParam, label="Priors", allowsNull=True,
                       pointerClass="SetOfVolumesFlex, VolumeFlex",
                       condition="particles and particles.getFlexInfo().getProgName() == 'Zernike3D'",
                       help='A set of volumes with Zernike3D coefficients associated (computed using '
                            '"Reference map" as reference) to add as prior information to the Zernike3D '
                            'space')
        form.addParam('priors', MultiPointerParam, label="Priors", allowsNull=True,
                      pointerClass="SetOfVolumes, Volume",
                      condition="particles and particles.getFlexInfo().getProgName() == 'Zernike3D'",
                      help='Volumes with Zernike3D coefficients associated (computed using '
                           '"Refernce map" as reference) to add as prior information to the Zernike3D '
                           'space')
        form.addParam('boxSize', IntParam, label="Box size",
                      condition="particles and particles.getFlexInfo().getProgName() == 'CryoDRGN'",
                      help="Volumes generated from the CryoDrgn network will be resampled to the "
                           "chosen box size (only for the visualization).")

    # --------------------------- INSERT steps functions ----------------------
    def _insertAllSteps(self):
        self._insertFunctionStep(self.launchInteractiveClustering, interactive=True)

    def _createOutput(self):
        particles = self.particles.get()
        partIds = list(particles.getIdSet())
        sr = particles.getSamplingRate()
        progName = particles.getFlexInfo().getProgName()

        # Get right imports
        if progName == const.NMA:
            createFn = self._createSetOfClassesStructFlex
            from flexutils.objects import ClassStructFlex as Class
            from flexutils.objects import AtomStructFlex as Rep
            from flexutils.objects import SetOfClassesStructFlex as SetOfClasses
        else:
            createFn = self._createSetOfClassesFlex
            from flexutils.objects import ClassFlex as Class
            from flexutils.objects import VolumeFlex as Rep
            from flexutils.objects import SetOfClassesFlex as SetOfClasses

        # Read KMean coefficients
        z_space_vw = []
        with open(self._getExtraPath('saved_selections.txt')) as f:
            lines = f.readlines()
            for line in lines:
                z_space_vw.append(np.fromstring(line, dtype=float, sep=' '))
        z_space_vw = np.asarray(z_space_vw[self.num_vol:])

        # Read KMean labels
        km_labels = np.loadtxt(self._getExtraPath('kmean_labels.txt'), delimiter=" ")
        if self.num_vol > 0:
            km_labels = km_labels[:-self.num_vol]

        # Create SetOfClasses3D
        suffix = getOutputSuffix(self, SetOfClasses)
        flexClasses = createFn(particles, suffix, progName=progName)

        # Popoulate SetOfClasses3D with KMean particles
        for clInx in range(z_space_vw.shape[0]):
            currIds = np.where(km_labels == clInx)[0]

            newClass = Class()
            newClass.copyInfo(particles)
            newClass.setAcquisition(particles.getAcquisition())
            representative = Rep(progName=progName)

            # ****** Fill representative information *******
            if particles.getFlexInfo().getProgName() == const.ZERNIKE3D:
                reference = particles.getFlexInfo().refMap.get()

                # Resize coefficients
                factor = (ImageHandler().read(reference).getDimensions()[0] / 64)
                z_space_vw[clInx] *= factor

                representative.setLocation(reference)

            elif particles.getFlexInfo().getProgName() == const.CRYODRGN:
                from cryodrgn.utils import generateVolumes
                generateVolumes(z_space_vw[clInx], particles.getFlexInfo()._cryodrgnWeights.get(),
                                particles.getFlexInfo()._cryodrgnConfig.get(), self._getExtraPath(),
                                downsample=self.boxSize.get(), apix=particles.getSamplingRate())
                ImageHandler().scaleSplines(self._getExtraPath('vol_000.mrc'),
                                            self._getExtraPath('class_%d.mrc') % clInx,
                                            finalDimension=particles.getXDim(), overwrite=True)
                representative.setLocation(self._getExtraPath('class_%d.mrc') % clInx)

            elif particles.getFlexInfo().getProgName() == const.HETSIREN:
                from flexutils.utils import generateVolumesHetSIREN
                generateVolumesHetSIREN(particles.getFlexInfo().modelPath.get(), z_space_vw[clInx],
                                        self._getExtraPath(), step=particles.getFlexInfo().coordStep.get())
                ImageHandler().scaleSplines(self._getExtraPath('decoded_map_class_1.mrc'),
                                            self._getExtraPath('class_%d.mrc') % clInx,
                                            finalDimension=particles.getXDim(), overwrite=True)
                representative.setLocation(self._getExtraPath('class_%d.mrc') % clInx)

            elif particles.getFlexInfo().getProgName() == const.NMA:
                reference = particles.getFlexInfo().refStruct.get()
                subset = particles.getFlexInfo().atomSubset

                representative.getFlexInfo().atomSubset = subset
                representative.setLocation(reference)

            representative.setZFlex(z_space_vw[clInx])
            representative.getFlexInfo().copyInfo(particles.getFlexInfo())
            # ********************

            if hasattr(representative, "setSamplingRate"):
                representative.setSamplingRate(sr)
            newClass.setRepresentative(representative)

            flexClasses.append(newClass)

            enabledClass = flexClasses[newClass.getObjId()]
            enabledClass.enableAppend()
            for itemId in currIds:
                item = particles[partIds[itemId]]
                enabledClass.append(item)

            flexClasses.update(enabledClass)

        # Save new output
        name = self.OUTPUT_PREFIX + suffix
        args = {}
        args[name] = flexClasses
        self._defineOutputs(**args)
        self._defineSourceRelation(particles, flexClasses)

    # --------------------------- STEPS functions -----------------------------
    def launchInteractiveClustering(self):
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
                            "--substitute binarize " % (outFile, outFile), numberOfMpi=1,
                            env=xmipp3.Plugin.getEnviron())

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

            # Get useful parameters
            self.num_vol = z_space_vol.shape[0]
            if self.num_vol > 0:
                z_space = np.vstack([z_space, z_space_vol])

            # Resize coefficients
            z_space = (64 / ImageHandler().read(reference).getDimensions()[0]) * z_space
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
        if not os.path.isfile(file_interp_val):
            np.savetxt(file_interp_val, interp_val)
        path = self._getExtraPath()

        # ********* Run viewer *********
        needsPackages = None

        if particles.getFlexInfo().getProgName() == const.ZERNIKE3D:
            L1 = particles.getFlexInfo().L1.get()
            L2 = particles.getFlexInfo().L2.get()
            args = "--data %s --z_space %s --interp_val %s --path %s " \
                   "--L1 %d --L2 %d --n_vol %d --mode Zernike3D" \
                   % (file_coords, file_z_space, file_interp_val, path, L1, L2,
                      self.num_vol)

        elif particles.getFlexInfo().getProgName() == const.CRYODRGN:
            needsPackages = [const.CRYODRGN, ]
            args = "--data %s --z_space %s --interp_val %s --path %s " \
                   "--weights %s --config %s --boxsize %d --sr %f --mode CryoDrgn" \
                   % (file_coords, file_z_space, file_interp_val, path,
                      particles.getFlexInfo()._cryodrgnWeights.get(),
                      particles.getFlexInfo()._cryodrgnConfig.get(), self.boxSize.get(),
                      particles.getSamplingRate())

        elif particles.getFlexInfo().getProgName() == const.HETSIREN:
            args = "--data %s --z_space %s --interp_val %s --path %s " \
                   "--weights %s --step %d --sr %f --mode HetSIREN" \
                   % (file_coords, file_z_space, file_interp_val, path,
                      particles.getFlexInfo().modelPath.get(),
                      particles.getFlexInfo().coordStep.get(),
                      particles.getSamplingRate())

        elif particles.getFlexInfo().getProgName() == const.NMA:
            args = "--data %s --z_space %s --interp_val %s --path %s " \
                   "--weights %s --sr %f --mode NMA" \
                   % (file_coords, file_z_space, file_interp_val, path,
                      particles.getFlexInfo().modelPath.get(),
                      particles.getSamplingRate())

        program = os.path.join(const.VIEWERS, "annotation_3d_tools", "viewer_3d_pc.py")
        program = flexutils.Plugin.getProgram(program, needsPackages=needsPackages)
        self.runJob(program, args)

        # *********

        if os.path.isfile(self._getExtraPath("saved_selections.txt")) and \
           askYesNo(Message.TITLE_SAVE_OUTPUT, Message.LABEL_SAVE_OUTPUT, None):
            self._createOutput()

    # --------------------------- INFO functions -----------------------------
    def _summary(self):
        summary = []
        if self.getOutputsSize() >= 1:
            for _, outClasses in self.iterOutputAttributes():
                summary.append("Output *%s*:" % outClasses.getNameId().split('.')[1])
                summary.append("    * Total clustered classes: *%s*" % outClasses.getSize())
        else:
            summary.append("Output clustered classes not ready yet")
        return summary

    def _methods(self):
        return [
            "Interactive automatic clustering of conformational spaces",
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
        elif particles.getFirstItem().getZRed().size != 3:
            errors.append("Cluster space is only implemented for reduced spaces of 3 dimensions. Please, used the "
                          "dimensionality reduction protocol available in Flexutils Plugin to generate a valid "
                          "set of particles")
        return errors
