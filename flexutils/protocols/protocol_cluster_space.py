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

from pyworkflow import BETA
from pyworkflow.object import String, CsvList
from pyworkflow.protocol.params import PointerParam, EnumParam, IntParam, BooleanParam, MultiPointerParam
import pyworkflow.utils as pwutils

from pwem.emlib.image import ImageHandler
from pwem.protocols import ProtAnalysis3D
from pwem.objects import SetOfClasses3D, Class3D, Volume

import flexutils
from flexutils.utils import getOutputSuffix, computeNormRows
import flexutils.constants as const

import xmipp3


class ProtFlexClusterSpace(ProtAnalysis3D):
    """ Interactive clustering of Zernikes3D space """

    _label = 'cluster space'
    _devStatus = BETA
    OUTPUT_PREFIX = 'flexible3DClasses'

    # --------------------------- DEFINE param functions ----------------------
    def _defineParams(self, form):
        form.addSection(label='General parameters')
        form.addParam('particles', PointerParam, label="Particles to annotate",
                      pointerClass='SetOfParticles', important=True,
                      help="Particles must have a flexibility information associated (Zernike3D, CryoDrgn...")
        form.addParam('reference', PointerParam, label="Reference map",
                      pointerClass='Volume', important=True,
                      condition="particles and not hasattr(particles,'refMap') "
                                "and hasattr(particles.getFirstItem(),'_xmipp_sphCoefficients')",
                      help='Map used as reference during the Zernike3D execution')
        form.addParam('mask', PointerParam, label="Zernike3D mask",
                      pointerClass='VolumeMask', important=True,
                      condition="particles and not hasattr(particles,'refMask') "
                                "and hasattr(particles.getFirstItem(),'_xmipp_sphCoefficients')",
                      help="Mask determining where to compute the Zernike3D deformation field")
        form.addParam('volumes', MultiPointerParam, label="Priors", allowsNull=True,
                      pointerClass="SetOfVolumes, Volume",
                      condition="particles and hasattr(particles.getFirstItem(),'_xmipp_sphCoefficients')",
                      help='A set of volumes with Zernike3D coefficients associated (computed using '
                           '"Reference map" as reference) to add as prior information to the Zernike3D '
                           'space')
        form.addParam('mode', EnumParam, choices=['UMAP', 'PCA'],
                      default=0, display=EnumParam.DISPLAY_HLIST,
                      label="Dimensionality reduction method",
                      help="\t * UMAP: usually leads to more meaningfull spaces, although execution "
                           "is higher\n"
                           "\t * PCA: faster but less meaningfull spaces \n"
                           "UMAP and PCA are only computed the first time the are used. Afterwards, they "
                           "will be reused to increase performance")
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

    # --------------------------- INSERT steps functions ----------------------
    def _insertAllSteps(self):
        self._insertFunctionStep(self.launchInteractiveClustering, interactive=True)

    def _createOutput(self):
        particles = self.particles.get()
        partIds = list(particles.getIdSet())
        sr = particles.getSamplingRate()

        # Read KMean coefficients
        z_space_vw = []
        with open(self._getExtraPath('saved_selections.txt')) as f:
            lines = f.readlines()
            for line in lines:
                z_space_vw.append(np.fromstring(line, dtype=float, sep=' '))
        z_space_vw = np.asarray(z_space_vw[self.num_vol:])

        # Read KMean labels
        km_labels = np.loadtxt(self._getExtraPath('kmean_labels.txt'), delimiter=" ")[:-self.num_vol]

        # Create SetOfClasses3D
        suffix = getOutputSuffix(self, SetOfClasses3D)
        classes3D = self._createSetOfClasses3D(particles, suffix)

        # Popoulate SetOfClasses3D with KMean particles
        for clInx in range(z_space_vw.shape[0]):
            currIds = np.where(km_labels == clInx)[0]

            newClass = Class3D()
            newClass.copyInfo(particles)
            newClass.setAcquisition(particles.getAcquisition())
            representative = Volume()
            csv_z_space = CsvList()
            for c in z_space_vw[clInx]:
                csv_z_space.append(c)

            # Fill representative information (for Zernike3D...)
            if hasattr(particles.getFirstItem(), "_xmipp_sphCoefficients"):
                reference = particles.refMap.get() if hasattr(particles, "refMap") else self.reference.get().getFileName()
                mask = particles.refMask.get() if hasattr(particles, "refMask") else self.mask.get().getFileName()

                # Resize coefficients
                factor = (ImageHandler().read(reference).getDimensions()[0] / 64)
                for idx in range(len(csv_z_space)):
                    csv_z_space[idx] *= factor

                L1 = particles.L1
                L2 = particles.L2
                Rmax = particles.Rmax
                reference_file = String(reference)
                mask_file = String(mask)

                representative.setLocation(reference)
                representative.L1 = L1
                representative.L2 = L2
                representative.Rmax = Rmax
                representative.refMap = reference_file
                representative.refMask = mask_file
                representative._xmipp_sphCoefficients = csv_z_space

            representative.setSamplingRate(sr)
            newClass.setRepresentative(representative)

            classes3D.append(newClass)

            enabledClass = classes3D[newClass.getObjId()]
            enabledClass.enableAppend()
            for itemId in currIds:
                item = particles[partIds[itemId]]
                enabledClass.append(item)

            classes3D.update(enabledClass)

        # Save new output
        name = self.OUTPUT_PREFIX + suffix
        args = {}
        args[name] = classes3D
        self._defineOutputs(**args)
        self._defineSourceRelation(particles, classes3D)
        self._updateOutputSet(name, classes3D, state=classes3D.STREAM_CLOSED)

    # --------------------------- STEPS functions -----------------------------
    def launchInteractiveClustering(self):
        particles = self.particles.get()
        self.num_vol = 0

        # ********* Get Z space *********
        if hasattr(particles.getFirstItem(), "_xmipp_sphCoefficients"):
            reference = particles.refMap.get() if hasattr(particles, "refMap") else self.reference.get().getFileName()
            mask = particles.refMask.get() if hasattr(particles, "refMask") else self.mask.get().getFileName()
            volumes = self.volumes.get()

            # Copy original reference and mask to extra
            ih = ImageHandler()
            ih.convert(reference, self._getExtraPath("reference_original.mrc"))
            ih.convert(mask, self._getExtraPath("mask_reference_original.mrc"))

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


            # Get image coefficients and scale them to reference size
            # FIXME: Can we do the for loop with the aggregate? (follow ID order)
            # factor = 64 / particles.getXDim()
            # z_clnm_part = particles.aggregate(["MAX"], "_index", ["_xmipp_sphCoefficients", "_index"])
            # z_clnm_part = factor * np.asarray([np.fromstring(d['_xmipp_sphCoefficients'], sep=",") for d in z_clnm_part])
            z_space_part = []
            for particle in particles.iterItems():
                z_space_part.append(np.fromstring(particle._xmipp_sphCoefficients.get(), sep=","))
            z_space_part = np.asarray(z_space_part)

            # Get volume coefficients (if exist) and scale them to reference size
            # FIXME: Can we do the for loop with the aggregate? (follow ID order)
            # z_clnm_vol = np.asarray([np.zeros(z_clnm_part.shape[1])])
            # if volumes:
            #     z_clnm_aux = volumes.aggregate(["MAX"], "_index", ["_xmipp_sphCoefficients", "_index"])
            #     z_clnm_aux = factor * np.asarray([np.fromstring(d['_xmipp_sphCoefficients'], sep=",") for d in z_clnm_aux])
            #     z_clnm_vol = factor * np.vstack([z_clnm_vol, z_clnm_aux])
            z_space_vol = np.asarray([np.zeros(z_space_part.shape[1])])
            if volumes:
                for volume in volumes.iterItems():
                    z_space_vol = np.vstack([z_space_vol, np.fromstring(volume._xmipp_sphCoefficients.get(), sep=",")])
                # z_clnm_vol *= factor

            # Get useful parameters
            self.num_vol = z_space_vol.shape[0]
            z_space = np.vstack([z_space_part, z_space_vol])

            # Resize coefficients
            z_space = (64 / ImageHandler().read(reference).getDimensions()[0]) * z_space

        # ********************

        # Generate files to call command line
        file_z_space = self._getExtraPath("z_space.txt")
        file_interp_val = self._getExtraPath("interp_val.txt")
        np.savetxt(file_z_space, z_space)

        # Compute/Read UMAP or PCA
        mode = self.mode.get()
        if mode == 0:
            file_coords = self._getExtraPath("umap_coords.txt")
            if not os.path.isfile(file_coords):
                args = "--input %s --umap --output %s --n_neighbors %d --n_epochs %d " \
                       % (file_z_space, file_coords, self.nb_umap.get(), self.epochs_umap.get())
                if self.densmap_umap.get():
                    args += " --densmap"
                program = os.path.join(const.XMIPP_SCRIPTS, "dimensionality_reduction.py")
                program = flexutils.Plugin.getProgram(program)
                self.runJob(program, args)
        elif mode == 1:
            file_coords = self._getExtraPath("pca_coords.txt")
            if not os.path.isfile(file_coords):
                args = "--input %s --pca --output %s" % (file_z_space, file_coords)
                program = os.path.join(const.XMIPP_SCRIPTS, "dimensionality_reduction.py")
                program = flexutils.Plugin.getProgram(program)
                self.runJob(program, args)

        # ********* Get interpolation value for coloring the space *********
        if hasattr(particles.getFirstItem(), "_xmipp_sphCoefficients"):
            interp_val = computeNormRows(z_space)
        else:
            interp_val = np.zeros([1, z_space.shape[0]])

        # *********

        # Generate files to call command line
        if not os.path.isfile(file_interp_val):
            np.savetxt(file_interp_val, interp_val)
        path = self._getExtraPath()

        # ********* Run viewer *********
        if hasattr(particles.getFirstItem(), "_xmipp_sphCoefficients"):
            args = "--data %s --z_space %s --interp_val %s --path %s " \
                   "--L1 %d --L2 %d --n_vol %d --mode Zernike3D" \
                   % (file_coords, file_z_space, file_interp_val, path, particles.L1.get(), particles.L2.get(),
                      self.num_vol)
            program = os.path.join(const.VIEWERS, "viewer_3d_pc.py")
            program = flexutils.Plugin.getProgram(program)
            self.runJob(program, args)

        # *********

        if os.path.isfile(self._getExtraPath("saved_selections.txt")):
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
            "Interactive automatic clustering of flexible spaces",
        ]
