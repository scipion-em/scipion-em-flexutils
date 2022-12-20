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
from sklearn.neighbors import KDTree

from pyworkflow import BETA
from pyworkflow.object import String, CsvList
from pyworkflow.protocol import LEVEL_ADVANCED
from pyworkflow.protocol.params import PointerParam, EnumParam, IntParam, BooleanParam, MultiPointerParam
import pyworkflow.utils as pwutils
from pyworkflow.utils.properties import Message
from pyworkflow.gui.dialog import askYesNo

from pwem.emlib.image import ImageHandler
from pwem.protocols import ProtAnalysis3D
from pwem.objects import SetOfClasses3D, Class3D, Volume, SetOfVolumes

import flexutils
from flexutils.utils import getOutputSuffix, computeNormRows
import flexutils.constants as const

import xmipp3


class ProtFlexAnnotateSpace(ProtAnalysis3D):
    """ Interactive annotation of conformational spaces """

    _label = 'annotate space'
    _devStatus = BETA
    OUTPUT_PREFIX = 'flexible3DClasses'

    # --------------------------- DEFINE param functions ----------------------
    def _defineParams(self, form):
        form.addSection(label='General parameters')
        form.addParam('particles', PointerParam, label="Particles to annotate",
                      pointerClass='SetOfParticles', important=True,
                      help="Particles must have a flexibility information associated (Zernike3D, CryoDrgn...")
        form.addParam('reference', PointerParam, label="Reference map",
                      condition="particles and not hasattr(particles,'refMap') "
                                "and hasattr(particles.getFirstItem(),'_xmipp_sphCoefficients')",
                      pointerClass='Volume', important=True,
                      help='Map used as reference during the Zernike3D execution')
        form.addParam('mask', PointerParam, label="Zernike3D mask",
                      condition="particles and not hasattr(particles,'refMask') "
                                "and hasattr(particles.getFirstItem(),'_xmipp_sphCoefficients')",
                      pointerClass='VolumeMask', important=True,
                      help="Mask determining where to compute the Zernike3D deformation field")
        form.addHidden('volumes', PointerParam, label="Priors", allowsNull=True,
                       pointerClass="SetOfVolumes, Volume",
                       condition="particles and hasattr(particles.getFirstItem(),'_xmipp_sphCoefficients')",
                       help='Volumes with Zernike3D coefficients associated (computed using '
                            '"Refernce map" as reference) to add as prior information to the Zernike3D '
                            'space')
        form.addParam('priors', MultiPointerParam, label="Priors", allowsNull=True,
                      pointerClass="SetOfVolumes, Volume",
                      condition="particles and hasattr(particles.getFirstItem(),'_xmipp_sphCoefficients')",
                      help='Volumes with Zernike3D coefficients associated (computed using '
                           '"Refernce map" as reference) to add as prior information to the Zernike3D '
                           'space')
        form.addParam('l1', IntParam, default=3,
                      label='Zernike Degree',
                      expertLevel=LEVEL_ADVANCED,
                      condition="particles and hasattr(particles.getFirstItem(),'_xmipp_sphCoefficients')"
                                "and not hasattr(particles,'L1')",
                      help='Degree Zernike Polynomials of the deformation=1,2,3,...')
        form.addParam('l2', IntParam, default=2,
                      label='Harmonical Degree',
                      condition="particles and hasattr(particles.getFirstItem(),'_xmipp_sphCoefficients')"
                                "and not hasattr(particles,'L2')",
                      expertLevel=LEVEL_ADVANCED,
                      help='Degree Spherical Harmonics of the deformation=1,2,3,...')
        form.addParam('boxSize', IntParam, label="Box size",
                      condition="particles and hasattr(particles.getFirstItem(),'_cryodrgnZValues')",
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

        # Read selected coefficients
        z_space_vw = []
        with open(self._getExtraPath('saved_selections.txt')) as f:
            lines = f.readlines()
            for line in lines:
                z_space_vw.append(np.fromstring(line, dtype=float, sep=' '))
        z_space_vw = np.asarray(z_space_vw[self.num_vol:])

        # Read space
        z_space = np.loadtxt(self._getExtraPath("z_space.txt"))

        # Create KDTree
        kdtree = KDTree(z_space)

        # Create SetOfClasses3Dl
        suffix = getOutputSuffix(self, SetOfClasses3D)
        classes3D = self._createSetOfClasses3D(particles, suffix)

        # Popoulate SetOfClasses3D with KMean particles
        for clInx in range(z_space_vw.shape[0]):
            _, currIds = kdtree.query(z_space_vw[clInx].reshape(1, -1), k=neighbors+10)
            currIds = currIds[0]

            newClass = Class3D()
            newClass.copyInfo(particles)
            newClass.setAcquisition(particles.getAcquisition())
            representative = Volume()
            representative.setSamplingRate(sr)

            csv_z_space = CsvList()
            for c in z_space_vw[clInx]:
                csv_z_space.append(c)

            # ****** Fill representative information *******
            if hasattr(particles.getFirstItem(), "_xmipp_sphCoefficients"):
                reference = particles.refMap.get() if hasattr(particles, "refMap") else self.reference.get().getFileName()
                mask = particles.refMask.get() if hasattr(particles, "refMask") else self.mask.get().getFileName()

                # Resize coefficients
                factor = (ImageHandler().read(reference).getDimensions()[0] / 64)
                for idx in range(len(csv_z_space)):
                    csv_z_space[idx] *= factor

                L1 = particles.L1.get() if hasattr(particles, 'L1') else self.l1.get()
                L2 = particles.L2.get() if hasattr(particles, 'L2') else self.l2.get()
                # Rmax = particles.Rmax
                reference_file = String(reference)
                mask_file = String(mask)

                representative.setLocation(reference)
                representative.L1 = L1
                representative.L2 = L2
                # representative.Rmax = Rmax
                representative.refMap = reference_file
                representative.refMask = mask_file
                representative._xmipp_sphCoefficients = csv_z_space

            elif hasattr(particles.getFirstItem(), "_cryodrgnZValues"):
                from cryodrgn.utils import generateVolumes
                generateVolumes(z_space_vw[clInx], particles._cryodrgnWeights.get(),
                                particles._cryodrgnConfig.get(), self._getExtraPath(), downsample=self.boxSize.get(),
                                apix=particles.getSamplingRate())
                ImageHandler().scaleSplines(self._getExtraPath('vol_000.mrc'),
                                            self._getExtraPath('class_%d.mrc') % clInx, 1,
                                            finalDimension=particles.getXDim())
                representative.setLocation(self._getExtraPath('class_%d.mrc') % clInx)
                representative._cryodrgnZValues = csv_z_space

            # ********************

            newClass.setRepresentative(representative)

            classes3D.append(newClass)

            enabledClass = classes3D[newClass.getObjId()]
            enabledClass.enableAppend()
            for idx in range(neighbors):
                itemId = currIds[idx]
                while itemId >= num_part:
                    currIds = np.delete(currIds, idx)
                    itemId = currIds[idx]
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
    def launchVolumeSlicer(self):
        particles = self.particles.get()
        self.num_vol = 0

        # ********* Get Z space *********
        if hasattr(particles.getFirstItem(), "_xmipp_sphCoefficients"):
            reference = particles.refMap.get() if hasattr(particles, "refMap") else self.reference.get().getFileName()
            mask = particles.refMask.get() if hasattr(particles, "refMask") else self.mask.get().getFileName()
            volumes = self.priors

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
            z_space = []
            for particle in particles.iterItems():
                z_space.append(np.fromstring(particle._xmipp_sphCoefficients.get(), sep=","))
            z_space = np.asarray(z_space)

            # Get volume coefficients (if exist) and scale them to reference size
            # FIXME: Can we do the for loop with the aggregate? (follow ID order)
            # z_clnm_vol = np.asarray([np.zeros(z_clnm_part.shape[1])])
            # if volumes:
            #     z_clnm_aux = volumes.aggregate(["MAX"], "_index", ["_xmipp_sphCoefficients", "_index"])
            #     z_clnm_aux = factor * np.asarray([np.fromstring(d['_xmipp_sphCoefficients'], sep=",") for d in z_clnm_aux])
            #     z_clnm_vol = factor * np.vstack([z_clnm_vol, z_clnm_aux])
            z_space_vol = []
            if volumes:
                for pointer in volumes:
                    item = pointer.get()
                    if isinstance(item, Volume):
                        z_space_vol.append(np.fromstring(item._xmipp_sphCoefficients.get(), sep=","))
                    elif isinstance(item, SetOfVolumes):
                        for volume in item.iterItems():
                            z_space_vol.append(np.fromstring(volume._xmipp_sphCoefficients.get(), sep=","))
            z_space_vol = np.asarray(z_space_vol)

            # Get useful parameters
            self.num_vol = z_space_vol.shape[0]
            if self.num_vol > 0:
                z_space = np.vstack([z_space, z_space_vol])

            # Resize coefficients
            z_space = (64 / ImageHandler().read(reference).getDimensions()[0]) * z_space

        elif hasattr(particles.getFirstItem(), "_cryodrgnZValues"):
            z_space = []
            for particle in particles.iterItems():
                z_space.append(np.fromstring(particle._cryodrgnZValues.get(), sep=","))
            z_space = np.asarray(z_space)

        # ********************

        # Generate files to call command line
        file_z_space = self._getExtraPath("z_space.txt")
        file_interp_val = self._getExtraPath("interp_val.txt")
        np.savetxt(file_z_space, z_space)

        # Compute/Read UMAP or PCA
        file_coords = self._getExtraPath("red_coords.txt")
        red_space = []
        for particle in particles.iterItems():
            red_space.append(np.fromstring(particle._red_space.get(), sep=","))
        red_space = np.asarray(red_space)
        np.savetxt(file_coords, red_space)

        # ********* Get interpolation value for coloring the space *********
        if hasattr(particles.getFirstItem(), "_xmipp_sphCoefficients"):
            interp_val = computeNormRows(z_space)
        else:
            interp_val = np.ones([1, z_space.shape[0]])

        # *********

        # Generate files to call command line
        np.savetxt(file_interp_val, interp_val)
        path = os.path.abspath(self._getExtraPath())

        # ********* Run viewer *********
        if hasattr(particles.getFirstItem(), "_xmipp_sphCoefficients"):
            L1 = particles.L1.get() if hasattr(particles, 'L1') else self.l1.get()
            L2 = particles.L2.get() if hasattr(particles, 'L2') else self.l2.get()
            args = "--data %s --z_space %s --interp_val %s --path %s " \
                   "--L1 %d --L2 %d --n_vol %d --mode Zernike3D" \
                   % (file_coords, file_z_space, file_interp_val, path,
                      L1, L2, self.num_vol)

        elif hasattr(particles.getFirstItem(), "_cryodrgnZValues"):
            args = "--data %s --z_space %s --interp_val %s --path %s " \
                   "--weights %s --config %s --boxsize %d --sr %f --mode CryoDrgn" \
                   % (file_coords, file_z_space, file_interp_val, path,
                      particles._cryodrgnWeights.get(), particles._cryodrgnConfig.get(), self.boxSize.get(),
                      particles.getSamplingRate())

        dimensions = red_space.shape[1]
        if dimensions == 2:
            program = os.path.join(const.VIEWERS, "viewer_interactive_2d.py")
        elif dimensions == 3:
            program = os.path.join(const.VIEWERS, "viewer_3d_slicer.py")
        program = flexutils.Plugin.getProgram(program)
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
                summary.append("    * Total annotated classes: *%s*" % outClasses.getSize())
        else:
            summary.append("Output annotated classes not ready yet")
        return summary

    def _methods(self):
        return [
            "Interactive annotation of conformational spaces",
        ]
