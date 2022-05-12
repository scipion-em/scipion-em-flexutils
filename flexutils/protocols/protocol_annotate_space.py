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
from pyworkflow.object import String
from pyworkflow.protocol import LEVEL_ADVANCED
from pyworkflow.protocol.params import PointerParam, EnumParam, IntParam
import pyworkflow.utils as pwutils

from pwem.protocols import ProtAnalysis3D
from pwem.objects import SetOfClasses3D, Class3D

import flexutils
from flexutils.utils import getOutputSuffix


class ProtFlexAnnotateSpace(ProtAnalysis3D):
    """ Interactive annotation of Zernikes3D space """

    _label = 'annotate space'
    _devStatus = BETA
    OUTPUT_PREFIX = 'zernike3DClasses'

    # --------------------------- DEFINE param functions ----------------------
    def _defineParams(self, form):
        form.addSection(label='General parameters')
        form.addParam('particles', PointerParam, label="Particles to annotate",
                      pointerClass='SetOfParticles', important=True,
                      help="Particles must have a set of Zernike3D coefficients associated")
        form.addParam('reference', PointerParam, label="Reference map",
                      pointerClass='Volume', important=True,
                      help='Map used as reference during the Zernike3D execution')
        form.addParam('mask', PointerParam, label="Zernike3D mask",
                      pointerClass='VolumeMask', important=True,
                      help="Mask determining where to compute the Zernike3D deformation field")
        form.addParam('volumes', PointerParam, label="Priors", allowsNull=True, pointerClass="SetOfVolumes",
                      help='A set of volumes with Zernike3D coefficients associated (computed using '
                           '"Refernce map" as reference) to add as prior information to the Zernike3D '
                           'space')
        form.addParam('mode', EnumParam, choices=['UMAP', 'PCA'],
                      default=0, display=EnumParam.DISPLAY_HLIST,
                      label="Dimensionality reduction method",
                      help="\t * UMAP: usually leads to more meaningfull spaces, although execution "
                           "is higher\n"
                           "\t * PCA: faster but less meaningfull spaces \n"
                           "UMAP and PCA are only computed the first time the are used. Afterwards, they "
                           "will be reused to increase performance")
        form.addParam('neighbours', IntParam, label="Number of particles to associate to selections",
                      default=5000, expertLevel=LEVEL_ADVANCED)

    # --------------------------- INSERT steps functions ----------------------
    def _insertAllSteps(self):
        self._insertFunctionStep(self.launchVolumeSlicer, interactive=True)

    def _createOutput(self):
        particles = self.particles.get()
        reference = self.reference.get()
        partIds = list(particles.getIdSet())
        neighbours = self.neighbours.get()
        num_part = particles.getSize()

        # Read selected coefficients
        z_clnm_vw = []
        with open(self._getExtraPath('saved_selections.txt')) as f:
            lines = f.readlines()
            for line in lines:
                z_clnm_vw.append(np.fromstring(line, dtype=float, sep=' '))
        z_clnm_vw = np.asarray(z_clnm_vw[1:])

        # Read Zernike coefficients
        z_clnm = np.loadtxt(self._getExtraPath("z_clnm.txt"))

        # Create KDTree
        kdtree = KDTree(z_clnm)

        # Create SetOfClasses3Dl
        suffix = getOutputSuffix(self, SetOfClasses3D)
        classes3D = self._createSetOfClasses3D(particles, suffix)

        # Popoulate SetOfClasses3D with KMean particles
        for clInx in range(z_clnm_vw.shape[0]):
            _, currIds = kdtree.query(z_clnm_vw[clInx].reshape(1, -1), k=neighbours+10)
            currIds = currIds[0]

            newClass = Class3D()
            newClass.copyInfo(particles)
            newClass.setAcquisition(particles.getAcquisition())
            newClass.setRepresentative(reference)
            newClass.getRepresentative()._xmipp_sphCoefficients = String(','.join(['%f' % c for c in z_clnm_vw[clInx]]))

            classes3D.append(newClass)

            enabledClass = classes3D[newClass.getObjId()]
            enabledClass.enableAppend()
            for idx in range(neighbours):
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
        reference = self.reference.get()
        mask = self.mask.get()
        volumes = self.volumes.get()

        # Resize reference map to increase real time conformation inspection performance
        inputFile = reference.getFileName()
        outFile = self._getExtraPath('reference.mrc')
        if not os.path.isfile(outFile):
            if pwutils.getExt(inputFile) == ".mrc":
                inputFile += ":mrc"
            self.runJob("xmipp_image_resize",
                        "-i %s -o %s --dim %d " % (inputFile,
                                                   outFile,
                                                   64), numberOfMpi=1)

        # Resize mask
        inputFile = mask.getFileName()
        outFile = self._getExtraPath('mask.mrc')
        if not os.path.isfile(outFile):
            if pwutils.getExt(inputFile) == ".mrc":
                inputFile += ":mrc"
            self.runJob("xmipp_image_resize",
                        "-i %s -o %s --dim %d " % (inputFile,
                                                   outFile,
                                                   64), numberOfMpi=1)
            self.runJob("xmipp_transform_threshold",
                        "-i %s -o %s --select below 0.01 "
                        "--substitute binarize " % (outFile, outFile), numberOfMpi=1)


        # Get image coefficients and scale them to reference size
        factor = 64 / reference.getDim()[0]
        z_clnm_part = particles.aggregate(["MAX"], "_xmipp_sphCoefficients", ["_xmipp_sphCoefficients"])
        z_clnm_part = factor * np.asarray([np.fromstring(d['_xmipp_sphCoefficients'], sep=",") for d in z_clnm_part])

        # Get volume coefficients (if exist) and scale them to reference size
        z_clnm_vol = np.asarray([np.zeros(z_clnm_part.shape[1])])
        if volumes:
            z_clnm_aux = volumes.aggregate(["MAX"], "_xmipp_sphCoefficients", ["_xmipp_sphCoefficients"])
            z_clnm_aux = factor * np.asarray([np.fromstring(d['_xmipp_sphCoefficients'], sep=",") for d in z_clnm_aux])
            z_clnm_vol = factor * np.vstack([z_clnm_vol, z_clnm_aux])

        # Get useful parameters
        num_vol = z_clnm_vol.shape[0]
        z_clnm = np.vstack([z_clnm_part, z_clnm_vol])

        # Compute/Read UMAP or PCA
        mode = self.mode.get()
        if mode == 0:
            file_coords = self._getExtraPath("umap_coords.txt")
            if not os.path.isfile(file_coords):
                from umap import UMAP
                umap = UMAP(n_components=3, n_neighbors=15, n_epochs=1000).fit(z_clnm)
                coords = umap.transform(z_clnm)
                np.savetxt(file_coords, coords)
        elif mode == 1:
            file_coords = self._getExtraPath("pca_coords.txt")
            if not os.path.isfile(file_coords):
                from sklearn.decomposition import PCA
                pca = PCA(n_components=3).fit(z_clnm)
                coords = pca.transform(z_clnm)
                np.savetxt(file_coords, coords)
        deformation = self.computeNormRows(z_clnm)

        path = self._getExtraPath()

        # Generate files to call command line
        file_z_clnm = self._getExtraPath("z_clnm.txt")
        file_deformation = self._getExtraPath("deformation.txt")
        np.savetxt(file_z_clnm, z_clnm)
        np.savetxt(file_deformation, deformation)

        # Run slicer
        program = "python " + os.path.join(os.path.dirname(flexutils.__file__), "viewers", "viewer_3d_slicer.py")
        args = "--coords %s --z_clnm %s --deformation %s --path %s --map_file %s " \
               "--mask_file %s --l1 %d --l2 %d --d %d --num_vol %d " \
               % (file_coords, file_z_clnm, file_deformation, path, self._getExtraPath('reference.mrc'),
                  self._getExtraPath('mask.mrc'), particles.L1.get(), particles.L2.get(),
                  2 * particles.Rmax.get(), num_vol)
        self.runJob(program, args)

        if os.path.isfile(self._getExtraPath("saved_selections.txt")):
            self._createOutput()

    # --------------------------- UTILS functions ----------------------------
    def computeNormRows(self, array):
        norm = []
        size = int(array.shape[1] / 3)
        for vec in array:
            c_3d = np.vstack([vec[:size], vec[size:2 * size], vec[2 * size:]])
            norm.append(np.linalg.norm(np.linalg.norm(c_3d, axis=1)))
        return np.vstack(norm).flatten()

    # --------------------------- INFO functions -----------------------------
    def _summary(self):
        summary = []
        if self.getOutputsSize() >= 1:
            for _, outClasses in self.iterOutputAttributes():
                summary.append("Output *%s*:" % outClasses.getNameId().split('.')[1])
                summary.append("    * Total Zernike3D classes: *%s*" % outClasses.getSize())
        else:
            summary.append("Output Zernike3D classes not ready yet")
        return summary

    def _methods(self):
        return [
            "Interactive annotation of Zernike3D coefficient space",
        ]
