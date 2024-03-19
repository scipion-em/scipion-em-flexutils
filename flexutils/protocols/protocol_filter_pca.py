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


import numpy as np
from sklearn.decomposition import PCA

from pyworkflow import NEW
from pyworkflow.protocol.params import PointerParam, StringParam

from pwem.protocols import ProtAnalysis3D, ProtFlexBase
from pwem.objects import ParticleFlex


class ProtFlexFilterPCA(ProtAnalysis3D, ProtFlexBase):
    """ Dimensionality reduction of spaces based on different methods """

    _label = 'filter PCA'
    _devStatus = NEW
    OUTPUT_PREFIX = 'outputParticles'

    # --------------------------- DEFINE param functions ----------------------
    def _defineParams(self, form):
        form.addSection(label='General parameters')
        form.addParam('particles', PointerParam, label="Input particles",
                      pointerClass='SetOfParticlesFlex', important=True,
                      help="Particles must have a flexibility information associated (Zernike3D, HetSIREN, CryoDRGN...")
        form.addParam('keepPCA', StringParam, default="1,2,3", label="PCA components to keep",
                      help="Determines the PCA components to keep (values or ranges are valid inputs).\n"
                           "Example: 1,2,3 / 5:6 / 2: / :5 / 90% are valid entries")


    # --------------------------- INSERT steps functions ----------------------
    def _insertAllSteps(self):
        self._insertFunctionStep(self.filterPCASpace)
        self._insertFunctionStep(self.createOutputStep)

    # --------------------------- STEPS functions -----------------------------
    def filterPCASpace(self):
        particles = self.particles.get()
        keepPCA = self.stringToList(self.keepPCA.get())

        # ********* Get Z space *********
        z_space = []
        for particle in particles.iterItems():
            z_space.append(particle.getZFlex())
        z_space = np.asarray(z_space)
        # ********************

        # Define PCA space
        pca = PCA(n_components=z_space.shape[1]).fit(z_space)
        explained_variance_ratio = pca.explained_variance_ratio_
        z_pca = pca.transform(z_space)

        # PCA mask
        mask = np.zeros(z_pca.shape[1], dtype=float)
        if "%" in self.keepPCA.get():
            cum_sum = np.cumsum(explained_variance_ratio)
            idx = np.argmin(np.abs(cum_sum - keepPCA))
            mask[:idx + 1] = 1
        else:
            mask[keepPCA] = 1

        # Get masked PCA space
        z_pca = z_pca * mask[None, ...]
        explained_variance_ratio = explained_variance_ratio[mask.astype(bool)]

        # Get new Flex space
        z_masked = pca.inverse_transform(z_pca)
        np.savetxt(self._getExtraPath("z_pca.txt"), z_pca)
        np.savetxt(self._getExtraPath("z_masked.txt"), z_masked)
        np.savetxt(self._getExtraPath("explained_variance_ratio.txt"), explained_variance_ratio)

    def createOutputStep(self):
        z_masked = np.loadtxt(self._getExtraPath("z_masked.txt"))

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

            outParticle.setZFlex(z_masked[idx])

            partSet.append(outParticle)

            idx += 1

        self._defineOutputs(outputParticles=partSet)
        self._defineTransformRelation(self.particles, partSet)

    # --------------------------- UTILS functions ----------------------------
    def stringToList(self, input_str, max_num=100):
        # Remove brackets [] if present
        cleaned_str = input_str.replace('[', '').replace(']', '')

        # Split the cleaned string by both commas and spaces to get all numbers or ranges
        segments = [seg for part in cleaned_str.split(',') for seg in part.split()]

        result = []
        for segment in segments:
            if ':' in segment:  # Handle range
                start, end = segment.split(':')
                start = int(start) if start else 1  # If start is empty, default to 1
                end = int(end) if end else max_num  # If end is empty, default to max_num
                result.extend(range(start - 1, end))  # Add range to result
            elif '%' in segment:
                result.append(0.01 * float(segment[:-1]))
            else:
                result.append(int(segment) - 1)  # Add single number to result

        return result

    # --------------------------- INFO functions -----------------------------
    def _summary(self):
        summary = []
        if self.getOutputsSize() >= 1:
            keepPCA = self.stringToList(self.keepPCA.get())
            explained_variance_ratio = np.loadtxt(self._getExtraPath("explained_variance_ratio.txt"))
            for _, outParticles in self.iterOutputAttributes():
                summary.append("Output *%s*:" % outParticles.getNameId().split('.')[1])
            if "%" in self.keepPCA.get():
                cum_sum = np.cumsum(explained_variance_ratio)
                idx = np.argmin(np.abs(cum_sum - keepPCA))
                keepPCA = range(0, idx)
            for pca, ratio in zip(keepPCA, explained_variance_ratio):
                summary.append(f"PC_{pca + 1} --> {ratio:.2f}%")
            summary.append(f"Total variance explained by kept components --> {100. * explained_variance_ratio.sum()}%")
        else:
            summary.append("Output particles not ready yet")
        return summary

    def _methods(self):
        return [
            "PCA filtering of conformational spaces",
        ]

    def _validate(self):
        errors = []

        # Check number of reduced dimensions
        particles = self.particles.get()
        original_dimensions = len(particles.getFirstItem().getZFlex())
        max_dimensions = max(self.stringToList(self.keepPCA.get())) + 1
        min_dimensions = min(self.stringToList(self.keepPCA.get())) + 1
        if max_dimensions > original_dimensions:
            errors.append(f"The number of dimensions in the reduced space must be smalller than or equal to "
                          f"the number of dimensions in the original space (original dimensions "
                          f"= {original_dimensions}. Currently set to {max_dimensions}."
                          "it is set to {dimensions}")
        if min_dimensions < 1:
            errors.append(f"The minimum PCA dimension to be considered must be 1, but {min_dimensions} was given.")

        return errors

