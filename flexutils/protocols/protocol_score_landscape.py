# **************************************************************************
# *
# * Authors:    David Herreros Calero (dherreros@cnb.csic.es)
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
# *  e-mail address 'coss@cnb.csic.es'
# *
# **************************************************************************


import os
import numpy as np

from pyworkflow import NEW
import pyworkflow.protocol.params as params
import pyworkflow.utils as pwutils
from pyworkflow.object import Float

from pwem.protocols import ProtAnalysis3D, ProtFlexBase

import flexutils
import flexutils.constants as const


class ProtFlexScoreLandscape(ProtAnalysis3D, ProtFlexBase):
    """
    Scoring and (optional) filtering of landscape samples based on neighbour distance
    """

    """
            The ProtFlexScoreLandscape protocol evaluates conformational
            landscapes by measuring the local neighbourhood consistency of
            each particle within the flexibility space. Its main purpose is
            to identify samples that behave as structural outliers and, if
            desired, remove them from the dataset to improve the quality and
            interpretability of downstream flexibility analyses.

            Inputs and General Workflow

            The protocol requires a set of particles associated with a
            flexibility landscape representation. Each particle contains a
            conformational descriptor that defines its position in the latent
            or flexibility space. During execution, the protocol extracts the
            landscape coordinates from all particles and computes neighbour-
            based Z-scores that estimate how isolated or atypical each sample
            is relative to its surrounding conformational environment.

            Users can control the number of neighbours considered during the
            analysis. Smaller neighbourhood values emphasize highly local
            structural variations, whereas larger neighbourhoods capture more
            global conformational organization. The protocol also provides an
            optional approximate nearest-neighbour strategy that accelerates
            computation for large datasets while slightly reducing neighbour
            search precision.

            The protocol supports two operation modes. In score mode, all
            particles are preserved and assigned an outlier score describing
            their relative distance within the landscape. In filter mode,
            particles with Z-scores above a user-defined threshold are
            discarded, generating a cleaned conformational landscape with
            reduced influence from isolated or noisy samples.

            Outputs and Interpretation

            After execution, the protocol produces a new set of particles
            containing the computed outlier scores. When filtering is enabled,
            only particles satisfying the selected Z-score threshold are kept
            in the output dataset. The resulting landscape can then be used
            for downstream dimensionality reduction, clustering, trajectory
            analysis, or conformational interpretation.

            Biological Perspective

            In cryo-EM flexibility analysis, conformational landscapes may
            contain noisy particles, reconstruction artefacts, or rare
            structural states that distort the global organization of the
            embedding space. By identifying samples with abnormal neighbour
            distributions, this protocol helps improve the robustness and
            biological consistency of downstream analyses.

            From a biological perspective, filtering should be applied with
            caution because highly isolated samples may represent either
            reconstruction noise or genuinely rare conformational states.
            Conservative thresholds are generally recommended when studying
            subtle structural heterogeneity or transient motions.

            Final Perspective

            ProtFlexScoreLandscape provides an efficient strategy for scoring
            and refining conformational landscapes based on neighbour-distance
            statistics. By detecting anomalous samples and optionally removing
            them, the protocol improves the stability, clarity, and biological
            interpretability of cryo-EM flexibility analysis workflows.
        """

    _label = 'score/filter landscape'
    _devStatus = NEW

    def _defineParams(self, form):
        form.addSection(label='Input')
        form.addParam('inputParticles', params.PointerParam,
                      pointerClass='SetOfParticlesFlex',
                      label="Input landscape", important=True,
                      help='Input particles with the flexibility landscape to be analyzed')
        form.addParam('fast', params.BooleanParam, default=True, label="Fast NN search?",
                      help="If True, apprixmate neighbour search will be used. This will increase "
                           "performance at the expense of reducing the accuracy of the neighbour search")
        form.addParam("neighbours", params.IntParam, default=10, label="Number of neighbours",
                      help="Number of nearest neighbours to be search for each landscape sample. Smaller "
                           "values will better approximate the locality features of a sample, while global "
                           "values will capture more general landscape features.")
        form.addParam('filter', params.EnumParam, choices=['score', 'filter'],
                       default=0, display=params.EnumParam.DISPLAY_HLIST,
                       label='Operation mode',
                       help='Determine whether to retrieve the landscape will be scored or '
                            'filtered out based on a threshold')
        form.addParam('outliersThreshold', params.FloatParam, default=1,
                      label="Outliers distance threshold", condition='filter == 1',
                      help='Z-Score value from 0 to infinite. Only coordinates with a Z-Score smaller than '
                           'or equal to the threshold will be kept in the output')

    # --------------------------- INSERT steps functions ----------------------
    def _insertAllSteps(self):
        self._insertFunctionStep('detectOutliers')
        self._insertFunctionStep('createOutputStep')

    def detectOutliers(self):
        self.scoreOutliers = {}
        self.scoreOutliers = []
        inputParticles = self.inputParticles.get()

        # Prepare landscape
        landscape_path = self._getExtraPath("landscape.txt")
        landscape = np.asarray([particle.getZFlex() for particle in inputParticles.iterItems()])
        np.savetxt(landscape_path, landscape)

        # Compute Z-Scores
        args = "--i %s --o %s --neighbours %d" % \
               (landscape_path, self._getExtraPath(), self.neighbours.get())
        program = "compute_landscape_zscores.py"
        program = flexutils.Plugin.getProgram(program)
        self.runJob(program, args)

    def createOutputStep(self):
        outliersThreshold = self.outliersThreshold.get()
        inputParticles = self.inputParticles.get()
        progName = inputParticles.getFlexInfo().getProgName()
        outParticles = self._createSetOfParticlesFlex(progName=progName)
        outParticles.copyInfo(inputParticles)
        z_scores = np.loadtxt(self._getExtraPath("z_scores.txt"))

        idx = 0
        for particle in inputParticles.iterItems():
            outParticle = particle.clone()
            outParticle.setObjId(None)
            z_score = z_scores[idx]
            if self.filter.get():
                if outliersThreshold >= z_score:
                    outParticle.outlierScore = Float(z_score)
                    outParticles.append(outParticle)
            else:
                outParticles.outlierScore = Float(z_score)
                outParticles.append(outParticle)
            idx += 1

        self._defineOutputs(outputParticles=outParticles)
        self._defineSourceRelation(inputParticles, outParticles)


    # --------------------------- UTILS functions ----------------------

    # --------------------------- INFO functions ----------------------
    def _methods(self):
        methodsMsgs = []
        if self.filter.get() == 0:
            methodsMsgs.append("*Operation mode*: score")
            filter = False
        else:
            methodsMsgs.append("*Operation mode*: filter")
            filter = True
        methodsMsgs.append("*Score Outliers*: True")
        if filter:
            methodsMsgs.append("    * Outlier threshold: %.2f" % self.outliersThreshold.get())
        return methodsMsgs

    def _summary(self):
        summary = []
        if self.getOutputsSize() >= 1:
            summary.append("Output *%s*:" % self.outputParticles.getNameId().split('.')[1])
            summary.append("    * Number of particles kept: *%s*" % self.outputParticles.getSize())
        else:
            summary.append("Output particles not ready yet.")
        return summary
