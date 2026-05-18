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
import pandas as pd

from pyworkflow import NEW
from pyworkflow.protocol.params import PointerParam, IntParam, EnumParam

from pwem.protocols import ProtAnalysis3D

import flexutils


class ProtFlexOptimalClusters(ProtAnalysis3D):
    """ Optimal cluster number analysis for flexibility spaces """

    """
            The ProtFlexOptimalClusters protocol estimates the most suitable
            number of clusters within a conformational flexibility space.
            Its main purpose is to analyze the structural distribution of
            particles in a latent or flexibility space and identify the
            cluster configuration that best represents the underlying
            conformational heterogeneity of the dataset.

            Inputs and General Workflow

            The protocol requires a set of particles containing flexibility
            information, such as latent coordinates obtained from methods
            like Zernike3D, HetSIREN, or CryoDRGN. Users can define the
            maximum number of clusters to evaluate as well as the clustering
            strategy to be used during the analysis.

            During execution, the protocol extracts the flexibility coordinates
            associated with all particles and stores them as a numerical
            conformational space. The protocol then performs an automatic
            clustering analysis using the selected clustering method and
            evaluates multiple cluster configurations up to the user-defined
            maximum value.

            Several statistical metrics are computed to estimate the optimal
            number of clusters, including methods such as Gap Statistic,
            Elbow analysis, Silhouette score, Calinski-Harabasz index, and
            Davies-Bouldin score. These metrics help determine the clustering
            configuration that best captures the structural organization of
            the flexibility landscape.

            Outputs and Interpretation

            After execution, the protocol generates clustering analysis
            results containing the estimated optimal number of clusters
            according to the evaluated statistical criteria. These results
            provide guidance for downstream conformational classification
            and structural interpretation workflows.

            Biological Perspective

            Determining the appropriate number of conformational states is
            a fundamental step in cryo-EM flexibility analysis. An incorrect
            number of clusters may artificially merge distinct conformations
            or fragment biologically related states. By automatically
            evaluating multiple clustering criteria, this protocol helps
            identify meaningful structural subdivisions within heterogeneous
            datasets.

            Final Perspective

            ProtFlexOptimalClusters provides an automated strategy for
            exploring conformational heterogeneity and estimating the most
            representative number of structural states in flexibility spaces.
            This improves the robustness and interpretability of downstream
            cryo-EM conformational analysis workflows.
        """

    _label = 'find optimal clusters'
    _devStatus = NEW
    OUTPUT_PREFIX = 'selectedReference'
    CHOICES = ["KMeans"]

    # --------------------------- DEFINE param functions ----------------------
    def _defineParams(self, form):
        form.addSection(label='General parameters')
        form.addParam('inputParticles', PointerParam, label="Input particles",
                      pointerClass='SetOfParticlesFlex', important=True,
                      help="Particles with the flexibility space to be analyzed")
        form.addParam('maxClusters', IntParam, default=15, label="Maximum number of clusters",
                      help="Determines up to how many clusters the analysis will be carried "
                           "out")
        form.addParam('clusterMethod', EnumParam, default=0, choices=self.CHOICES,
                      label="Clustering method", display=EnumParam.DISPLAY_COMBO,
                      help="Determines the clustering method that will be used to analyze "
                           "the flexible space")
        form.addParallelSection(threads=4, mpi=0)

    # --------------------------- INSERT steps functions ----------------------
    def _insertAllSteps(self):
        self._insertFunctionStep(self.clusterAnalysis)

    # --------------------------- STEPS functions -----------------------------
    def clusterAnalysis(self):
        inputParticles = self.inputParticles.get()
        maxClusters = self.maxClusters.get()
        clusterMethod = self.CHOICES[self.clusterMethod.get()]
        outPath = self._getExtraPath()
        dataPath = self._getExtraPath("flex_space.txt")

        # Save flex space to file
        flex_space = []
        for particle in inputParticles.iterItems():
            flex_space.append(particle.getZFlex())
        flex_space = np.asarray(flex_space)
        np.savetxt(dataPath, flex_space)

        # Optimal cluster analysis
        args = "--data_file %s --out_path %s --max_clusters %d --cluster_method %s" \
               % (dataPath, outPath, maxClusters, clusterMethod)
        program = flexutils.Plugin.getTensorflowProgram("find_optimal_clusters.py", python=False)
        self.runJob(program, args, numberOfMpi=1)


    # --------------------------- UTILS functions ----------------------------

    # --------------------------- INFO functions -----------------------------
    def _summary(self):
        summary = []
        if os.path.isfile(self._getExtraPath("auto_clustering_results.csv")):
            summary.append("Analysis results (optimal cluster number):")
            df = pd.read_csv(self._getExtraPath("auto_clustering_results.csv"))
            for _, row in df.iterrows():
                summary.append(f"     - {row['Method']}: {row['Best_K']}")
        else:
            summary.append("Finding optimal number of clusters...")
        return summary

    def _methods(self):
        return [
            "Optimal cluster analysis based on the following methods: \n"
            "     - Gap statistic (gapStatistic)\n"
            "     - Elbow (elbow)\n"
            "     - Silhouette (silhouette)\n"
            "     - Calinski Harabasz (ch)\n"
            "     - Davies Bouldin (db)\n"
        ]
