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
