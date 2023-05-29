# **************************************************************************
# *
# * Authors:     David Herreros Calero (dherreros@cnb.csic.es)
# *              James Krieger (jmkrieger@cnb.csic.es)
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
from pyworkflow.protocol.params import PointerParam, FloatParam

from pwem.protocols import ProtAnalysis3D
from pwem.objects import SetOfTrajFrames

import flexutils
import flexutils.constants as const
from flexutils.utils import getOutputSuffix
from flexutils.objects import ClassStructFrameFlex, AtomStructFlex, SetOfClassesStructFrameFlex
from flexutils.protocols.protocol_base import ProtFlexBase

from tqdm import tqdm

import prody as pd


class ProDyProtClusterStructuresEnsemble(ProtAnalysis3D, ProtFlexBase):
    """ Automatic clustering at atomic structure level based on a threshold distance """

    _label = 'structure clustering - ensemble'
    _devStatus = BETA
    OUTPUT_PREFIX = 'clusteredStructures'

    # --------------------------- DEFINE param functions ----------------------
    def _defineParams(self, form):
        form.addSection(label='General parameters')
        form.addParam('ensemble', PointerParam, label="ensemble set",
                      pointerClass='SetOfTrajFrames', important=True,
                      help="Ensemble readable by ProDy in ens.npz or dcd format")
        form.addParam('structure', PointerParam, label="Atomic structure",
                      pointerClass="AtomStruct", allowsNull=True,
                      help="Reference atomic structure with the same number of Calpha atoms")
        form.addParam('dist', FloatParam, default=5., label="Distance threshold",
                      help="Determine the minimum RMSD between the representative structures "
                           "of each cluster. The number of clusters found by the algorithm will "
                           "depend on the distance threshold: the smaller the threshold, the larger "
                           "the number of clusters that will be retrieved. Units are in Angstroms.")

    # --------------------------- INSERT steps functions ----------------------
    def _insertAllSteps(self):
        self._insertFunctionStep(self.computeClusters)
        self._insertFunctionStep(self.createOutputStep)

    # --------------------------- STEPS functions -----------------------------
    def computeClusters(self):
        # Input data
        ensemble_fname = self.ensemble.get().getFirstItem().getFileName()
        if ensemble_fname.endswith('.dcd'):
            ensemble = pd.parseDCD(ensemble_fname)
        elif ensemble_fname.endswith('.ens.npz'):
            ensemble = pd.loadEnsemble(ensemble_fname)
 
        structure = pd.parsePDB(self.structure.get().getFileName(), 
                                subset='ca', compressed=False)
        dist_thr = self.dist.get()

        structure_coords = structure.getCoords()
        ensemble.setCoords(structure_coords)

        # Cluster generation
        clusters = []
        frames_cluster = []
        for idi, conf in enumerate(tqdm(ensemble)):
            coords_struct_d = conf.getCoords()
            if clusters:
                found = False
                for idx in range(len(clusters)):
                    d = np.sum((clusters[idx] - coords_struct_d) ** 2, axis=1)
                    rmsd = np.sqrt(np.sum(d) / d.size)
                    if rmsd < dist_thr:
                        clusters[idx] = clusters[idx] + ((coords_struct_d - clusters[idx]) / (len(frames_cluster[idx]) + 1))
                        frames_cluster[idx].append(idi)
                        found = True
                        break
                if not found:
                    clusters.append(coords_struct_d)
                    frames_cluster.append([idi])
            else:
                clusters.append(coords_struct_d)
                frames_cluster.append([idi])

        output_dir = self._getExtraPath()

        # Write clustered structures to output folder
        clustered_struct_folder = os.path.join(output_dir, "clustered_structures")
        os.mkdir(clustered_struct_folder)
        for sid, cluster_coords in enumerate(tqdm(clusters)):
            clustered_struct = structure.copy()
            clustered_struct.setCoords(cluster_coords)
            pd.writePDB(os.path.join(clustered_struct_folder, "cluster_%d.pdb" % sid), clustered_struct)

        # Write image ids for each cluster in folder
        cluster_ids_folder = os.path.join(output_dir, "cluster_img_ids")
        os.mkdir(cluster_ids_folder)
        for sid, vec_ids in enumerate(tqdm(frames_cluster)):
            np.savetxt(os.path.join(cluster_ids_folder, "cluster_%d.txt" % sid), np.asarray(vec_ids))

    def createOutputStep(self):
        frames = self.ensemble.get()
        frameIds = list(frames.getIdSet())
        rep_path = self._getExtraPath("clustered_structures/")
        img_rep_path = self._getExtraPath("cluster_img_ids/")
        num_classes = len([name for name in os.listdir(rep_path) if os.path.isfile(os.path.join(rep_path, name))])

        # Create SetOfClasses3D
        suffix = getOutputSuffix(self, SetOfClassesStructFrameFlex)
        classes = self._createSetOfClassesStructFlex(frames, suffix=suffix,
                                                     progName=const.PRODY)

        # Popoulate SetOfClassesStructFlex with structures from KMean frames
        for clInx in range(num_classes):

            newClass = ClassStructFrameFlex(progName=const.PRODY)
            newClass.copyInfo(frames)
            representative = AtomStructFlex(progName=const.PRODY)
            representative.setFileName(os.path.join(rep_path, "cluster_%d.pdb" % clInx))

            newClass.setRepresentative(representative)

            classes.append(newClass)

            enabledClass = classes[newClass.getObjId()]
            enabledClass.enableAppend()

            img_rep_ids = np.loadtxt(os.path.join(img_rep_path, "cluster_%d.txt" % clInx))

            if img_rep_ids.size > 1:
                for img_idx in img_rep_ids:
                    item = frames[frameIds[int(img_idx)]]
                    enabledClass.append(item)
            else:
                item = frames[frameIds[int(img_rep_ids)]]
                enabledClass.append(item)

            classes.update(enabledClass)

        # Save new output
        name = self.OUTPUT_PREFIX + suffix
        args = {}
        args[name] = classes
        self._defineOutputs(**args)
        self._defineSourceRelation(frames, classes)

    # --------------------------- INFO functions -----------------------------
    def _summary(self):
        summary = []
        if self.getOutputsSize() >= 1:
            for _, outClasses in self.iterOutputAttributes():
                summary.append("Output *%s*:" % outClasses.getNameId().split('.')[1])
                summary.append("    * Total number of clusters: *%s*" % outClasses.getSize())
        else:
            summary.append("Output clusters not ready yet")
        return summary

    def _methods(self):
        return [
            "Automatic RMSD based clustering of Zernike3D atomic structures",
        ]

    # ----------------------- VALIDATE functions ----------------------------------------
    def validate(self):
        """ Try to find errors on define params. """
        errors = []
        frames = self.ensemble.get()
        if isinstance(frames, SetOfTrajFrames):
            if not (frames.getFirstItem().getFileName().endswith('.dcd') or 
                    frames.getFirstItem().getFileName().endswith('.ens.npz')):
                errors.append("The set of frames is not related to a .dcd or .ens.npz file"
                              "readable with ProDy. Please, provide a set of frames "
                              "with the correct file format.")
        return errors
