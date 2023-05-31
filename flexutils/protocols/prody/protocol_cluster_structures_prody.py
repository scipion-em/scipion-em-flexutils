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

        args = "--ensemble % s --pdb %s --distThr %f --odir %s" \
               % (self.ensemble.get().getFirstItem().getFileName(), 
                  self.structure.get().getFileName(), 
                  self.dist.get(), self._getExtraPath())
        program = os.path.join(const.PRODY_SCRIPTS, "structure_rmsd_clustering.py")
        program = flexutils.Plugin.getProgram(program)
        self.runJob(program, args)


    def createOutputStep(self):
        frames = self.ensemble.get()
        frameIds = list(frames.getIdSet())
        rep_path = self._getExtraPath("clustered_structures/")
        img_rep_path = self._getExtraPath("cluster_img_ids/")
        num_classes = len([name for name in os.listdir(rep_path) if os.path.isfile(os.path.join(rep_path, name))])

        # Create SetOfClasses3D
        suffix = getOutputSuffix(self, SetOfClassesStructFrameFlex)
        classes = self._createSetOfClassesStructFrameFlex(frames, suffix=suffix,
                                                          progName=const.PRODY)

        # Populate SetOfClassesStructFrameFlex with structures from KMean frames
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
