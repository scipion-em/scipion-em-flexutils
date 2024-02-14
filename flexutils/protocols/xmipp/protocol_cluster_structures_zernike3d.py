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

from pyworkflow import NEW
from pyworkflow.protocol.params import PointerParam, FloatParam

from pwem.protocols import ProtAnalysis3D, ProtFlexBase
from pwem.objects import ClassStructFlex, AtomStructFlex, SetOfClassesStructFlex, SetOfParticlesFlex

import flexutils
import flexutils.constants as const
from flexutils.utils import getOutputSuffix


class XmippProtClusterStructuresZernike3D(ProtAnalysis3D, ProtFlexBase):
    """ Automatic clustering at atomic structure level based on a threshold distance """

    _label = 'structure clustering - Zernike3D'
    _devStatus = NEW
    OUTPUT_PREFIX = 'clusteredStructures'

    # --------------------------- DEFINE param functions ----------------------
    def _defineParams(self, form):
        form.addSection(label='General parameters')
        form.addParam('particles', PointerParam, label="Zernike3D particles",
                      pointerClass='SetOfParticlesFlex', important=True)
        form.addParam('structure', PointerParam, label="Atomic structure",
                      pointerClass="AtomStruct", allowsNull=True,
                      help="Reference atomic structure that should be traced (or at least "
                           "aligned) with the reference map used during the Zernike3D analysis")
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
        particles = self.particles.get()
        structure = self.structure.get().getFileName()
        dist_thr = self.dist.get()
        boxSize = particles.getXDim()
        sr = particles.getSamplingRate()
        L1 = particles.getFlexInfo().L1.get()
        L2 = particles.getFlexInfo().L2.get()

        # Retrieve Zernike3D coefficients
        z_space_part = [particle.getZFlex() for particle in particles.iterItems()]
        z_space_part = sr * np.asarray(z_space_part)
        z_space_file = self._getExtraPath("z_space.txt")
        np.savetxt(z_space_file, z_space_part)

        # Compute clusters
        args = "--z_space % s --pdb %s --boxSize %f --sr %f --distThr %f " \
               "--L1 %d --L2 %d --odir %s" \
               % (z_space_file, structure, boxSize, sr, dist_thr, L1, L2, self._getExtraPath())
        program = os.path.join(const.XMIPP_SCRIPTS, "structure_rmsd_clustering.py")
        program = flexutils.Plugin.getProgram(program)
        self.runJob(program, args)

    def createOutputStep(self):
        particles = self.particles.get()
        partIds = list(particles.getIdSet())
        rep_path = self._getExtraPath("clustered_structures/")
        img_rep_path = self._getExtraPath("cluster_img_ids/")
        num_classes = len([name for name in os.listdir(rep_path) if os.path.isfile(os.path.join(rep_path, name))])

        # Create SetOfClasses3D
        suffix = getOutputSuffix(self, SetOfClassesStructFlex)
        classes = self._createSetOfClassesStructFlex(particles, suffix=suffix,
                                                     progName=const.ZERNIKE3D)

        # Popoulate SetOfClassesStructFlex with KMean particles
        for clInx in range(num_classes):

            newClass = ClassStructFlex(progName=const.ZERNIKE3D)
            newClass.copyInfo(particles)
            newClass.setHasCTF(particles.hasCTF())
            representative = AtomStructFlex(progName=const.ZERNIKE3D)
            representative.setFileName(os.path.join(rep_path, "cluster_%d.pdb" % clInx))

            newClass.setRepresentative(representative)

            classes.append(newClass)

            enabledClass = classes[newClass.getObjId()]
            enabledClass.enableAppend()

            img_rep_ids = np.loadtxt(os.path.join(img_rep_path, "cluster_%d.txt" % clInx))

            if img_rep_ids.size > 1:
                for img_idx in img_rep_ids:
                    item = particles[partIds[int(img_idx)]]
                    enabledClass.append(item)
            else:
                item = particles[partIds[int(img_rep_ids)]]
                enabledClass.append(item)

            classes.update(enabledClass)

        # Save new output
        name = self.OUTPUT_PREFIX + suffix
        args = {}
        args[name] = classes
        self._defineOutputs(**args)
        self._defineSourceRelation(particles, classes)

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
        particles = self.particles.get()
        if isinstance(particles, SetOfParticlesFlex):
            if particles.getFlexInfo().getProgName() != const.ZERNIKE3D:
                errors.append("The flexibility information associated with the particles is not "
                              "coming from the Zernike3D algorithm. Please, provide a set of particles "
                              "with the correct flexibility information.")
        return errors
