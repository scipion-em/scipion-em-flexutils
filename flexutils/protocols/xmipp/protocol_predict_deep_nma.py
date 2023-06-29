# **************************************************************************
# *
# * Authors:     David Herreros Calero (dherreos@cnb.csic.es)
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
import re
from xmipp_metadata.metadata import XmippMetaData
import prody as pd

import pyworkflow.protocol.params as params
from pyworkflow.object import Integer, Float, String, CsvList, Boolean
from pyworkflow.utils.path import moveFile
from pyworkflow import VERSION_2_0

from pwem.protocols import ProtAnalysis3D
import pwem.emlib.metadata as md
from pwem.constants import ALIGN_PROJ

from xmipp3.convert import createItemMatrix, setXmippAttributes, writeSetOfParticles, \
                           geometryFromMatrix, matrixFromGeometry
import xmipp3

import flexutils
from flexutils.protocols import ProtFlexBase
from flexutils.objects import ParticleFlex
import flexutils.constants as const


class TensorflowProtPredictDeepNMA(ProtAnalysis3D, ProtFlexBase):
    """ Predict NMA coefficents for a set of particles based on a trained
     DeepNMA network. """
    _label = 'predict - DeepNMA'
    _lastUpdateVersion = VERSION_2_0
    _subset = ["ca", "bb", None]

    # --------------------------- DEFINE param functions -----------------------
    def _defineParams(self, form):
        form.addSection(label='Input')
        form.addHidden(params.USE_GPU, params.BooleanParam, default=True,
                       label="Use GPU for execution",
                       help="This protocol has both CPU and GPU implementation.\
                             Select the one you want to use.")
        form.addHidden(params.GPU_LIST, params.StringParam, default='0',
                       expertLevel=params.LEVEL_ADVANCED,
                       label="Choose GPU IDs",
                       help="Add a list of GPU devices that can be used")
        group = form.addGroup("Data")
        group.addParam('inputParticles', params.PointerParam, label="Input particles to predict",
                      pointerClass='SetOfParticles')
        group.addParam('deepNMAProtocol', params.PointerParam, label="DeepNMA trained network",
                       pointerClass='TensorflowProtAngularAlignmentDeepNMA',
                       help="Previously executed 'angular align - DeepNMA'. "
                            "This will allow to load the network trained in that protocol to be used during "
                            "the prediction")
        form.addParallelSection(threads=4, mpi=0)

    def _createFilenameTemplates(self):
        """ Centralize how files are called """
        myDict = {
            'imgsFn': self._getExtraPath('input_particles.xmd'),
            'fnVol': self._getExtraPath('volume.mrc'),
            'fnVolMask': self._getExtraPath('mask.mrc'),
            'fnStruct': self._getExtraPath('structure.txt'),
            'fnOutDir': self._getExtraPath()
        }
        self._updateFilenamesDict(myDict)

    # --------------------------- INSERT steps functions -----------------------
    def _insertAllSteps(self):
        self._createFilenameTemplates()
        self._insertFunctionStep(self.writeMetaDataStep)
        self._insertFunctionStep(self.predictStep)
        self._insertFunctionStep(self.createOutputStep)

    # --------------------------- STEPS functions -----------------------
    def writeMetaDataStep(self):
        imgsFn = self._getFileName('imgsFn')
        structure = self._getFileName('fnStruct')

        inputParticles = self.inputParticles.get()
        deepNMAProtocol = self.deepNMAProtocol.get()
        Xdim = inputParticles.getXDim()
        self.newXdim = deepNMAProtocol.boxSize.get()
        refStruct = deepNMAProtocol.inputStruct.get().getFileName()
        subset = self._subset[deepNMAProtocol.atomSubset.get()]

        pd_struct = pd.parsePDB(refStruct, subset=subset, compressed=False)
        pdb_coordinates = pd_struct.getCoords()
        pdb_coordinates = np.c_[pdb_coordinates, np.ones(pdb_coordinates.shape[0])]
        # pdb_lines = self.readPDB(deepNMAProtocol.inputStruct.get().getFileName())
        # pdb_coordinates = np.array(self.PDB2List(pdb_lines))
        np.savetxt(structure, pdb_coordinates)

        writeSetOfParticles(inputParticles, imgsFn)

        if self.newXdim != Xdim:
            params = "-i %s -o %s --save_metadata_stack %s --fourier %d" % \
                     (imgsFn,
                      self._getExtraPath('scaled_particles.stk'),
                      self._getExtraPath('scaled_particles.xmd'),
                      self.newXdim)
            if self.numberOfMpi.get() > 1:
                params += " --mpi_job_size %d" % int(inputParticles.getSize() / self.numberOfMpi.get())
            self.runJob("xmipp_image_resize", params, numberOfMpi=self.numberOfMpi.get(),
                        env=xmipp3.Plugin.getEnviron())
            moveFile(self._getExtraPath('scaled_particles.xmd'), imgsFn)

    def predictStep(self):
        deepNMAProtocol = self.deepNMAProtocol.get()
        md_file = self._getFileName('imgsFn')
        weigths_file = deepNMAProtocol._getExtraPath(os.path.join('network', 'deep_nma_model'))
        n_modes = deepNMAProtocol.n_modes.get()
        pad = deepNMAProtocol.pad.get()
        correctionFactor = self.inputParticles.get().getXDim() / deepNMAProtocol.boxSize.get()
        sr = correctionFactor * self.inputParticles.get().getSamplingRate()
        applyCTF = deepNMAProtocol.ctfType.get()
        args = "--md_file %s --weigths_file %s --n_modes %d " \
               "--pad %d --sr %f --apply_ctf %d" \
               % (md_file, weigths_file, n_modes, pad, sr, applyCTF)

        if deepNMAProtocol.refinePose.get():
            args += " --refine_pose"

        if deepNMAProtocol.ctfType.get() == 0:
            args += " --ctf_type apply"
        elif deepNMAProtocol.ctfType.get() == 1:
            args += " --ctf_type wiener"

        if deepNMAProtocol.architecture.get() == 0:
            args += " --architecture convnn"
        elif deepNMAProtocol.architecture.get() == 1:
            args += " --architecture mlpnn"


        if self.useGpu.get():
            gpu_list = ','.join([str(elem) for elem in self.getGpuList()])
            args += " --gpu %s" % gpu_list
        program = flexutils.Plugin.getTensorflowProgram("predict_deep_nma.py", python=False)
        self.runJob(program, args, numberOfMpi=1)

    def createOutputStep(self):
        inputParticles = self.inputParticles.get()
        deepNMAProtocol = self.deepNMAProtocol.get()
        Xdim = inputParticles.getXDim()
        self.newXdim = deepNMAProtocol.boxSize.get()
        model_path = deepNMAProtocol._getExtraPath(os.path.join('network', 'deep_nma_model'))
        md_file = self._getFileName('imgsFn')

        metadata = XmippMetaData(md_file)
        nma_space = np.asarray([np.fromstring(item, sep=',') for item in metadata[:, 'nmaCoefficients']])

        if metadata.isMetaDataLabel('delta_angle_rot'):
            delta_rot = metadata[:, 'delta_angle_rot']
            delta_tilt = metadata[:, 'delta_angle_tilt']
            delta_psi = metadata[:, 'delta_angle_psi']
            delta_shift_x = metadata[:, 'delta_shift_x']
            delta_shift_y = metadata[:, 'delta_shift_y']

        refinePose = deepNMAProtocol.refinePose.get()

        inputSet = self.inputParticles.get()
        partSet = self._createSetOfParticlesFlex(progName=const.NMA)
        partSet.setHasCTF(inputSet.hasCTF())

        partSet.copyInfo(inputSet)
        partSet.setAlignmentProj()

        correctionFactor = Xdim / self.newXdim

        inverseTransform = partSet.getAlignment() == ALIGN_PROJ

        for idx, particle in enumerate(inputSet.iterItems()):

            outParticle = ParticleFlex(progName=const.NMA)
            outParticle.copyInfo(particle)

            outParticle.setZFlex(nma_space[idx])

            if refinePose:
                tr_ori = particle.getTransform().getMatrix()
                shifts, angles = geometryFromMatrix(tr_ori, inverseTransform)

                # Apply delta angles
                angles[0] += delta_rot[idx]
                angles[1] += delta_tilt[idx]
                angles[2] += delta_psi[idx]

                # Apply delta shifts
                shifts[0] += correctionFactor * delta_shift_x[idx]
                shifts[1] += correctionFactor * delta_shift_y[idx]

                # Set new transformation matrix
                tr = matrixFromGeometry(shifts, angles, inverseTransform)
                outParticle.getTransform().setMatrix(tr)

            partSet.append(outParticle)

        partSet.getFlexInfo().n_modes = Integer(deepNMAProtocol.n_modes.get())
        partSet.getFlexInfo().modelPath = String(model_path)
        partSet.getFlexInfo().atomSubset = String(self._subset[deepNMAProtocol.atomSubset.get()])

        structure = deepNMAProtocol.inputStruct.get().getFileName()
        partSet.getFlexInfo().refStruct = String(structure)

        partSet.getFlexInfo().refPose = refinePose

        self._defineOutputs(outputParticles=partSet)
        self._defineTransformRelation(self.inputParticles, partSet)

    # --------------------------- UTILS functions -----------------------
    def _updateParticle(self, item, row):
        setXmippAttributes(item, row, md.MDL_ANGLE_ROT, md.MDL_ANGLE_TILT,
                           md.MDL_ANGLE_PSI, md.MDL_SHIFT_X, md.MDL_SHIFT_Y,
                           md.MDL_FLIP)
        createItemMatrix(item, row, align=ALIGN_PROJ)

    def getInputParticles(self):
        return self.inputParticles.get()

    def readPDB(self, fnIn):
        with open(fnIn) as f:
            lines = f.readlines()
        return lines

    def PDB2List(self, lines):
        deepNMAProtocol = self.deepNMAProtocol.get()
        newlines = []
        for line in lines:
            eval = re.search(r'^ATOM\s+\d+\s+/N|CA|C|O/\s+', line) \
                if deepNMAProtocol.onlyBackbone.get() else line.startswith("ATOM ")
            if eval:
                try:
                    x = float(line[30:38])
                    y = float(line[38:46])
                    z = float(line[46:54])
                    val = float(line[54:60])
                    newline = [x, y, z, val]
                    newlines.append(newline)
                except:
                    pass
        return newlines

    # ----------------------- VALIDATE functions ----------------------------------------
    def validate(self):
        """ Try to find errors on define params. """
        errors = []
        return errors
