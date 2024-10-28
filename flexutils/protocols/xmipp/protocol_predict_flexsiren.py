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
from glob import glob
from xmipp_metadata.metadata import XmippMetaData
from xmipp_metadata.image_handler import ImageHandler

import pyworkflow.protocol.params as params
from pyworkflow.object import Integer, Float, String, Boolean
from pyworkflow.utils.path import moveFile
from pyworkflow import VERSION_2_0

from pwem.protocols import ProtAnalysis3D, ProtFlexBase
import pwem.emlib.metadata as md
from pwem.constants import ALIGN_PROJ
from pwem.objects import ParticleFlex

from xmipp3.convert import createItemMatrix, setXmippAttributes, writeSetOfParticles, \
    geometryFromMatrix, matrixFromGeometry
import xmipp3

import flexutils
import flexutils.constants as const
from flexutils.utils import getXmippFileName
from flexutils.protocols.xmipp.utils.pdb_parser import AtomicModelParser


class TensorflowProtPredictFlexSIREN(ProtAnalysis3D, ProtFlexBase):
    """ Predict FlexSIREN coefficents for a set of particles based on a trained
     FlexSIREN network. """
    _label = 'predict - FlexSIREN'
    _lastUpdateVersion = VERSION_2_0

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
        group.addParam('flexsirenProtocol', params.PointerParam, label="FlexSIREN trained network",
                       pointerClass='TensorflowProtAngularAlignmentFlexSIREN',
                       help="Previously executed 'flexible align - FlexSIREN'. "
                            "This will allow to load the network trained in that protocol to be used during "
                            "the prediction")
        group = form.addGroup("Mask type")
        group.addParam('convertBinary', params.BooleanParam, default=False,
                       label="Associate field to binary mask?",
                       help="If a regions mask has been used to trained the FlexSIREN network, you "
                            "can set this parameter to yes to reassociate the deformation coefficients "
                            "to a binary mask generated from the regions mask.")
        form.addParallelSection(threads=4, mpi=0)

    def _createFilenameTemplates(self):
        """ Centralize how files are called """
        myDict = {
            'imgsFn': self._getExtraPath('input_particles.xmd'),
            'fnVol': self._getExtraPath('volume.mrc'),
            'fnVolMask': self._getExtraPath('mask.mrc'),
            'fnStruct': self._getExtraPath('structure.txt'),
            'fnBond': self._getExtraPath("bonds.txt"),
            'fnDihedral': self._getExtraPath("dihedrals.txt"),
            'fnCA': self._getExtraPath("ca_indices.txt"),
            'fnOutDir': self._getExtraPath()
        }
        self._updateFilenamesDict(myDict)

    # --------------------------- INSERT steps functions -----------------------
    def _insertAllSteps(self):
        self._createFilenameTemplates()
        self._insertFunctionStep(self.writeMetaDataStep)
        self._insertFunctionStep(self.predictStep)
        if self.convertBinary:
            self._insertFunctionStep(self.convertBinaryStep)
        self._insertFunctionStep(self.createOutputStep)

    # --------------------------- STEPS functions -----------------------
    def writeMetaDataStep(self):
        imgsFn = self._getFileName('imgsFn')
        fnVol = self._getFileName('fnVol')
        fnVolMask = self._getFileName('fnVolMask')
        md_file = self._getFileName('imgsFn')
        bonds_file = self._getFileName("fnBond")
        dihedrals_file = self._getFileName("fnDihedral")

        inputParticles = self.inputParticles.get()
        flexsirenProtocol = self.flexsirenProtocol.get()
        Xdim = inputParticles.getXDim()
        self.newXdim = flexsirenProtocol.boxSize.get()
        i_sr = 1. / inputParticles.getSamplingRate()

        ih = ImageHandler()
        inputVolume = flexsirenProtocol.inputVolume.get().getFileName()
        ih.convert(getXmippFileName(inputVolume), fnVol)
        curr_vol_dim = ImageHandler(getXmippFileName(inputVolume)).getDimensions()[-1]
        if curr_vol_dim != self.newXdim:
            self.runJob("xmipp_image_resize",
                        "-i %s --dim %d " % (fnVol, self.newXdim), numberOfMpi=1, env=xmipp3.Plugin.getEnviron())

        inputMask = flexsirenProtocol.inputVolumeMask.get().getFileName()
        if inputMask:
            ih.convert(getXmippFileName(inputMask), fnVolMask)
            curr_mask_dim = ImageHandler(getXmippFileName(inputMask)).getDimensions()[-1]
            if curr_mask_dim != self.newXdim:
                self.runJob("xmipp_image_resize",
                            "-i %s --dim %d --interp nearest" % (fnVolMask, self.newXdim), numberOfMpi=1,
                            env=xmipp3.Plugin.getEnviron())

        if flexsirenProtocol.referenceType.get() == 1:  # Structure reference
            inputVolume = flexsirenProtocol.inputVolume.get().getFileName()
            inputStruct = flexsirenProtocol.inputStruct.get().getFileName()
            structure_file = self._getFileName('fnStruct')
            connect_file = self._getFileName('fnConnect')
            ca_file = self._getFileName('fnCA')
            parser = AtomicModelParser(inputStruct, flexsirenProtocol._subset[flexsirenProtocol.atomSubset.get()])
            pdb_coordinates = parser.get_atom_coordinates()
            covalent = parser.get_covalent_bonds()
            dihedrals = parser.get_dihedral_angles()
            ca_indices = parser.get_ca_indices()
            pdb_coordinates *= i_sr
            ih = ImageHandler(getXmippFileName(inputVolume))
            vol = ih.getData()
            factor = 0.5 * ih.getDimensions()[-1]
            pdb_indices = np.round(pdb_coordinates + factor).astype(int)
            values = vol[pdb_indices[:, 2], pdb_indices[:, 1], pdb_indices[:, 0]]
            pdb_coordinates = np.c_[pdb_coordinates, values]
            np.savetxt(structure_file, pdb_coordinates)
            np.savetxt(bonds_file, covalent)
            np.savetxt(dihedrals_file, dihedrals)
            np.savetxt(ca_file, ca_indices)

        writeSetOfParticles(inputParticles, imgsFn)

        # Write extra attributes (if needed)
        md = XmippMetaData(md_file)
        if hasattr(inputParticles.getFirstItem(), "_xmipp_subtomo_labels"):
            labels = np.asarray([int(particle._xmipp_subtomo_labels) for particle in inputParticles.iterItems()])
            md[:, "subtomo_labels"] = labels
        md.write(md_file, overwrite=True)

        if self.newXdim != Xdim:
            params = "-i %s -o %s --save_metadata_stack %s --fourier %d" % \
                     (imgsFn,
                      self._getTmpPath('scaled_particles.stk'),
                      self._getExtraPath('scaled_particles.xmd'),
                      self.newXdim)
            if self.numberOfMpi.get() > 1:
                params += " --mpi_job_size %d" % int(inputParticles.getSize() / self.numberOfMpi.get())
            self.runJob("xmipp_image_resize", params, numberOfMpi=self.numberOfMpi.get(),
                        env=xmipp3.Plugin.getEnviron())
            moveFile(self._getExtraPath('scaled_particles.xmd'), imgsFn)

    def predictStep(self):
        flexsirenProtocol = self.flexsirenProtocol.get()
        md_file = self._getFileName('imgsFn')
        weigths_file = glob(flexsirenProtocol._getExtraPath(os.path.join('network', 'flexsiren_model*')))[0]
        latDim = flexsirenProtocol.latDim.get()
        pad = flexsirenProtocol.pad.get()
        correctionFactor = self.inputParticles.get().getXDim() / flexsirenProtocol.boxSize.get()
        sr = correctionFactor * self.inputParticles.get().getSamplingRate()
        applyCTF = flexsirenProtocol.applyCTF.get()
        disPose = flexsirenProtocol.disPose.get()
        disCTF = flexsirenProtocol.disCTF.get()
        args = "--md_file %s --weigths_file %s --lat_dim %d " \
               "--pad %d --sr %f --apply_ctf %d" \
               % (md_file, weigths_file, latDim, pad, sr, applyCTF)

        if flexsirenProtocol.refinePose.get():
            args += " --refine_pose"

        if flexsirenProtocol.ctfType.get() == 0:
            args += " --ctf_type apply"
        elif flexsirenProtocol.ctfType.get() == 1:
            args += " --ctf_type wiener"

        if flexsirenProtocol.architecture.get() == 0:
            args += " --architecture convnn"
        elif flexsirenProtocol.architecture.get() == 1:
            args += " --architecture mlpnn"

        if disPose:
            args += " --pose_reg 1.0"

        if disCTF:
            args += " --ctf_reg 1.0"

        if self.useGpu.get():
            gpu_list = ','.join([str(elem) for elem in self.getGpuList()])
            args += " --gpu %s" % gpu_list
        program = flexutils.Plugin.getTensorflowProgram("predict_flexsiren.py", python=False)
        self.runJob(program, args, numberOfMpi=1)

    def convertBinaryStep(self):
        flexsirenProtocol = self.flexsirenProtocol.get()
        maskRegOri = flexsirenProtocol.inputVolumeMask.get().getFileName()
        maskReg = self._getFileName('fnVolMask')
        maskBin = self._getExtraPath("binary_mask.mrc")
        maskBinOri = self._getExtraPath("binary_mask_ori.mrc")
        latDim = flexsirenProtocol.latDim.get()
        md_file = self._getFileName('imgsFn')

        # Convert mask to binary (original)
        data = ImageHandler(maskRegOri).getData()
        data[data > 0] = 1.0
        ImageHandler().write(data, maskBinOri, overwrite=True)

        # Convert mask to binary (downsampled)
        data = ImageHandler(maskReg).getData()
        data[data > 0] = 1.0
        boxsize = data.shape[0]
        ImageHandler().write(data, maskBin, overwrite=True)

    def createOutputStep(self):
        inputParticles = self.inputParticles.get()
        flexsirenProtocol = self.flexsirenProtocol.get()
        Xdim = inputParticles.getXDim()
        self.newXdim = flexsirenProtocol.boxSize.get()
        model_path = glob(flexsirenProtocol._getExtraPath(os.path.join('network', 'flexsiren_model*')))[0]
        md_file = self._getFileName('imgsFn')

        metadata = XmippMetaData(md_file)
        z_space = np.asarray([np.fromstring(item, sep=',') for item in metadata[:, 'zCoefficients']])

        if metadata.isMetaDataLabel('delta_angle_rot'):
            delta_rot = metadata[:, 'delta_angle_rot']
            delta_tilt = metadata[:, 'delta_angle_tilt']
            delta_psi = metadata[:, 'delta_angle_psi']
            delta_shift_x = metadata[:, 'delta_shift_x']
            delta_shift_y = metadata[:, 'delta_shift_y']

        refinePose = flexsirenProtocol.refinePose.get()

        inputSet = self.inputParticles.get()
        partSet = self._createSetOfParticlesFlex(progName=const.FLEXSIREN)
        partSet.setHasCTF(inputSet.hasCTF())

        partSet.copyInfo(inputSet)
        partSet.getFlexInfo().setProgName(const.FLEXSIREN)
        partSet.setAlignmentProj()

        correctionFactor = Xdim / self.newXdim

        inverseTransform = partSet.getAlignment() == ALIGN_PROJ

        idx = 0
        for particle in inputSet.iterItems():
            # z = correctionFactor * zernike_space[idx]

            outParticle = ParticleFlex(progName=const.FLEXSIREN)
            outParticle.copyInfo(particle)
            outParticle.getFlexInfo().setProgName(const.FLEXSIREN)

            outParticle.setZFlex(z_space[idx])

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

            idx += 1

        partSet.getFlexInfo().latDim = Integer(flexsirenProtocol.latDim.get())
        partSet.getFlexInfo().Rmax = Float(Xdim / 2)
        partSet.getFlexInfo().modelPath = String(model_path)
        partSet.getFlexInfo().disPose = Boolean(disPose)
        partSet.getFlexInfo().disCTF = Boolean(disCTF)

        inputMask = self._getExtraPath("binary_mask_ori.mrc") if self.convertBinary.get() \
            else flexsirenProtocol.inputVolumeMask.get().getFileName()
        inputVolume = flexsirenProtocol.inputVolume.get().getFileName()
        partSet.getFlexInfo().refMask = String(inputMask)
        partSet.getFlexInfo().refMap = String(inputVolume)

        partSet.getFlexInfo().refPose = refinePose

        self._defineOutputs(outputParticles=partSet)
        self._defineTransformRelation(self.inputParticles, partSet)

    # --------------------------- UTILS functions -----------------------
    def _updateParticle(self, item, row):
        setXmippAttributes(item, row, md.MDL_ANGLE_ROT, md.MDL_ANGLE_TILT,
                           md.MDL_ANGLE_PSI, md.MDL_SHIFT_X, md.MDL_SHIFT_Y,
                           md.MDL_FLIP, md.MDL_SPH_DEFORMATION,
                           md.MDL_SPH_COEFFICIENTS)
        createItemMatrix(item, row, align=ALIGN_PROJ)

    def getInputParticles(self):
        return self.inputParticles.get()

    def readPDB(self, fnIn):
        with open(fnIn) as f:
            lines = f.readlines()
        return lines

    def PDB2List(self, lines):
        flexsirenProtocol = self.flexsirenProtocol.get()
        newlines = []
        for line in lines:
            eval = re.search(r'^ATOM\s+\d+\s+/N|CA|C|O/\s+', line) \
                if flexsirenProtocol.onlyBackbone.get() else line.startswith("ATOM ")
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

    # ----------------------- VALIDATE functions -----------------------
    def validate(self):
        """ Try to find errors on define params. """
        errors = []
        return errors
