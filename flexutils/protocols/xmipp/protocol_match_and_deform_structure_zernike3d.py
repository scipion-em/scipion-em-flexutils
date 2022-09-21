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
import glob

import numpy as np
from pwem.protocols import ProtAnalysis3D
from pwem.objects import AtomStruct, SetOfAtomStructs, Volume

import pyworkflow.protocol.params as params
import pyworkflow.utils as pwutils
from pyworkflow.object import Float, Integer, String

import flexutils
from flexutils.utils import readZernikeFile
import flexutils.constants as const


class XmippMatchDeformSructZernike3D(ProtAnalysis3D):
    """ Zernike3D deformation field computation between two structures based on atomic matches"""
    _label = 'structure match and deform - Zernike3D'

    # --------------------------- DEFINE param functions --------------------------------------------
    def _defineParams(self, form):
        form.addSection(label='Input')
        form.addParam('inputs', params.MultiPointerParam, label="Input structures",
                      pointerClass='AtomStruct, SetOfAtomStructs',
                      help="Target structures")
        form.addParam('reference', params.PointerParam, label="Reference structure",
                      pointerClass='AtomStruct',
                      help="Structure to be deformed")
        form.addParam('l1', params.IntParam, default=3,
                      label='Zernike Degree',
                      expertLevel=params.LEVEL_ADVANCED,
                      help='Degree Zernike Polynomials of the deformation=1,2,3,...')
        form.addParam('l2', params.IntParam, default=2,
                      label='Harmonical Degree',
                      expertLevel=params.LEVEL_ADVANCED,
                      help='Degree Spherical Harmonics of the deformation=1,2,3,...')
        form.addParam('volume', params.PointerParam, label="Reference structure map",
                      pointerClass='Volume',
                      allowsNull=True,
                      expertLevel=params.LEVEL_ADVANCED,
                      help="Computed coefficients will be associated to this map instead to the "
                           "structure")
        form.addParam('mask', params.PointerParam, label="Reference map mask",
                      pointerClass='VolumeMask',
                      allowsNull=True,
                      expertLevel=params.LEVEL_ADVANCED,
                      help="Mask to associate to reference volume")
        form.addParam('moveBoxOrigin', params.BooleanParam, default=False, condition="volume",
                      label="Move structure to box origin?",
                      help="If PDB has been aligned inside Scipion, set to False. Otherwise, this option will "
                           "correctly place the PDB in the origin of the volume.")

    # --------------------------- INSERT steps functions -----------------------
    def _insertAllSteps(self):
        self._insertFunctionStep("deformStep")
        self._insertFunctionStep("createOutputStep")

    # --------------------------- STEPS functions ------------------------------
    def deformStep(self):
        reference = self.reference.get().getFileName()
        l1 = self.l1.get()
        l2 = self.l2.get()
        if self.volume.get():
            volume = self.volume.get()
            rmax = int(0.5 * volume.getSamplingRate() * volume.getXDim())
        else:
            rmax = None

        input_files = []
        for pointer in self.inputs:
            obj = pointer.get()
            if isinstance(obj, AtomStruct):
                input_files.append(obj.getFileName())
            elif isinstance(obj, SetOfAtomStructs):
                for struct in obj:
                    input_files.append(struct.getFileName())

        for idp, input in enumerate(input_files):
            output = self._getExtraPath("structure_deformed_%d.pdb" % (idp + 1))
            args = "--i %s --r %s --o %s --l1 %d --l2 %d" \
                   % (reference, input, output, l1, l2)
            if rmax:
                args += " --rmax %d" % rmax
            if self.moveBoxOrigin.get() and rmax:
                args += " --d %d" % rmax
            elif not self.moveBoxOrigin.get() and rmax:
                args += " --d 0"
            program = os.path.join(const.XMIPP_SCRIPTS, "find_z_clnm_structure.py")
            program = flexutils.Plugin.getProgram(program)
            self.runJob(program, args)
            pwutils.moveFile(self._getExtraPath("z_clnm.txt"),
                             self._getExtraPath("z_clnm_%d.txt" % (idp + 1)))
            pwutils.moveFile(self._getExtraPath("rmsd_def.txt"),
                             self._getExtraPath("rmsd_def_%d.txt" % (idp + 1)))

    def createOutputStep(self):
        if self.volume.get():
            outputs = self._createSetOfVolumes()
            outputs.setSamplingRate(self.volume.get().getSamplingRate())
            Rmax = Float(int(0.5 * self.volume.get().getXDim()))
            outputs.refMap = String(self.volume.get().getFileName())
            outputs.refMask = String(self.mask.get().getFileName())
        else:
            outputs = SetOfAtomStructs().create(self._getPath())
            basis_params, _ = readZernikeFile(self._getExtraPath("z_clnm_1.txt"))
            Rmax = basis_params[2]

        L1 = Integer(self.l1.get())
        L2 = Integer(self.l2.get())

        outputs.L1 = L1
        outputs.L2 = L2
        outputs.Rmax = Rmax

        input_files = len(glob.glob(self._getExtraPath("z_clnm_*.txt")))
        for idp in range(input_files):
            z_clnm_file = self._getExtraPath("z_clnm_%d.txt" % (idp + 1))
            basis_params, z_clnm = readZernikeFile(z_clnm_file)
            rmsd_def = np.loadtxt(self._getExtraPath("rmsd_def_%d.txt" % (idp + 1)))
            deformation = Float(rmsd_def)

            print("Deformation for structure %d = %f" % (idp + 1, rmsd_def))

            if self.volume.get():
                output = Volume()
                output.setFileName(self.volume.get().getFileName())
                output.setSamplingRate(self.volume.get().getSamplingRate())
                z_clnm_vol = z_clnm[0] / output.getSamplingRate()
                output.L1 = L1
                output.L2 = L2
                output.Rmax = Rmax
                output._xmipp_sphDeformation = deformation
                output._xmipp_sphCoefficients = String(','.join(['%f' % c for c in z_clnm_vol]))
                output.refMap = String(output.getFileName())
                if self.mask.get():
                    output.refMask = String(self.mask.get().getFileName())
            else:
                outFile = self._getExtraPath("structure_deformed.pdb")
                output = AtomStruct(outFile)
                output.L1 = L1
                output.L2 = L2
                output.Rmax = Rmax
                output._xmipp_sphDeformation = deformation
                output._xmipp_sphCoefficients = String(','.join(['%f' % c for c in z_clnm[0]]))

            outputs.append(output)

        if self.volume.get():
            self._defineOutputs(zernikeVolumes=outputs)
        else:
            self._defineOutputs(zernikeStructures=outputs)
        self._defineSourceRelation(self.inputs, outputs)
        self._defineSourceRelation(self.reference, outputs)

    # --------------------------- UTILS functions ------------------------------
