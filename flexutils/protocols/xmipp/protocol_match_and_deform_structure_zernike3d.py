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
from pwem.protocols import ProtAnalysis3D
from pwem.objects import AtomStruct

import pyworkflow.protocol.params as params
import pyworkflow.utils as pwutils
from pyworkflow.object import Float, Integer

import flexutils
from flexutils.utils import readZernikeFile
import flexutils.constants as const


class XmippMatchDeformSructZernike3D(ProtAnalysis3D):
    """ Zernike3D deformation field computation between two structures based on atomic matches"""
    _label = 'structure match and deform - Zernike3D'

    # --------------------------- DEFINE param functions --------------------------------------------
    def _defineParams(self, form):
        form.addSection(label='Input')
        form.addParam('input', params.PointerParam, label="Input structure",
                      pointerClass='AtomStruct',
                      help="Structure to be deformed")
        form.addParam('reference', params.PointerParam, label="Reference structure",
                      pointerClass='AtomStruct',
                      help="Target structure")
        form.addParam('l1', params.IntParam, default=7,
                      label='Zernike Degree',
                      expertLevel=params.LEVEL_ADVANCED,
                      help='Degree Zernike Polynomials of the deformation=1,2,3,...')
        form.addParam('l2', params.IntParam, default=7,
                      label='Harmonical Degree',
                      expertLevel=params.LEVEL_ADVANCED,
                      help='Degree Spherical Harmonics of the deformation=1,2,3,...')

    # --------------------------- INSERT steps functions -----------------------
    def _insertAllSteps(self):
        self._insertFunctionStep("deformStep")
        self._insertFunctionStep("createOutputStep")

    # --------------------------- STEPS functions ------------------------------
    def deformStep(self):
        input = self.input.get().getFileName()
        reference = self.reference.get().getFileName()
        output = self._getExtraPath("structure_deformed.pdb")
        l1 = self.l1.get()
        l2 = self.l2.get()
        args = "--i %s --r %s --o %s --l1 %d --l2 %d " \
               % (input, reference, output, l1, l2)
        program = os.path.join(const.XMIPP_SCRIPTS, "find_z_clnm_structure.py")
        program = flexutils.Plugin.getProgram(program)
        self.runJob(program, args)

    def createOutputStep(self):
        z_clnm_file = self._getExtraPath("z_clnm.txt")
        basis_params, z_clnm = readZernikeFile(z_clnm_file)
        rmsd_def = np.loadtxt(self._getExtraPath("rmsd_def.txt"))

        L1 = Integer(self.l1.get())
        L2 = Integer(self.l2.get())
        Rmax = Float(basis_params[2])
        rmsd = Float(rmsd_def[1])
        deformation = Float(rmsd_def[2])

        outFile = self._getExtraPath("structure_deformed.pdb")
        pdb = AtomStruct(outFile)
        pdb.L1 = L1
        pdb.L2 = L2
        pdb.Rmax = Rmax
        pdb.rmsd = rmsd
        pdb.deformation = deformation
        self._defineOutputs(deformedStructure=pdb)
        self._defineSourceRelation(self.input, pdb)
        self._defineSourceRelation(self.reference, pdb)

    # --------------------------- UTILS functions ------------------------------
    def writeZernikeFile(self, file):
        volume = self.volume.get()
        L1 = volume.L1.get()
        L2 = volume.L2.get()
        Rmax = volume.getSamplingRate() * volume.Rmax.get() if self.applyPDB.get() else volume.Rmax.get()
        z_clnm = volume._xmipp_sphCoefficients.get()
        with open(file, 'w') as fid:
            fid.write(' '.join(map(str, [L1, L2, Rmax])) + "\n")
            fid.write(z_clnm.replace(",", " ") + "\n")
