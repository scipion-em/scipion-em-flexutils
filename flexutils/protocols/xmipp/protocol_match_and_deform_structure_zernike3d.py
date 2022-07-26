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
        deformation = Float(rmsd_def[2])

        print("Deformation = %f" % rmsd_def[2])

        if self.volume.get():
            outVol = self.volume.get().copy()
            z_clnm_vol = z_clnm[0] / outVol.getSamplingRate()
            outVol.L1 = L1
            outVol.L2 = L2
            outVol.Rmax = Rmax
            outVol._xmipp_sphDeformation = deformation
            outVol._xmipp_sphCoefficients = String(','.join(['%f' % c for c in z_clnm_vol]))
            if self.mask.get():
                outVol.refMask = String(self.mask.get().getFileName())
            self._defineOutputs(deformedStructure=outVol)
            self._defineSourceRelation(self.input, outVol)
            self._defineSourceRelation(self.reference, outVol)
        else:
            outFile = self._getExtraPath("structure_deformed.pdb")
            pdb = AtomStruct(outFile)
            pdb.L1 = L1
            pdb.L2 = L2
            pdb.Rmax = Rmax
            pdb._xmipp_sphDeformation = deformation
            pdb._xmipp_sphCoefficients = String(','.join(['%f' % c for c in z_clnm[0]]))
            self._defineOutputs(deformedStructure=pdb)
            self._defineSourceRelation(self.input, pdb)
            self._defineSourceRelation(self.reference, pdb)

    # --------------------------- UTILS functions ------------------------------
