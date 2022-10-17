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
from pwem.objects import Volume

import pyworkflow.protocol.params as params
import pyworkflow.utils as pwutils
from pyworkflow.object import Float, Integer, String

import flexutils
from flexutils.utils import readZernikeFile, getXmippFileName
import flexutils.constants as const


class XmippMatchDeformMapZernike3D(ProtAnalysis3D):
    """ Zernike3D deformation field computation between two map based on voxel matches"""
    _label = 'map match and deform - Zernike3D'
    match_method = ["skeleton", "border"]

    # --------------------------- DEFINE param functions --------------------------------------------
    def _defineParams(self, form):
        form.addSection(label='Input')
        form.addParam('input', params.PointerParam, label="Input map",
                      pointerClass='Volume',
                      help="Map to be deformed")
        form.addParam('reference', params.PointerParam, label="Reference map",
                      pointerClass='Volume',
                      help="Target map")
        form.addParam('l1', params.IntParam, default=7,
                      label='Zernike Degree',
                      expertLevel=params.LEVEL_ADVANCED,
                      help='Degree Zernike Polynomials of the deformation=1,2,3,...')
        form.addParam('l2', params.IntParam, default=7,
                      label='Harmonical Degree',
                      expertLevel=params.LEVEL_ADVANCED,
                      help='Degree Spherical Harmonics of the deformation=1,2,3,...')
        form.addParam('matchMode', params.EnumParam, choices=self.match_method, default=0,
                      label='Voxel matching algorithm', display=params.EnumParam.DISPLAY_HLIST,
                      expertLevel=params.LEVEL_ADVANCED,
                      help='- Skeleton mode: faster and works well on maps presenting similar conformations.\n'
                           '- Border mode: slower but able to find matches when conformations are not similar')
        form.addParam('gsSteps', params.IntParam,
                      allowsNull=True,
                      label='Global registration steps',
                      expertLevel=params.LEVEL_ADVANCED,
                      help='Perform a global registration of the maps before computing the deformation fields')

    # --------------------------- INSERT steps functions -----------------------
    def _insertAllSteps(self):
        self._insertFunctionStep("deformStep")
        self._insertFunctionStep("createOutputStep")

    # --------------------------- STEPS functions ------------------------------
    def deformStep(self):
        input = getXmippFileName(self.input.get().getFileName())
        reference = getXmippFileName(self.reference.get().getFileName())
        output = self._getExtraPath("map_deformed.mrc")
        l1 = self.l1.get()
        l2 = self.l2.get()
        args = "--i %s --r %s --o %s --l1 %d --l2 %d --%s" \
               % (input, reference, output, l1, l2, self.match_method[self.matchMode.get()])
        if self.gsSteps.get():
            args += " --gs %d" % self.gsSteps.get()
        program = os.path.join(const.XMIPP_SCRIPTS, "find_z_clnm_map.py")
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
        # mask = String(self._getExtraPath("mask.mrc"))

        outFile = self._getExtraPath("map_deformed.mrc")
        vol = Volume()
        vol.setLocation(outFile)
        vol.setSamplingRate(self.input.get().getSamplingRate())
        vol.L1 = L1
        vol.L2 = L2
        vol.Rmax = Rmax
        vol._xmipp_sphDeformation = deformation
        vol._xmipp_sphCoefficients = String(','.join(['%f' % c for c in z_clnm[0]]))
        # vol.refMask = mask
        self._defineOutputs(deformedMap=vol)
        self._defineSourceRelation(self.input, vol)
        self._defineSourceRelation(self.reference, vol)

    # --------------------------- UTILS functions ------------------------------
