# **************************************************************************
# *
# * Authors:  David Herreros (dherreros@cnb.csic.es)
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


import numpy as np
import os

from pyworkflow.viewer import DESKTOP_TKINTER, WEB_DJANGO, ProtocolViewer
import pyworkflow.protocol.params as params
from pyworkflow.utils.process import runJob

from flexutils.protocols.xmipp.protocol_structure_landscape import XmippProtStructureLanscapes
from flexutils.protocols.protocol_dimred import ProtFlexDimRedSpace

import flexutils.constants as const
from flexutils.utils import computeNormRows
import flexutils


class XmippReducedSpaceViewer(ProtocolViewer):
    """ Visualize reduced conformational space """
    _label = 'viewer reduced space - Zernike3D'
    _targets = [XmippProtStructureLanscapes, ProtFlexDimRedSpace]
    _environments = [DESKTOP_TKINTER, WEB_DJANGO]

    def __init__(self, **kwargs):
        ProtocolViewer.__init__(self, **kwargs)
        self._data = None

    def getData(self):
        if self._data is None:
            self._data = self.loadData()
        return self._data


    def _defineParams(self, form):
        form.addSection(label='Show reduced conformational space')
        form.addParam('doShowSpace', params.LabelParam,
                      label="Display the reduced conformational space")

    def _getVisualizeDict(self):
        # self.protocol._createFilenameTemplates()
        return {'doShowSpace': self._doShowSpace}

    def _doShowSpace(self, param=None):
        red_space = []
        z_clnm = []
        for particle in self.protocol.outputParticles.iterItems():
            z_clnm.append(np.fromstring(particle._xmipp_sphCoefficients.get(), sep=","))
            red_space.append(np.fromstring(particle._red_space.get(), sep=","))
        z_clnm = np.asarray(z_clnm)
        red_space = np.asarray(red_space)

        # Generate files to call command line
        file_red_space = self.protocol._getExtraPath("red_coords.txt")
        file_deformation = self.protocol._getExtraPath("deformation.txt")
        np.savetxt(file_red_space, red_space)

        deformation = computeNormRows(z_clnm)

        # Generate files to call command line
        np.savetxt(file_deformation, deformation)

        # Run slicer
        args = "--coords %s --deformation %s" \
               % (file_red_space, file_deformation)
        program = os.path.join(const.VIEWERS, "viewer_point_cloud.py")
        program = flexutils.Plugin.getProgram(program)
        runJob(None, program, args)