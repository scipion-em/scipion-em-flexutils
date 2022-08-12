# **************************************************************************
# *
# * Authors:  David Herreros Calero (dherreros@cnb.csic.es)
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

import pyworkflow.protocol.params as params
from pyworkflow.viewer import DESKTOP_TKINTER, WEB_DJANGO, ProtocolViewer

from pwem.viewers import ChimeraView

from flexutils.protocols.xmipp.protocol_match_and_deform_structure_zernike3d import XmippMatchDeformSructZernike3D


class FlexShowStructuresViewer(ProtocolViewer):
    """ Show structures using ChimeraX """
    _label = 'show structures ChimeraX'
    _targets = [XmippMatchDeformSructZernike3D]
    _environments = [DESKTOP_TKINTER, WEB_DJANGO]
    OPEN_FILE = "open %s\n"
    VIEW = "view\n"

    def _defineParams(self, form):
        form.addSection(label='Show deformation')
        form.addParam('doShowStructures', params.LabelParam,
                      label="Display the strucutures")

    def _getVisualizeDict(self):
        return {'doShowStructures': self._doShowStructures}

    def _doShowStructures(self, param=None):
        # Get shortest path
        input = os.path.abspath(self.protocol.input.get().getFileName())
        reference = os.path.abspath(self.protocol.reference.get().getFileName())
        output = os.path.abspath(self.protocol.deformedStructure.getFileName())

        scriptFile = self.protocol._getPath('show_structures_chimera.cxc')
        fhCmd = open(scriptFile, 'w')

        fhCmd.write(self.OPEN_FILE % input)
        fhCmd.write(self.OPEN_FILE % reference)
        fhCmd.write(self.OPEN_FILE % output)
        fhCmd.write("cartoon style #1-3 width 3 thickness 1.5\n")

        view = ChimeraView(scriptFile)
        return [view]
