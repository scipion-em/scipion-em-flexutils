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

from flexutils.protocols.xmipp.protocol_match_and_deform_map_zernike3d import XmippMatchDeformMapZernike3D


class FlexShowMapsViewer(ProtocolViewer):
    """ Show maps using ChimeraX """
    _label = 'show maps ChimeraX'
    _targets = [XmippMatchDeformMapZernike3D]
    _environments = [DESKTOP_TKINTER, WEB_DJANGO]
    OPEN_FILE = "open %s\n"
    VIEW = "view\n"
    VOXEL_SIZE = "volume #%d voxelSize %f\n"

    def _defineParams(self, form):
        form.addSection(label='Show deformation')
        form.addParam('doShowMaps', params.LabelParam,
                      label="Display the maps")

    def _getVisualizeDict(self):
        return {'doShowMaps': self._doShowMaps}

    def _doShowMaps(self, param=None):
        # Get shortest path
        input_vol = self.protocol.input.get()
        reference_vol = self.protocol.reference.get()
        output_vol = self.protocol.deformedMap
        input = os.path.abspath(input_vol.getFileName())
        reference = os.path.abspath(reference_vol.getFileName())
        output = os.path.abspath(output_vol.getFileName())

        scriptFile = self.protocol._getPath('show_maps_chimera.cxc')
        fhCmd = open(scriptFile, 'w')

        fhCmd.write(self.OPEN_FILE % input)
        fhCmd.write(self.OPEN_FILE % reference)
        fhCmd.write(self.OPEN_FILE % output)
        fhCmd.write(self.VOXEL_SIZE % (1, input_vol.getSamplingRate()))
        fhCmd.write(self.VOXEL_SIZE % (2, reference_vol.getSamplingRate()))
        fhCmd.write(self.VOXEL_SIZE % (3, output_vol.getSamplingRate()))

        view = ChimeraView(scriptFile)
        return [view]
