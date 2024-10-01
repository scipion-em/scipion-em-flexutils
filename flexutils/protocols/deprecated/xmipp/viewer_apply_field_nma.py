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

from pyworkflow.object import Set
from pyworkflow.utils import removeExt
from pyworkflow.viewer import DESKTOP_TKINTER, WEB_DJANGO, ProtocolViewer
import pyworkflow.protocol.params as params

from pwem.viewers import ChimeraView

from flexutils.protocols.xmipp.protocol_apply_field_nma import XmippApplyFieldNMA


class XmippApplyFieldNMAView(ProtocolViewer):
    """ Visualize a NMA structure """
    _label = 'viewer apply field NMA'
    _targets = [XmippApplyFieldNMA]
    _environments = [DESKTOP_TKINTER, WEB_DJANGO]
    OPEN_FILE = "open %s\n"
    VIEW = "view\n"

    def __init__(self, **kwargs):
        ProtocolViewer.__init__(self, **kwargs)
        self.only_apply = True
        self.deformed = self.protocol.deformed
        self.have_set = isinstance(self.deformed, Set)
        self.choices = list(self.deformed.getIdSet())

    def _defineParams(self, form):
        form.addSection(label='Show deformation')
        form.addParam('idChoice', params.EnumParam,
                      condition='have_set',
                      choices=[str(idx) for idx in self.choices], default=0,
                      label='Structure to display', display=params.EnumParam.DISPLAY_COMBO,
                      help='Select which strucutre to display from the IDs of the set')
        form.addParam('doShowStrain', params.LabelParam,
                      label="Display the strain deformation")
        form.addParam('doShowRotation', params.LabelParam,
                      label="Display the rotation deformation")
        form.addParam('doShowPDB', params.LabelParam,
                      label="Display original and deformed PDB or volume")
        form.addParam('doShowMorph', params.LabelParam,
                      label="Display a morphing between the original and deformed PDB or volume")

    def _getVisualizeDict(self):
        if self.have_set:
            self.chosen = self.deformed[self.choices[self.idChoice.get()]]
        else:
            self.chosen = self.deformed

        return {'doShowStrain': self._doShowStrainStruct,
                'doShowRotation': self._doShowRotationStruct,
                'doShowPDB': self._doShowPDB,
                'doShowMorph': self._doShowMorph}

    # ------------------- Mode Structure Methods -------------------
    def _doShowStrainStruct(self, param=None):
        scriptFile = self.protocol._getPath('strain_chimera.cxc')
        fhCmd = open(scriptFile, 'w')

        fnbase = removeExt(self.chosen.getFileName())
        fnStrain = os.path.abspath(fnbase)

        fhCmd.write(self.OPEN_FILE % (fnStrain + "_strain.pdb"))
        fhCmd.write("show cartoons\n")
        fhCmd.write("cartoon style width 1.5 thick 1.5\n")
        fhCmd.write("style stick\n")
        fhCmd.write('color by occupancy palette rainbow\n')
        fhCmd.write(self.VIEW)
        fhCmd.close()

        view = ChimeraView(scriptFile)
        return [view]

    def _doShowRotationStruct(self, param=None):
        scriptFile = self.protocol._getPath('strain_chimera.cxc')
        fhCmd = open(scriptFile, 'w')

        fnbase = removeExt(self.chosen.getFileName())
        fnStrain = os.path.abspath(fnbase)

        fhCmd.write(self.OPEN_FILE % (fnStrain + "_rotation.pdb"))
        fhCmd.write("show cartoons\n")
        fhCmd.write("cartoon style width 1.5 thick 1.5\n")
        fhCmd.write("style stick\n")
        fhCmd.write('color by occupancy palette rainbow\n')
        fhCmd.write(self.VIEW)
        fhCmd.close()

        view = ChimeraView(scriptFile)
        return [view]

    def _doShowPDB(self, obj, **kwargs):
        scriptFile = self.protocol._getPath('pdb_deform_chimera.cxc')
        fhCmd = open(scriptFile, 'w')
        inputFile = self.protocol.inputStruct.get().getFirstItem().getFlexInfo().refStruct.get()
        inputFile = os.path.abspath(inputFile)
        outputFile = os.path.abspath(self.chosen.getFileName())

        fhCmd.write(self.OPEN_FILE % inputFile)
        fhCmd.write(self.OPEN_FILE % outputFile)
        # fhCmd.write("start Model Panel\n")
        fhCmd.write("show cartoons\n")
        fhCmd.write("cartoon style width 1.5 thick 1.5\n")
        fhCmd.write("style stick\n")
        fhCmd.write("color bymodel\n")
        fhCmd.close()

        view = ChimeraView(scriptFile)
        return [view]

    def _doShowMorph(self, obj, **kwargs):
        scriptFile = self.protocol._getPath('pdb_deform_chimera.cxc')
        fhCmd = open(scriptFile, 'w')
        inputFile = self.protocol.inputStruct.get().getFirstItem().getFlexInfo().refStruct.get()
        inputFile = os.path.abspath(inputFile)
        outputFile = os.path.abspath(self.chosen.getFileName())

        fhCmd.write(self.OPEN_FILE % inputFile)
        fhCmd.write(self.OPEN_FILE % outputFile)
        fhCmd.write("hide models\n")
        fhCmd.write("morph #1,2 frames 50 play false\n")
        fhCmd.write("coordset #3 1,\n")
        fhCmd.write("wait 50\n")
        fhCmd.write("coordset #3 50,1\n")
        fhCmd.close()

        view = ChimeraView(scriptFile)
        return [view]
    # ------------------- ------------------- -------------------

