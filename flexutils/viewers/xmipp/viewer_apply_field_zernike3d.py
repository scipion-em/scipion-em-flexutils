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
from pyworkflow.viewer import DESKTOP_TKINTER, WEB_DJANGO, ProtocolViewer
import pyworkflow.protocol.params as params
from pyworkflow.utils import removeExt

from pwem.viewers import ChimeraView
from pwem import objects

from flexutils.protocols.xmipp.protocol_apply_field_zernike3d import XmippApplyFieldZernike3D


class XmippApplyFieldZernike3DView(ProtocolViewer):
    """ Visualize a Zernike3D map/structure """
    _label = 'viewer apply field Zernike3D'
    # _targets = [XmippProtVolumeDeformZernike3D, XmippApplyFieldZernike3D]
    _targets = [XmippApplyFieldZernike3D]
    _environments = [DESKTOP_TKINTER, WEB_DJANGO]
    OPEN_FILE = "open %s\n"
    VOXEL_SIZE = "volume #%d voxelSize %s\n"
    VOL_HIDE = "vol #%d hide\n"
    VIEW = "view\n"

    def __init__(self, **kwargs):
        ProtocolViewer.__init__(self, **kwargs)
        if type(self.protocol).__name__ == XmippApplyFieldZernike3D.__name__:
            self.deformed = self.protocol.deformed
            self.have_set = isinstance(self.deformed, Set)
            if self.have_set:
                self.choices = list(self.deformed.getIdSet())
            else:
                self.choices = [0]
                self.deformed = [self.deformed]
        else:
            self.have_set = False
            self.choices = None

        if isinstance(self.deformed, objects.SetOfVolumesFlex) or isinstance(self.deformed[0], objects.VolumeFlex):
            self.mode = "Map"
        else:
            self.mode = "Structure"

    def _defineParams(self, form):
        form.addSection(label='Show deformation')
        form.addParam('idChoice', params.EnumParam,
                      condition='have_set',
                      choices=[str(idx) for idx in self.choices], default=0,
                      label=f'{self.mode} to display', display=params.EnumParam.DISPLAY_COMBO,
                      help=f'Select which {self.mode} to display from the IDs of the set')
        form.addParam('doShowStrain', params.LabelParam,
                      label="Display the strain deformation")
        form.addParam('doShowRotation', params.LabelParam,
                      label="Display the rotation deformation")
        if self.mode == "Map":
            form.addParam('doShowMorphOrigRef', params.LabelParam,
                          label="Display the morphing between original and reference structure")
            form.addParam('doShowMorphDeformedRef', params.LabelParam,
                          label="Display the morphing between deformed and reference structure")
        elif self.mode == "Structure":
            form.addParam('doShowPDB', params.LabelParam,
                          label="Display original and deformed PDB or volume")
            form.addParam('doShowMorph', params.LabelParam,
                          label="Display a morphing between the original and deformed structure")

    def _getVisualizeDict(self):
        if type(self.protocol).__name__ == XmippApplyFieldZernike3D.__name__:
            if self.have_set:
                self.chosen = self.deformed[self.choices[self.idChoice.get()]]
            else:
                self.chosen = self.deformed[0]

            if self.mode == "Map":
                self.inputVol = self.protocol.inputVolume.get() if not self.have_set else \
                                self.protocol.inputVolume.get()[list(self.deformed.getIdSet())[self.idChoice.get()]]
                myDict = {'fnRefVol': self.inputVol.getFileName(),
                          'fnOutVol': self.chosen.getFileName()}

                self.protocol._updateFilenamesDict(myDict)
                # if not self.have_set:
                #     self.protocol._createFilenameTemplates()

        if self.mode == "Map":
            return {'doShowStrain': self._doShowStrain,
                    'doShowRotation': self._doShowRotation,
                    'doShowMorphOrigRef': self._doShowMorphOrigRef,
                    'doShowMorphDeformedRef': self._doShowDeformedOrigRef}
        elif self.mode == "Structure":
            return {'doShowStrain': self._doShowStrainStruct,
                    'doShowRotation': self._doShowRotationStruct,
                    'doShowPDB': self._doShowPDB,
                    'doShowMorph': self._doShowMorph}

    # ------------------- Mode MAP Methods -------------------
    def _doShowStrain(self, param=None):
        scriptFile = self.protocol._getPath('strain_chimera.cxc')
        fhCmd = open(scriptFile, 'w')
        fnref = os.path.abspath(self.chosen.getFileName())
        smprt = self.chosen.getSamplingRate()

        fnbase2 = removeExt(self.chosen.getFileName())
        fnStrain = os.path.abspath(fnbase2)

        fhCmd.write(self.OPEN_FILE % fnref)
        fhCmd.write(self.OPEN_FILE % (fnStrain + "_strain.mrc"))
        counter = 1
        fhCmd.write(self.VOXEL_SIZE % (counter, str(smprt)))
        counter += 1
        fhCmd.write(self.VOXEL_SIZE % (counter, str(smprt)))
        fhCmd.write(self.VOL_HIDE % counter)
        fhCmd.write('color sample #%d map #%d palette rainbow\n' % (counter - 1, counter))
        fhCmd.write(self.VIEW)
        fhCmd.close()

        view = ChimeraView(scriptFile)
        return [view]

    def _doShowRotation(self, param=None):
        scriptFile = self.protocol._getPath('rotation_chimera.cxc')
        fhCmd = open(scriptFile, 'w')
        fnref = os.path.abspath(self.chosen.getFileName())
        smprt = self.chosen.getSamplingRate()

        fnbase2 = removeExt(self.chosen.getFileName())
        fnStrain = os.path.abspath(fnbase2)

        fhCmd.write(self.OPEN_FILE % fnref)
        fhCmd.write(self.OPEN_FILE % (fnStrain + "_rotation.mrc"))
        counter = 1
        fhCmd.write(self.VOXEL_SIZE % (counter, str(smprt)))
        counter += 1
        fhCmd.write(self.VOXEL_SIZE % (counter, str(smprt)))
        fhCmd.write(self.VOL_HIDE % (counter))
        fhCmd.write('color sample #%d map #%d palette rainbow\n' % (counter - 1, counter))
        fhCmd.write(self.VIEW)
        fhCmd.close()

        view = ChimeraView(scriptFile)
        return [view]

    def _doShowMorphOrigRef(self, param=None):
        scriptFile = self.protocol._getPath('morph_orig_ref_chimera.cxc')
        fhCmd = open(scriptFile, 'w')
        fninput = os.path.abspath(self.inputVol.getFileName())
        fnref = os.path.abspath(self.chosen.getFlexInfo().refMap.get())
        smprt = self.chosen.getSamplingRate()

        fhCmd.write(self.OPEN_FILE % fninput)
        fhCmd.write(self.OPEN_FILE % fnref)

        counter = 1
        fhCmd.write(self.VOXEL_SIZE % (counter, str(smprt)))
        fhCmd.write(self.VOL_HIDE % (counter))
        counter += 1
        fhCmd.write(self.VOXEL_SIZE % (counter, str(smprt)))
        fhCmd.write(self.VOL_HIDE % (counter))
        fhCmd.write("volume morph #%d,%d frames 100 playStep 0.01\n" % (counter - 1, counter))
        fhCmd.write(self.VIEW)
        fhCmd.close()

        view = ChimeraView(scriptFile)
        return [view]

    def _doShowDeformedOrigRef(self, param=None):
        if self.protocol.applyPDB.get():
            raise ValueError("This viewer is only for volumes, not atomic structures")

        scriptFile = self.protocol._getPath('morph_deformed_ref_chimera.cxc')
        fhCmd = open(scriptFile, 'w')
        fninput = os.path.abspath(self.inputVol.getFileName())
        fnref = os.path.abspath(self.chosen.getFileName())
        smprt = self.chosen.getSamplingRate()

        fhCmd.write(self.OPEN_FILE % fninput)
        fhCmd.write(self.OPEN_FILE % fnref)
        counter = 1
        fhCmd.write(self.VOXEL_SIZE % (counter, str(smprt)))
        fhCmd.write(self.VOL_HIDE % counter)
        counter += 1
        fhCmd.write(self.VOXEL_SIZE % (counter, str(smprt)))
        # fhCmd.write("focus\n")
        fhCmd.write(self.VOL_HIDE % counter)
        fhCmd.write("volume morph #%d,%d frames 100 playStep 0.01\n" % (counter-1, counter))
        fhCmd.write(self.VIEW)
        fhCmd.close()

        view = ChimeraView(scriptFile)
        return [view]
    # ------------------- ------------------- -------------------

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
        if self.protocol.applyPDB.get():
            scriptFile = self.protocol._getPath('pdb_deform_chimera.cxc')
            fhCmd = open(scriptFile, 'w')
            inputFile = os.path.abspath(self.protocol.inputPDB.get().getFileName())
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
        else:
            raise ValueError("This viewer is only for atomic structures")

    def _doShowMorph(self, obj, **kwargs):
        if self.protocol.applyPDB.get():
            scriptFile = self.protocol._getPath('pdb_deform_chimera.cxc')
            fhCmd = open(scriptFile, 'w')
            inputFile = os.path.abspath(self.protocol.inputPDB.get().getFileName())
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
        else:
            raise ValueError("This viewer is only for atomic structures")
    # ------------------- ------------------- -------------------

