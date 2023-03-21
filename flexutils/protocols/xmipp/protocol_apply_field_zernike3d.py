# **************************************************************************
# *
# * Authors:     David Herreros Calero (dherreros@cnb.csic.es)
# *              James Krieger (jmkrieger@cnb.csic.es)
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

from pwem.protocols import ProtAnalysis3D

import pyworkflow.protocol.params as params
import pyworkflow.utils as pwutils
from pyworkflow.object import Float

from flexutils.protocols import ProtFlexBase
from flexutils.objects import VolumeFlex, AtomStructFlex
import flexutils.constants as const


class XmippApplyFieldZernike3D(ProtAnalysis3D, ProtFlexBase):
    """ Protocol for PDB deformation based on Zernike3D basis. """
    _label = 'apply deformation field - Zernike3D'

    # --------------------------- DEFINE param functions -----------------------
    def _defineParams(self, form):
        form.addSection(label='Input')
        form.addParam('inputVolume', params.PointerParam, label="Zernike3D volume(s)",
                      important=True, pointerClass="SetOfVolumesFlex, VolumeFlex",
                      help='Volume(s) with Zernike3D coefficients assigned.')
        form.addParam('applyPDB', params.BooleanParam, label="Apply to structure?",
                      default=False,
                      help="If True, you will be able to provide an atomic structure to be deformed "
                           "based on the Zernike3D coefficients associated to the input volume(s). "
                           "If False, the coefficients will be applied to the volume(s) directly.")
        form.addParam('inputPDB', params.PointerParam, label="Input PDB",
                      pointerClass='AtomStruct', allowsNull=True, condition="applyPDB==True",
                      help='Atomic structure to apply the deformation fields defined by the '
                           'Zernike3D coefficients associated to the input volume. '
                           'For better results, the volume(s) and structure should be aligned')
        form.addParam('moveBoxOrigin', params.BooleanParam, default=False, condition="applyPDB==True",
                      label="Move structure to box origin?",
                      help="If PDB has been aligned inside Scipion, set to False. Otherwise, this option will "
                           "correctly place the PDB in the origin of the volume.")

    # --------------------------- INSERT steps functions -----------------------
    def _insertAllSteps(self):
        self._insertFunctionStep("deformStep")
        self._insertFunctionStep("createOutputStep")

    # --------------------------- STEPS functions ------------------------------
    def deformStep(self):
        volumes = self.volume.get()
        if isinstance(volumes, VolumeFlex):
            volumes = [volumes]
            num_vols = 1
        else:
            volumes = volumes
            num_vols = volumes.getSize()

        self.len_num_vols = len(str(num_vols))

        idx = 0
        for volume in volumes:
            i_pad = str(idx).zfill(self.len_num_vols)

            # Write coefficients to file
            z_clnm_file = self._getExtraPath("z_clnm_{0}.txt".format(i_pad))
            z_clnm = volume.getZFlex()

            if self.applyPDB.get():
                z_clnm *= volume.getSamplingRate()

            z_clnm = np.char.mod('%f', z_clnm)
            z_clnm = ",".join(z_clnm)

            self.writeZernikeFile(volume, z_clnm, z_clnm_file)

            if self.applyPDB.get():
                boxSize = volume.getXDim()
                samplingRate = volume.getSamplingRate()
                outFile = pwutils.removeBaseExt(self.inputPDB.get().getFileName()) + '_deformed_{0}.pdb'.format(i_pad)
                params = ' --pdb %s --clnm %s -o %s --sr %f' % \
                         (self.inputPDB.get().getFileName(), z_clnm_file, self._getExtraPath(outFile),
                          samplingRate)
                if self.moveBoxOrigin.get():
                    params += " --boxsize %d" % boxSize
                self.runJob("xmipp_pdb_sph_deform", params)
            else:
                outFile = self._getExtraPath("deformed_volume_{0}.mrc".format(i_pad))
                volume_file = volume.getFileName()
                if pwutils.getExt(volume_file) == ".mrc":
                    volume_file += ":mrc"
                params = "-i %s  --step 1 --blobr 2 -o %s --clnm %s" % \
                         (volume_file, outFile, z_clnm_file)
                if volume.getFlexInfo().refMask:
                    mask_file = volume.getFlexInfo().refMask.get()
                    if pwutils.getExt(mask_file) == ".mrc":
                        volume_file += ":mrc"
                    params += " --mask %s" % mask_file
                self.runJob("xmipp_volume_apply_coefficient_zernike3d", params)

            idx += 1

    def createOutputStep(self):
        volumes = self.volume.get()
        Rmax = Float(int(0.5 * volumes.getXDim()))
        if isinstance(volumes, VolumeFlex):
            L1 = volumes.getFlexInfo().L1
            L2 = volumes.getFlexInfo().L2
            refMap = volumes.getFlexInfo().refMap
            refMask = volumes.getFlexInfo().refMask
            z_clnm = volumes.getZFlex()

            if self.applyPDB.get():
                outFile = pwutils.removeBaseExt(self.inputPDB.get().getFileName()) + '_deformed.pdb'
                pdb = AtomStructFlex(filename=self._getExtraPath(outFile), progName=const.ZERNIKE3D)
                pdb.getFlexInfo().L1 = L1
                pdb.getFlexInfo().L2 = L2
                pdb.getFlexInfo().Rmax = Float(volumes.getSamplingRate() * Rmax.get())
                pdb.getFlexInfo().refMap = refMap
                pdb.getFlexInfo().refMask = refMask
                pdb.setZFlex(z_clnm)
                self._defineOutputs(deformed=pdb)
                self._defineSourceRelation(self.inputPDB, pdb)
                self._defineSourceRelation(self.volume, pdb)
            else:
                vol = VolumeFlex(progName=const.ZERNIKE3D)
                vol.setSamplingRate(volumes.getSamplingRate())
                vol.setFileName(self._getExtraPath("deformed_volume.mrc"))
                vol.getFlexInfo().L1 = L1
                vol.getFlexInfo().L2 = L2
                vol.getFlexInfo().Rmax = Rmax
                vol.getFlexInfo().refMap = refMap
                vol.getFlexInfo().refMask = refMask
                vol.setZFlex(z_clnm)
                self._defineOutputs(deformed=vol)
                self._defineSourceRelation(volumes, vol)
        else:
            L1 = volumes.getFirstItem().getFlexInfo().L1
            L2 = volumes.getFirstItem().getFlexInfo().L2
            refMap = volumes.getFirstItem().getFlexInfo().refMap
            refMask = volumes.getFirstItem().getFlexInfo().refMask
            if self.applyPDB.get():
                pdbs = self._createSetOfAtomStructFlex(progName=const.ZERNIKE3D)
            else:
                vols = self._createSetOfVolumesFlex(progName=const.ZERNIKE3D)
                vols.setSamplingRate(volumes.getSamplingRate())

            idx = 0
            for volume in volumes:
                i_pad = str(idx).zfill(self.len_num_vols)

                z_clnm = volume.getZFlex()

                if self.applyPDB.get():
                    outFile = pwutils.removeBaseExt(self.inputPDB.get().getFileName()) + '_deformed_{0}.pdb'.format(i_pad)
                    pdb = AtomStructFlex(filename=self._getExtraPath(outFile), progName=const.ZERNIKE3D)
                    pdb.getFlexInfo().L1 = L1
                    pdb.getFlexInfo().L2 = L2
                    pdb.getFlexInfo().Rmax = Float(volume.getSamplingRate() * Rmax.get())
                    pdb.getFlexInfo().refMap = refMap
                    pdb.getFlexInfo().refMask = refMask
                    pdb.setZFlex(z_clnm)

                    pdbs.append(pdb)
                else:
                    vol = VolumeFlex(progName=const.ZERNIKE3D)
                    vol.setSamplingRate(volume.getSamplingRate())
                    vol.setFileName(self._getExtraPath("deformed_volume_{0}.mrc".format(i_pad)))
                    vol.getFlexInfo().L1 = L1
                    vol.getFlexInfo().L2 = L2
                    vol.getFlexInfo().Rmax = Rmax
                    vol.getFlexInfo().refMap = refMap
                    vol.getFlexInfo().refMask = refMask
                    vol.setZFlex(z_clnm)

                    vols.append(vol)

                    idx += 1

            if self.applyPDB.get():
                self._defineOutputs(deformed=pdbs)
                self._defineSourceRelation(self.inputPDB, pdbs)
                self._defineSourceRelation(self.volume, pdbs)
            else:
                self._defineOutputs(deformed=vols)
                self._defineSourceRelation(self.volume, vols)

    # --------------------------- UTILS functions ------------------------------
    def writeZernikeFile(self, volume, z_clnm, file):
        L1 = volume.getFlexInfo().L1.get()
        L2 = volume.getFlexInfo().L2.get()
        Rmax = int(0.5 * volume.getXDim())
        Rmax = volume.getSamplingRate() * Rmax if self.applyPDB.get() else Rmax
        with open(file, 'w') as fid:
            fid.write(' '.join(map(str, [L1, L2, Rmax])) + "\n")
            fid.write(z_clnm.replace(",", " ") + "\n")