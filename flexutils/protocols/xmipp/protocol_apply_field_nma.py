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


import numpy as np
import prody as pd
from pathlib import Path
import os

from pwem.protocols import ProtAnalysis3D, ProtFlexBase
from pwem.objects import AtomStructFlex

import pyworkflow.protocol.params as params
import pyworkflow.utils as pwutils

import xmipp3

import flexutils
import flexutils.constants as const
from flexutils.protocols.xmipp.utils.utils import inscribedRadius

class XmippApplyFieldNMA(ProtAnalysis3D, ProtFlexBase):
    """ Protocol for PDB deformation based on NMA basis. """
    _label = 'apply deformation field - NMA'
    _subset = ["ca", "bb", None]

    # --------------------------- DEFINE param functions -----------------------
    def _defineParams(self, form):
        form.addSection(label='Input')
        form.addParam('inputStruct', params.PointerParam, label="NMA structure(s)",
                      important=True, pointerClass="SetOfAtomStructFlex, AtomStructFlex",
                      help='Structure(s) with NMA coefficients assigned.')

    # --------------------------- INSERT steps functions -----------------------
    def _insertAllSteps(self):
        self._insertFunctionStep(self.deformStep)
        self._insertFunctionStep(self.analyzeRSStep)
        self._insertFunctionStep(self.createOutputStep)

    # --------------------------- STEPS functions ------------------------------
    def deformStep(self):
        structs = self.inputStruct.get()
        if isinstance(structs, AtomStructFlex):
            structs = [structs]
            num_structs = 1
        else:
            structs = structs
            num_structs = structs.getSize()
        base_name = pwutils.removeBaseExt(structs.getFirstItem().getFileName())

        self.len_num_structs = len(str(num_structs))

        # Load NMA basis
        basis_file = Path(Path(structs.getFirstItem().getFlexInfo().modelPath.get()).parent.parent, "nma_basis.anm.npz")
        anm = pd.loadModel(str(basis_file))
        U = anm.getEigvecs()

        # Read structure file and get coords
        subset = structs.getFirstItem().getFlexInfo().atomSubset.get()
        pd_struct = pd.parsePDB(structs.getFirstItem().getFileName(), subset=subset, compressed=False)
        coords = pd_struct.getCoords()

        # Get and save boxsize (for strain/rotation analysis)
        boxsize = np.ceil(2 * inscribedRadius(coords)).astype(int)
        boxsize_path = self._getExtraPath("boxsize.txt")
        with open(boxsize_path, "w") as fid:
            fid.write(str(boxsize))

        # Get and save indices (for strain/rotation analysis)
        coords_centered = coords - np.mean(coords, axis=0)
        indices = np.round(coords_centered + int(0.5 * boxsize)).astype(int)
        indices = np.transpose(np.vstack([indices[:, 2], indices[:, 1], indices[:, 0]]))  # Indices must be in ZYX order
        indices_path = self._getExtraPath("indices.txt")
        np.savetxt(indices_path, indices)

        idx = 0
        for struct in structs:
            i_pad = str(idx).zfill(self.len_num_structs)

            # Get deformation field
            c_nma = struct.getZFlex()
            d_f = (U @ c_nma.T).reshape(-1, 3)

            # Apply deformation field
            c_moved = coords + d_f

            # Save deformation field (for strain/rotation analysis)
            field_path = self._getExtraPath("def_field_{0}.txt".format(i_pad))
            np.savetxt(field_path, d_f)

            # Saved deformed structure
            out_struct = pd_struct.copy()
            out_struct.setCoords(c_moved)
            pd.writePDB(self._getExtraPath(base_name + '_deformed_{0}.pdb'.format(i_pad)), out_struct)

            idx += 1

    def analyzeRSStep(self):
        indices_path = self._getExtraPath("indices.txt")
        with open(self._getExtraPath("boxsize.txt"), "r") as fid:
            boxsize = int(fid.readlines()[0])
        origin = np.floor(0.5 * boxsize).astype(int)

        structs = self.inputStruct.get()
        if isinstance(structs, AtomStructFlex):
            num_structs = 1
        else:
            num_structs = structs.getSize()

        struct_file = structs.getFirstItem().getFileName()
        base_name = pwutils.removeBaseExt(struct_file)

        for idx in range(num_structs):
            i_pad = str(idx).zfill(self.len_num_structs)

            # Get paths for this structure
            field_path = self._getExtraPath("def_field_{0}.txt".format(i_pad))
            out_path_strain = self._getExtraPath(base_name + '_deformed_{0}_strain.pdb'.format(i_pad))
            out_path_rotation = self._getExtraPath(base_name + '_deformed_{0}_rotation.pdb'.format(i_pad))

            # RS analysis
            args = "--field %s --indices %s --out_path %s --boxsize %d" % \
                   (field_path, indices_path, self._getExtraPath(), boxsize)
            program = os.path.join(const.XMIPP_SCRIPTS, "strain_rotation_analysis.py")
            program = flexutils.Plugin.getProgram(program)
            self.runJob(program, args)

            # Strain labeling
            program = 'xmipp_pdb_label_from_volume'
            args = "--pdb %s --vol %s -o %s --sampling 1.0 --radius 5 --origin %d %d %d" % \
                   (struct_file, self._getExtraPath("strain.mrc"), out_path_strain,
                    origin, origin, origin)
            self.runJob(program, args, env=xmipp3.Plugin.getEnviron())

            # Rotation labeling
            args = "--pdb %s --vol %s -o %s --sampling 1.0 --radius 5 --origin %d %d %d" % \
                   (struct_file, self._getExtraPath("rotation.mrc"), out_path_rotation,
                    origin, origin, origin)
            self.runJob(program, args, env=xmipp3.Plugin.getEnviron())

        # Cleaning to save some memory
        pwutils.cleanPattern(self._getExtraPath("*.txt"))

    def createOutputStep(self):
        structs = self.inputStruct.get()
        base_name = pwutils.removeBaseExt(structs.getFirstItem().getFileName())

        if isinstance(structs, AtomStructFlex):
            c_nma = structs.getZFlex()

            outFile = self._getExtraPath(base_name + '_deformed.pdb')
            out_struct = AtomStructFlex(progName=const.NMA)
            out_struct.copyInfo(structs)
            out_struct.setLocation(outFile)
            out_struct.setZFlex(c_nma)

            self._defineOutputs(deformed=out_struct)
            self._defineSourceRelation(self.inputStruct, out_struct)
        else:
            out_structs = self._createSetOfAtomStructFlex(progName=const.NMA)

            idx = 0
            for struct in structs:
                i_pad = str(idx).zfill(self.len_num_structs)

                c_nma = struct.getZFlex()

                outFile = self._getExtraPath(base_name + '_deformed_{0}.pdb'.format(i_pad))
                out_struct = AtomStructFlex(progName=const.NMA)
                out_struct.copyInfo(struct)
                out_struct.setLocation(outFile)
                out_struct.setZFlex(c_nma)

                out_structs.append(out_struct)

                idx += 1

            self._defineOutputs(deformed=out_structs)
            self._defineSourceRelation(self.inputStruct, out_structs)

    # --------------------------- UTILS functions ------------------------------
