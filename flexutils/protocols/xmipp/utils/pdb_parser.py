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
from Bio import PDB
import gemmi
from scipy.spatial import cKDTree
from Bio.PDB.vectors import calc_dihedral


class AtomicModelParser:
    def __init__(self, file_path, subset='all'):
        self.file_path = file_path
        self.subset = subset
        self.atoms = []
        self.atom_indices = []
        self.ca_indices = []
        self.file_format = self._determine_file_format(file_path)
        self.load_structure()

    def _determine_file_format(self, file_path):
        _, file_extension = os.path.splitext(file_path)
        if file_extension.lower() in ['.pdb', '.ent']:
            return 'pdb'
        elif file_extension.lower() == '.cif':
            return 'cif'
        else:
            raise ValueError("Unsupported file format. Use '.pdb', '.ent', or '.cif'.")

    def load_structure(self):
        if self.file_format == 'pdb':
            self._load_pdb()
        elif self.file_format == 'cif':
            self._load_cif()

    def _load_pdb(self):
        parser = PDB.PDBParser(QUIET=True)
        self.structure = parser.get_structure('structure', self.file_path)
        self._parse_structure()

    def _load_cif(self):
        self.structure = gemmi.read_structure(self.file_path)
        self._parse_structure()

    def _parse_structure(self):
        atom_index = 0
        for model in self.structure:
            for chain in model:
                for residue in chain:
                    for atom in residue:
                        if self.subset == 'all':
                            self.atoms.append(atom)
                            self.atom_indices.append(atom_index)
                        elif self.subset == 'bb' and atom.name in ['N', 'CA', 'C', 'O']:
                            self.atoms.append(atom)
                            self.atom_indices.append(atom_index)
                        elif self.subset == 'ca' and atom.name == 'CA':
                            self.atoms.append(atom)
                            self.atom_indices.append(atom_index)
                        if atom.name == 'CA':
                            self.ca_indices.append(atom_index)
                        atom_index += 1

    def get_atom_coordinates(self):
        if self.file_format == 'pdb':
            coordinates = np.array([atom.coord for atom in self.atoms])
        elif self.file_format == 'cif':
            coordinates = np.array([atom.pos.tolist() for atom in self.atoms])
        return coordinates

    def get_covalent_bonds(self):
        coordinates = self.get_atom_coordinates()
        tree = cKDTree(coordinates)
        pairs = tree.query_pairs(r=1.6)  # Covalent bond distance threshold
        bonds = list(pairs)
        return np.array(bonds)

    def _is_covalently_bonded(self, atom1, atom2, max_distance=1.6):
        distance = atom1 - atom2
        return distance < max_distance

    def get_dihedral_angles(self):
        dihedrals = []
        atom_index_map = {atom: idx for idx, atom in enumerate(self.atoms)}

        if self.file_format == 'pdb':
            parser = PDB.PDBParser(QUIET=True)
            structure = parser.get_structure('structure', self.file_path)
            for model in structure:
                for chain in model:
                    residues = list(chain)
                    for i in range(len(residues) - 3):
                        try:
                            atoms = [residues[j][atom_name] for j in range(i, i + 4) for atom_name in ['N', 'CA', 'C']]
                            if all(atom in atom_index_map for atom in atoms):
                                atom_indices = [atom_index_map[atom] for atom in atoms]
                                dihedrals.append(atom_indices)
                        except KeyError:
                            continue
        elif self.file_format == 'cif':
            structure = gemmi.read_structure(self.file_path)
            for model in structure:
                for chain in model:
                    residues = list(chain)
                    for i in range(len(residues) - 3):
                        try:
                            atoms = [[residues[j]['N'], residues[j]['CA'], residues[j]['C']] for j in range(i, i + 4)]
                            if all(atom in atom_index_map for atom in atoms):
                                atom_indices = [atom_index_map[atom] for atom in atoms]
                                dihedrals.append(atom_indices)
                        except KeyError:
                            continue
        return np.array(dihedrals)

    def get_ca_indices(self):
        return np.array(self.ca_indices)
