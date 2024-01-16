

import numpy as np
from Bio.PDB.PDBParser import PDBParser
from Bio.PDB.internal_coords import AtomKey

import prody as pd


class PDBUtils:
    # SELECT = {"all": "all", "bb": ["CA", "N", "C"], "ca": ["CA"]}
    SELECT = {"all": None, "bb": "bb", "ca": "ca"}

    def __init__(self, selectionString):
        self.selectionString = self.SELECT[selectionString]
        # self.parser = PDBParser()

    # def parsePDB(self, file):
    #     protein = self.parser.get_structure("protein", file)[0]
    #
    #     # create a selector to filter just the C-alpha atoms from the all atom array
    #     backbone_hedra = []
    #     # debug_backbone_hedra = []  # For debugging purposes
    #     idx_correspondence = dict()
    #     backbone_coords = []
    #     # debug_backbone_coords = []  # For debugging purposes
    #     chain_idx = 1
    #     for chain in protein:
    #         # Compute bond lengths, angles, dihedral angles
    #         chain.atom_to_internal_coordinates(verbose=True)
    #
    #         # Get connection indices only for backbone
    #         if chain.internal_coord.atomArrayIndex:
    #             for hedron in chain.internal_coord.hedra:
    #                 atmNameNdx = AtomKey.fields.atm
    #                 atomArrayIndex = chain.internal_coord.atomArrayIndex
    #                 atomArray = chain.internal_coord.atomArray
    #                 if isinstance(self.selectionString, list):
    #                     select = all([k.akl[atmNameNdx] in self.selectionString for k in hedron])
    #                 elif self.selectionString == "all":
    #                     select = True
    #                 if select:
    #                     new_hedron = []
    #                     for k in hedron:
    #                         hedron_idx = atomArrayIndex.get(k)
    #                         new_key = str(chain_idx) + "_" + str(hedron_idx)
    #                         if not new_key in idx_correspondence.keys():
    #                             idx_correspondence[new_key] = len(idx_correspondence)
    #                             backbone_coords.append(atomArray[hedron_idx][:-1])
    #                         new_hedron.append(idx_correspondence[new_key])
    #                     backbone_hedra.append(new_hedron)
    #                     # debug_backbone_hedra.append([atomArrayIndex.get(k) for k in hedron])
    #
    #             # # Debugging purposes
    #             # atmNameNdx = AtomKey.fields.atm
    #             # atomArrayIndex = chain.internal_coord.atomArrayIndex
    #             # backboneSelect = [
    #             #     atomArrayIndex.get(k) for k in atomArrayIndex.keys() if k.akl[atmNameNdx] in self.selectionString
    #             # ]
    #             # debug_backbone_coords.append(chain.internal_coord.atomArray[backboneSelect])
    #
    #         # Update chain index
    #         chain_idx += 1
    #
    #     backbone_hedra = np.asarray(backbone_hedra, dtype=int)
    #     # debug_backbone_hedra = np.asarray(debug_backbone_hedra, dtype=int)
    #     backbone_coords = np.asarray(backbone_coords, dtype=float)
    #     # debug_backbone_coords = np.vstack(debug_backbone_coords)[:, :-1]
    #
    #     # Debugging purposes
    #     # aux_1 = backbone_coords[np.argsort(np.linalg.norm(backbone_coords, axis=1))]
    #     # aux_2 = debug_backbone_coords[np.argsort(np.linalg.norm(debug_backbone_coords, axis=1))]
    #     # same_coords = np.array_equal(aux_1[:, 0], aux_2[:, 0]) and \
    #     #               np.array_equal(aux_1[:, 1], aux_2[:, 1]) and \
    #     #               np.array_equal(aux_1[:, 2], aux_2[:, 2])
    #
    #     return backbone_coords, backbone_hedra

    @classmethod
    def find_dihedrals(cls, bond_indices):
        # Convert bond indices to a set for faster lookup
        bonds = set(map(tuple, map(sorted, bond_indices)))
        dihedrals = []
        # Create a dictionary to store neighbors for each atom
        neighbors = {}
        for bond in bonds:
            atom1, atom2 = bond
            if atom1 not in neighbors:
                neighbors[atom1] = set()
            if atom2 not in neighbors:
                neighbors[atom2] = set()
            neighbors[atom1].add(atom2)
            neighbors[atom2].add(atom1)
        for bond in bonds:
            atom1, atom2 = bond
            # Potential neighbors for the dihedral
            for neighbor1 in neighbors[atom1]:
                if neighbor1 == atom2:
                    continue
                for neighbor2 in neighbors[atom2]:
                    if neighbor2 == atom1 or neighbor2 == neighbor1:
                        continue
                    # Check for unique dihedral indices
                    dihedral = tuple(sorted([neighbor1, atom1, atom2, neighbor2]))
                    if dihedral not in dihedrals:
                        dihedrals.append(dihedral)
        return np.array(dihedrals)
    def parsePDB(self, file):
        # Parse model file based on subset
        structure = pd.parsePDB(file, subset=self.selectionString, compressed=False)

        # Infer bonds from parsed model
        _ = structure.inferBonds()

        # Get bond indices
        bonds = np.asarray([bond.getIndices() for bond in structure.iterBonds()])

        # Get dihedral angles
        dihedrals = PDBUtils.find_dihedrals(bonds)

        # Get atom coordinates
        atom_coordinates = structure.getCoords()

        return atom_coordinates, dihedrals


    @staticmethod
    def calc_bond(coords, connectivity):
        def ReLU(x):
            return x * (x > 0)

        px = coords[connectivity[:, 0]]
        py = coords[connectivity[:, 1]]
        pz = coords[connectivity[:, 2]]
        dst_1 = np.sqrt(ReLU(np.sum((px - py) ** 2, axis=1)))
        dst_2 = np.sqrt(ReLU(np.sum((py - pz) ** 2, axis=1)))

        return np.stack([dst_1, dst_2]).T

    @staticmethod
    def calc_angle(coords, connectivity, degrees=True):
        p0 = coords[connectivity[:, 0]]
        p1 = coords[connectivity[:, 1]]
        p2 = coords[connectivity[:, 2]]
        b0 = p0 - p1
        b1 = p2 - p1
        ang = np.sum(b0 * b1, axis=1)
        n0 = np.linalg.norm(b0, axis=1) * np.linalg.norm(b1, axis=1)
        ang = np.nan_to_num(ang / n0)
        # ang = np.min(np.max(ang, axis=1), axis=0)
        ang = np.arccos(ang)

        if degrees:
            ang *= 180 / np.pi

        return ang
