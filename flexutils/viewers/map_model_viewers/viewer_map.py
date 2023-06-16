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
import os
from xmipp_metadata.image_handler import ImageHandler
from xmipp_metadata.metadata import XmippMetaData

from sklearn.neighbors import KDTree
from tqdm import tqdm
from Bio.PDB import PDBParser
import _thread
import multiprocessing

from traits.api import HasTraits, Instance, Array, Float, Int, Bool, String,\
    on_trait_change, Callable, List
from traits.trait_types import Button, Enum
from traitsui.api import View, Item, HGroup, Group, HSplit, VGroup, RangeEditor, TextEditor, EnumEditor

from mayavi import mlab
from mayavi.core.api import PipelineBase, Source
from mayavi.core.ui.api import SceneEditor, MayaviScene, \
    MlabSceneModel

import flexutils
from flexutils.protocols.xmipp.utils.utils import computeBasis, readZernikeFile, getCoordsAtLevel, \
                                                  resizeZernikeCoefficients
from flexutils.viewers.utils import xpdb


################################################################################
# The object implementing the dialog
class MapView(HasTraits):
    # Map view
    scene3d = Instance(MlabSceneModel, ())

    # Map (actor)
    ipw_map = Instance(PipelineBase)

    # Vector field (actor)
    ipw_df_map = Instance(PipelineBase)
    ipw_df_atom = Instance(PipelineBase)

    # Atomic model (actor)
    ipw_atom_pts = Instance(PipelineBase)
    ipw_atom_tubes = Instance(PipelineBase)

    # The values to color the map
    interp_val = Array()

    # Visualization style
    opacity = Float(0.0)
    op_min = Float(0.0)
    op_max = Float(1.0)

    # Map
    map = Array()
    map_file = String()
    coords_map = Array()

    # Atomic model
    atom_file = String("None")
    atom_model = List()

    # Deformation field
    df_stats = List()

    # Map colormaps
    colormap = Enum("Mean motion", "Std motion")
    repaint = Button(label="Repaint")

    # Arrow properties
    mask_arrows = Button("Mask arrows")
    mask_factor = String("20")
    scale_arrows = Button("Scale arrows")
    scale_factor = String("1")

    # Show atom model
    show_atom_model = Bool(False)

    # ---------------------------------------------------------------------------
    def __init__(self, **traits):
        self.class_inputs = traits
        super(MapView, self).__init__(**traits)
        self.map
        self.ipw_map
        self.coords_map
        self.atom_model
        self.ipw_atom_pts
        self.df_stats
        self.ipw_df_map
        self.ipw_df_atom

    # ---------------------------------------------------------------------------
    # Default values
    # ---------------------------------------------------------------------------
    # def _ipw_map_default(self):
    #     return self.display_map()

    # def _ipw_atom_pts_default(self):
    #     if self.atom_model[0] != "None":
    #         return self.display_atom_model()

    # def _ipw_df_map_default(self):
    #     return self.display_df_map()

    # def _ipw_df_atom_default(self):
    #     if self.atom_model[0] != "None":
    #         return self.display_df_atom()

    def _map_default(self):
        map = self.readMap(self.map_file)
        return map

    def _atom_model_default(self):
        if self.atom_file != "None":
            atom_model = self.readPDB(self.atom_file)
            return atom_model
        else:
            return ["None"]

    def _coords_map_default(self):
        return getCoordsAtLevel(self.map, 1)

    def _df_stats_default(self):
        metadata = XmippMetaData(self.metadata_file)
        z_space = np.asarray(metadata.getMetaDataColumns("sphCoefficients"))
        # mean_df = np.zeros(self.coords_map.shape)
        # std_df = np.zeros(self.coords_map.shape)
        path = os.path.dirname(self.metadata_file)
        df_mean_file = os.path.join(path, "mean_df.txt")
        df_std_file = os.path.join(path, "std_df.txt")

        if os.path.isfile(df_mean_file):
            mean_df = np.loadtxt(df_mean_file)
            std_df = np.loadtxt(df_std_file)
        else:
            print("Computing deformation field statistics from particles...")
            R = 0.5 * self.map.shape[0]
            coords_xo = self.coords_map - R
            mean_df, std_df = summers(z_space, int(self.class_inputs.get("L1")),
                              int(self.class_inputs.get("L2")), coords_xo, R,
                              int(self.class_inputs.get("thr")))
            # for z in tqdm(z_space):
            #     Z = computeBasis(L1=int(self.class_inputs.get("L1")),
            #                      L2=int(self.class_inputs.get("L2")),
            #                      pos=coords_xo, r=0.5*self.map.shape[0])
            #     A = resizeZernikeCoefficients(z)
            #     mean_df = mean_df + Z @ A.T
            #     std_df = std_df + mean_df * mean_df
            mean_df /= z_space.shape[0]
            std_df = np.sqrt(std_df / z_space.shape[0] - mean_df * mean_df)
            np.savetxt(df_mean_file, mean_df)
            np.savetxt(df_std_file, std_df)
        return [mean_df, std_df]

    # ---------------------------------------------------------------------------
    # Scene activation callbaks
    # ---------------------------------------------------------------------------
    @on_trait_change('scene3d.activated')
    def display_scene3d(self):
        self.scene3d.mlab.view(40, 50)
        self.scene3d.scene.background = (0, 0, 0)

    @on_trait_change('scene3d.activated')
    def display_map(self):
        volume = mlab.contour3d(self.map, color=(1, 1, 1), opacity=0.0, contours=1, figure=self.scene3d.mayavi_scene)
        volume.contour.auto_contours = False
        volume.contour.auto_update_range = False
        volume.actor.visible = True
        setattr(self, 'ipw_map', volume)

    @on_trait_change('scene3d.activated')
    def display_atom_model(self):
        if self.atom_model[0] != "None":
            # Xmipp convention is ZYX array
            i_sr = 1 / self.class_inputs["sr"]
            z, y, x, connections, scalars = self.atom_model
            pts = mlab.points3d(x, y, z, color=(1., 1., 1.), scale_factor=3 * i_sr, resolution=10, opacity=0.)
            pts.mlab_source.dataset.lines = np.array(connections)
            tube = mlab.pipeline.tube(pts, tube_radius=1. * i_sr)
            tube = mlab.pipeline.surface(tube, color=(0.5, 0.5, 0.5), opacity=0.)
            pts.visible = False
            tube.visible = False
            setattr(self, 'ipw_atom_pts', pts)
            setattr(self, 'ipw_atom_tubes', tube)

    @on_trait_change('scene3d.activated')
    def display_df_map(self):
        # Xmipp convention is ZYX array
        z, y, x = self.coords_map[:, 0], self.coords_map[:, 1], self.coords_map[:, 2]
        w, v, u = self.df_stats[0][:, 0], self.df_stats[0][:, 1], self.df_stats[0][:, 2]
        scalars = np.linalg.norm(self.df_stats[0], axis=1)
        arrows = mlab.quiver3d(x, y, z, u, v, w, colormap="viridis", mask_points=1, scale_factor=1,
                               scalars=scalars, figure=self.scene3d.mayavi_scene)
        arrows.glyph.color_mode = 'color_by_scalar'
        setattr(self, 'ipw_df_map', arrows)

    @on_trait_change('scene3d.activated')
    def display_df_atom(self):
        if self.atom_model[0] != "None":
            # Xmipp convention is ZYX array
            coords_map = np.asarray([self.coords_map[:, 2], self.coords_map[:, 1], self.coords_map[:, 0]]).T
            tree = KDTree(coords_map)
            z, y, x, _, _ = self.atom_model
            coords_atoms = np.asarray([x, y, z]).T
            _, ids = tree.query(coords_atoms, k=1)
            self.ids = np.asarray(ids).reshape(-1)
            w, v, u = self.df_stats[0][self.ids, 0], self.df_stats[0][self.ids, 1], self.df_stats[0][self.ids, 2]
            scalars = np.linalg.norm(self.df_stats[0][self.ids, :], axis=1)
            arrows = mlab.quiver3d(x, y, z, u, v, w, colormap="viridis", mask_points=1, scale_factor=1,
                                   scalars=scalars, figure=self.scene3d.mayavi_scene)
            arrows.glyph.color_mode = 'color_by_scalar'
            arrows.visible = False
            setattr(self, 'ipw_df_atom', arrows)

    @on_trait_change("opacity")
    def change_opacity(self):
        volume = getattr(self, 'ipw_map')
        atoms = getattr(self, 'ipw_atom_pts')
        tubes = getattr(self, 'ipw_atom_tubes')
        volume.actor.property.opacity = self.opacity
        if self.atom_model[0] != "None":
            atoms.actor.property.opacity = self.opacity
            tubes.actor.property.opacity = self.opacity

        # scalars = volume.mlab_source.scalars
        # contour = np.amin(scalars) + (np.amax(scalars) - np.amin(scalars)) * self.contour_level
        # volume.contour.contours = [contour]

    @on_trait_change("show_atom_model")
    def show_map_or_atoms(self):
        if self.atom_model[0] != "None":
            self.ipw_atom_pts.visible = self.show_atom_model
            self.ipw_atom_tubes.visible = self.show_atom_model
            self.ipw_df_atom.visible = self.show_atom_model
            self.ipw_df_map.visible = not self.show_atom_model
            self.ipw_map.visible = not self.show_atom_model

    def _repaint_fired(self):
        arrows_map = getattr(self, 'ipw_df_map')
        arrows_atom = getattr(self, 'ipw_df_atom')
        if self.colormap == "Mean motion":
            scalars = np.linalg.norm(self.df_stats[0], axis=1)
            # map = np.copy(self.map)
            # for idx, coord in enumerate(self.coords_map):
            #     map[coord[2], coord[1], coord[0]] = mean[idx]
            # map[self.coords_map[:, 2], self.coords_map[:, 1], self.coords_map[:, 0]] = mean
            # max_val = np.amax(mean)
            # min_val = np.amin(mean)
        elif self.colormap == "Std motion":
            scalars = np.linalg.norm(self.df_stats[1], axis=1)
            # map = np.copy(self.map)
            # map[self.coords_map[:, 2], self.coords_map[:, 1], self.coords_map[:, 0]] = std
            # max_val = np.amax(std)
            # min_val = np.amin(std)
        arrows_map.mlab_source.reset(scalars=scalars)
        if self.atom_model[0] != "None":
            arrows_atom.mlab_source.reset(scalars=scalars[self.ids])
        # volume.contour.maximum_contour = max_val
        # volume.contour.minimum_contour = min_val
        # volume.actor.visible = True
        # self.ipw_map
        # volume = mlab.contour3d(map, colormap="viridis", figure=self.scene3d.mayavi_scene)
        # setattr(self, 'ipw_map', volume)

    def _mask_arrows_fired(self):
        mask_factor = int(self.mask_factor)
        arrows_map = getattr(self, 'ipw_df_map')
        arrows_map.glyph.mask_points.on_ratio = mask_factor
        if self.atom_model[0] != "None":
            arrows_atom = getattr(self, 'ipw_df_atom')
            arrows_atom.glyph.mask_points.on_ratio = mask_factor

    def _scale_arrows_fired(self):
        scale_factor = float(self.scale_factor)
        arrows = getattr(self, 'ipw_df_map')
        arrows.glyph.glyph.scale_factor = scale_factor
        if self.atom_model[0] != "None":
            arrows_atom = getattr(self, 'ipw_df_atom')
            arrows_atom.glyph.glyph.scale_factor = scale_factor


    # ---------------------------------------------------------------------------
    # Conversion functions
    # ---------------------------------------------------------------------------
    def getDataAtThreshold(self, data, level):
        return np.asarray(np.where(data >= level))

    # ---------------------------------------------------------------------------
    # Read functions
    # ---------------------------------------------------------------------------
    def readMap(self, file):
        map = ImageHandler().read(file).getData()
        return map

    def readPDB(self, file):
        sloppyparser = PDBParser(
            PERMISSIVE=True, structure_builder=xpdb.SloppyStructureBuilder()
        )
        structure = sloppyparser.get_structure("struct", file)

        nodes = dict()
        edges = list()
        atoms = set()
        last_atom_label = None
        last_chain_label = None

        for idx, atom in enumerate(structure.get_atoms()):
            if atom.get_name() in ["CA", "C"]:
                x, y, z = atom.get_coord()
                nodes[idx + 1] = (atom.get_name(), x, y, z)
                atoms.add(atom.get_name())
                chain_label = atom.get_parent().get_parent().get_id()
                if chain_label == last_chain_label:
                    edges.append((idx + 1, last_atom_label))
                last_atom_label = idx + 1
                last_chain_label = chain_label

        atoms = list(atoms)
        atoms.sort()
        atoms = dict(zip(atoms, range(len(atoms))))

        # Turn the graph into 3D positions, and a connection list.
        labels = dict()

        x = list()
        y = list()
        z = list()
        scalars = list()

        for index, label in enumerate(nodes):
            labels[label] = index
            this_scalar, this_x, this_y, this_z = nodes[label]
            scalars.append(atoms[this_scalar])
            x.append(float(this_x))
            y.append(float(this_y))
            z.append(float(this_z))

        connections = list()

        for start, stop in edges:
            norm = np.sqrt((nodes[stop][1] - nodes[start][1]) * (nodes[stop][1] - nodes[start][1]) +
                           (nodes[stop][2] - nodes[start][2]) * (nodes[stop][2] - nodes[start][2]) +
                           (nodes[stop][3] - nodes[start][3]) * (nodes[stop][3] - nodes[start][3]))
            if norm < 5:
                connections.append((labels[start], labels[stop]))

        sr = self.class_inputs["sr"]
        x = np.array(x) / sr
        y = np.array(y) / sr
        z = np.array(z) / sr
        cm_x, cm_y, cm_z = np.mean(x), np.mean(y), np.mean(z)
        cm_map_x, cm_map_y, cm_map_z = np.mean(self.coords_map[:, 0]), \
                                       np.mean(self.coords_map[:, 1]), \
                                       np.mean(self.coords_map[:, 2])
        scalars = np.array(scalars)
        return [x - cm_x + cm_map_x,
                y - cm_y + cm_map_y,
                z - cm_z + cm_map_z, np.asarray(connections), scalars]

    # ---------------------------------------------------------------------------
    # Write functions
    # ---------------------------------------------------------------------------

    # ---------------------------------------------------------------------------
    # View functions
    # ---------------------------------------------------------------------------

    # ---------------------------------------------------------------------------
    # Utils functions
    # ---------------------------------------------------------------------------

    # ---------------------------------------------------------------------------
    # The layout of the dialog created
    # ---------------------------------------------------------------------------
    view = View(HSplit(
        HGroup(
            Group(
                Item('scene3d',
                     editor=SceneEditor(scene_class=MayaviScene),
                     height=250, width=300),
                show_labels=False
            ),
            VGroup(
                Group(
                    Item('opacity',
                         editor=RangeEditor(format='%.02f', low_name='op_min', high_name='op_max', mode='slider')
                         )
                ),
                Group(
                    Item('show_atom_model'),
                ),
                Group(
                    Item('colormap',
                         editor=EnumEditor(
                             values={"Mean motion": "Mean motion",
                                     "Std motion": "Std motion"}
                         )),
                    Item('repaint', style='custom', show_label=False),
                    columns=2),
                Group(
                    Item('mask_factor', show_label=False, editor=TextEditor()),
                    Item('mask_arrows'),
                    show_labels=False, columns=2
                ),
                Group(
                    Item('scale_factor', show_label=False, editor=TextEditor()),
                    Item('scale_arrows'),
                    show_labels=False, columns=2
                ),
            ),
        ),
    ),
        resizable=True,
        title='Map rendering',
        icon=os.path.join(os.path.dirname(flexutils.__file__), "icon_square.png")
    )


# ---------------------------------------------------------------------------
# Parallel computation of motion statistics
# ---------------------------------------------------------------------------
def computation(z, L1, L2, coords, r):
    Z = computeBasis(L1=L1,
                     L2=L2,
                     pos=coords, r=r)
    A = resizeZernikeCoefficients(z)
    return Z @ A.T

def summers(z_space, L1, L2, coords, r, processes):
    pool = multiprocessing.Pool(processes=processes)
    pbar = tqdm(total=z_space.shape[0])

    class Sum:
        def __init__(self, coords):
            self.value_1 = np.zeros(coords.shape)
            self.value_2 = np.zeros(coords.shape)
            self.lock = _thread.allocate_lock()
            self.count = 0

        def add(self, value):
            self.count += 1
            self.lock.acquire()
            self.value_1 += value
            self.value_2 += value * value
            self.lock.release()
            pbar.update(1)

    sumArr = Sum(coords)
    for z in z_space:
        singlepoolresult = pool.apply_async(computation, (z, L1, L2, coords, r), callback=sumArr.add)

    pool.close()
    pool.join()

    return sumArr.value_1, sumArr.value_2
# ---------------------------------------------------------------------------


if __name__ == '__main__':
    import argparse

    # Input parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--i', type=str, required=True)
    parser.add_argument('--r', type=str, required=True)

    def float_or_str(s):
        try:
            return float(s)
        except ValueError:
            return s

    parsed, unknown = parser.parse_known_args()
    for arg in unknown:
        if arg.startswith(("-", "--")):
            # you can pass any arguments to add_argument
            parser.add_argument(arg.split('=')[0], type=float_or_str)

    args = parser.parse_args()

    # Input
    input_dict = vars(args)
    input_dict['metadata_file'] = args.i
    input_dict['map_file'] = args.r
    # input_dict['z_space'] = z_space
    # input_dict['interp_val'] = interp_val

    # Initialize volume slicer
    m = MapView(**input_dict)
    m.configure_traits()
