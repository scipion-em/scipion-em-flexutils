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
import pickle
from xmipp_metadata.image_handler import ImageHandler

from sklearn.neighbors import KDTree
from sklearn.cluster import KMeans

from traits.api import HasTraits, Instance, Array, Float, String,\
    on_trait_change, Callable
from traits.trait_types import Button, Enum
from traitsui.api import View, Item, HGroup, Group, HSplit, VGroup, RangeEditor, TextEditor, EnumEditor

from pyface.image_resource import ImageResource

from tvtk.api import tvtk

from mayavi import mlab
from mayavi.core.api import PipelineBase
from mayavi.core.ui.api import SceneEditor, MayaviScene, \
    MlabSceneModel

from PyQt5.QtCore import QThread

import flexutils
from flexutils.viewers.utils.pyqt_worker import GenerateVolumesWorker
from flexutils.viewers.chimera_viewers.viewer_morph_chimerax import FlexMorphChimeraX


################################################################################
# The object implementing the dialog
class PointCloudView(HasTraits):
    # The data to plot
    data = Array()
    grid = Array()

    # The values to interpolate
    interp_val = Array()

    # The 4 views displayed (Zernike space)
    scene3d = Instance(MlabSceneModel, ())

    # Real time conformation view
    scene_c = Instance(MlabSceneModel, ())

    # Line cuts for 3d view
    # line_3d_x = Instance(PipelineBase)
    # line_3d_y = Instance(PipelineBase)
    # line_3d_z = Instance(PipelineBase)

    # Point cloud (actor)
    ipw_pc = Instance(PipelineBase)
    ipw_sel = Instance(PipelineBase)

    # Map (actor)
    ipw_map = Instance(PipelineBase)

    # Visualization style
    opacity = Float(0.0)
    threshold = Float(0.0)
    op_min = Float(0.0)
    op_max = Float(1.0)

    _axis_names = dict(x=0, y=1, z=2)

    # Output path
    path = String()

    # Space parameters
    z_space = Array()
    vector_file = String('z_vw.txt')

    # For picking coefficients
    cursor_position = Array()

    # Map
    map = Array()
    generated_map = Array()

    # KMeans
    n_clusters = String("10")

    # Selections
    current_sel = String('origin')
    ipw_label = Instance(PipelineBase)
    save_file = String('saved_selections.txt')

    # Save selections
    save_selections = Button("Save selections")
    compute_kmeans = Button("Compute KMeans")

    # Generate map function
    mode = String()
    generate_map = Callable()

    # Points properties
    scale_pts = Button("Scale spheres")
    scale_factor = String("10")

    # ChimeraX morphing
    morphing_choice = Enum('Salesman', 'Random walk')
    morph_chimerax = Button(label="",
                            image=ImageResource(os.path.join(os.path.dirname(flexutils.__file__), "chimerax_logo.png")))

    # ---------------------------------------------------------------------------
    def __init__(self, **traits):
        self.class_inputs = traits
        super(PointCloudView, self).__init__(**traits)
        self.map
        self.ipw_pc
        self.ipw_map
        # self.line_3d_x
        # self.line_3d_y
        # self.line_3d_z
        self.path

        # Scale data to fix in box of side 300
        bounding = np.amax(np.abs(np.amax(self.data, axis=0) - np.amin(self.data, axis=0)))
        self.data = (300 / bounding) * self.data

        # Create KDTree
        self.kdtree_data = KDTree(self.data)
        self.kdtree_z_pace = KDTree(self.z_space)

        # Selections
        self.selections = dict()
        pathFile = os.path.join(self.path, "selections_dict.pkl")
        if os.path.isfile(pathFile):
            self.readSelectionsDict()

        _, idx = self.kdtree_data.query(np.asarray([0, 0, 0]).reshape(1, -1), k=1)
        self.selections['origin'] = idx[0][0]
        if not os.path.isfile(pathFile) and self.mode == "Zernike3D":
            for idx in range(int(self.class_inputs["n_vol"])):
                self.selections['class_%d' % (idx + 1)] = self.data.shape[0] - (idx + 1)

        # Worker threads
        self.thread_vol = None
        self.thread_chimerax = None

        self.ipw_sel

    # ---------------------------------------------------------------------------
    # Default values
    # ---------------------------------------------------------------------------
    # def make_line_3d(self, axis_name):
    #     if axis_name == "x":
    #         normal = np.array([1, 0, 0])
    #     elif axis_name == "y":
    #         normal = np.array([0, 1, 0])
    #     elif axis_name == "z":
    #         normal = np.array([0, 0, 1])
    #     white = (1, 1, 1)
    #     grid_size = np.asarray(self.grid.shape)
    #     point1 = 0.5 * grid_size * (1 - normal)
    #     point2 = grid_size * normal + point1
    #     x = [point1[0], point2[0]]
    #     y = [point1[1], point2[1]]
    #     z = [point1[2], point2[2]]
    #     line = mlab.plot3d(x, y, z, color=white, tube_radius=1., figure=self.scene3d.mayavi_scene)
    #     return line

    # def _line_3d_x_default(self):
    #     return self.make_line_3d('x')
    #
    # def _line_3d_y_default(self):
    #     return self.make_line_3d('y')
    #
    # def _line_3d_z_default(self):
    #     return self.make_line_3d('z')

    # def _ipw_pc_default(self):
    #     return self.display_space_cloud()

    # def _ipw_sel_default(self):
    #     return self.display_selections()

    # def _ipw_map_default(self):
    #     return self.display_map()

    def _map_default(self):
        # map = self.readMap(self.map_file)
        return np.zeros([64, 64, 64])

    # ---------------------------------------------------------------------------
    # Scene activation callbaks
    # ---------------------------------------------------------------------------
    @on_trait_change('scene3d.activated')
    def display_scene3d(self):
        # outline = mlab.pipeline.outline(self.data_src3d,
        #                                 figure=self.scene3d.mayavi_scene,
        #                                 )
        self.scene3d.mlab.view(40, 50)
        self.scene3d.scene.background = (0, 0, 0)
        # Keep the view always pointing up
        self.scene3d.scene.interactor.interactor_style = \
            tvtk.InteractorStyleTerrain()

        # Pick event
        # self.scene3d.scene.interactor.add_observer('KeyPressEvent', self.on_pick)
        self.picker = self.scene3d.mayavi_scene.on_mouse_pick(self.picker_callback)
        # Decrease the tolerance, so that we can more easily select a precise
        # point.
        self.picker.tolerance = 0.01

    @on_trait_change('scene_c.activated')
    def display_scene_c(self):
        self.scene_c.mlab.view(40, 50)
        self.scene_c.scene.background = (0, 0, 0)

    @on_trait_change('scene3d.activated')
    def display_space_cloud(self):
        scatter = mlab.pipeline.scalar_scatter(self.data[:, 2], self.data[:, 1], self.data[:, 0], self.interp_val,
                                               figure=self.scene3d.mayavi_scene)
        scatter = mlab.pipeline.glyph(scatter,
                                      opacity=self.opacity,
                                      scale_mode='none', scale_factor=10., mode='sphere', colormap="viridis",
                                      figure=self.scene3d.mayavi_scene)
        scatter.actor.actor.pickable = 0
        setattr(self, 'ipw_pc', scatter)

    @on_trait_change('scene3d.activated')
    def display_selections(self):
        data = self.data[list(self.selections.values())]
        scatter = mlab.points3d(data[:, 2], data[:, 1], data[:, 0],
                                scale_mode='none', scale_factor=10., mode='sphere', color=(1, 1, 1),
                                figure=self.scene3d.mayavi_scene)
        setattr(self, 'ipw_sel', scatter)

    @on_trait_change('scene3d.activated')
    def populateLabels(self):
        self.ipw_label = self.addLabel('origin')

    def addLabel(self, label):
        data = self.data[self.selections[label]]
        text = mlab.text3d(data[2] + 0.1, data[1] + 0.1, data[0] + 0.1, label, scale=10., color=(1, 1, 1),
                           figure=self.scene3d.mayavi_scene)
        return text

    @on_trait_change('scene_c.activated')
    def display_map(self):
        # Possible visualization
        # volume = mlab.pipeline.scalar_field(self.map, figure=self.scene_c.mayavi_scene)
        # volume = mlab.pipeline.volume(volume, figure=self.scene_c.mayavi_scene, color=(1, 1, 1),
        #                               vmin=0.0, vmax=0.05)

        # Chimera visualization (slow)
        volume = mlab.contour3d(self.map, color=(1, 1, 1), contours=1, figure=self.scene_c.mayavi_scene)
        volume.contour.auto_contours = False
        volume.contour.auto_update_range = True
        setattr(self, 'ipw_map', volume)

    @on_trait_change("opacity")
    def change_opacity_point_cloud(self):
        self.ipw_pc.actor.property.opacity = self.opacity

    @on_trait_change("threshold")
    def change_map_contour(self):
        volume = getattr(self, 'ipw_map')
        val_max = np.amax(volume.mlab_source.m_data.scalar_data)
        val_min = np.amin(volume.mlab_source.m_data.scalar_data)
        if val_max != val_min:
            contour = (val_max - val_min) * self.threshold + val_min
            volume.contour.contours = [contour]

    @on_trait_change("n_clusters")
    def change_n_clusters(self):
        if not str(self.n_clusters).isdigit():
            self.n_clusters = "10"

    # def on_pick(self, obj, evt):
    #     if obj.GetKeyCode() == "c":
    #         pos = np.asarray([self.cursor_position[2],
    #                           self.cursor_position[1],
    #                           self.cursor_position[0]]).reshape(1, -1)
    #         _, ind = self.kdtree_data.query(pos, k=1)
    #         self.selections.append(self.z_space[ind, :])

    def _save_selections_fired(self):
        pathFile = os.path.join(self.path, self.save_file)
        with open(pathFile, 'w') as fid:
            for key, idc in self.selections.items():
                if "origin" not in key and "class" not in key:
                    vector = self.z_space[idc]
                    fid.write(' '.join(map(str, vector.reshape(-1))) + "\n")
        self.saveSelectionsDict()
        np.savetxt(os.path.join(self.path, 'kmean_labels.txt'), self.interp_val)

    def _compute_kmeans_fired(self):
        # First delete all automatic selections associated to a previous KMeans
        for key in list(self.selections.keys()):
            if 'kmean' in key:
                del self.selections[key]
        # Compute KMeans and save automatic selection
        if int(self.n_clusters) > 0:
            clusters = KMeans(n_clusters=int(self.n_clusters)).fit(self.z_space)
            centers = clusters.cluster_centers_
            self.interp_val = clusters.labels_
            _, inds = self.kdtree_z_pace.query(centers, k=1)
            for idx, ind in enumerate(inds):
                self.selections["kmean_%d" % (idx + 1)] = ind[0]
        data = self.data[list(self.selections.values())]
        self.ipw_sel.mlab_source.reset(x=data[:, 2], y=data[:, 1], z=data[:, 0])
        self.ipw_pc.mlab_source.scalars = self.interp_val

    def _scale_pts_fired(self):
        scale_factor = float(self.scale_factor)
        pts, pts_sel = getattr(self, 'ipw_pc'), getattr(self, 'ipw_sel')
        pts.glyph.glyph.scale_factor = scale_factor
        pts_sel.glyph.glyph.scale_factor = scale_factor + 1.
        self.ipw_label.scale = np.array([scale_factor, scale_factor, scale_factor])

    def _morph_chimerax_fired(self):
        if self.thread_chimerax is None:
            # Morph maps in chimerax based on different ordering methods
            idm = np.asarray(list(self.selections.values()))
            sel_names = list(self.selections.keys())
            if len(self.selections) > 1:
                args = (self.z_space[idm], sel_names, self.mode, self.path)

                # Create worker in separate thread
                self.createThreadChimeraX(*args, **self.class_inputs)
                self.thread_chimerax.start()

    def picker_callback(self, picker):
        """ Picker callback: this get called when on pick events.
        """
        if picker.actors.last_actor in self.ipw_sel.actor.actors:
            # Find which data point corresponds to the point picked:
            # we have to account for the fact that each data point is
            # represented by a glyph with several points
            glyph_points = self.ipw_sel.glyph.glyph_source.glyph_source.output.points.to_array()
            point_id = int(picker.point_id / glyph_points.shape[0])
            # If the no points have been selected, we have '-1'
            if point_id != -1:
                # Retrieve the coordinates coorresponding to that data
                # point
                self.current_sel = list(self.selections.keys())[point_id]
                px, py, pz = self.ipw_sel.mlab_source.x[point_id], \
                             self.ipw_sel.mlab_source.y[point_id], \
                             self.ipw_sel.mlab_source.z[point_id]

                if self.thread_vol is None:
                    self.moveViewToPosition([px, py, pz])

                    # Update label
                    self.ipw_label.trait_set(position=[px + 0.1, py + 0.1, pz + 0.1], text=self.current_sel)
                    self.ipw_label.actor.property.color = (1, 1, 1)

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

    def readSelectionsDict(self):
        pathFile = os.path.join(self.path, "selections_dict.pkl")
        selections_dict_file = open(pathFile, "rb")
        self.selections = pickle.load(selections_dict_file)

    # ---------------------------------------------------------------------------
    # Write functions
    # ---------------------------------------------------------------------------
    def writeVectorFile(self, vector):
        pathFile = os.path.join(self.path, self.vector_file)
        with open(pathFile, 'w') as fid:
            fid.write(' '.join(map(str, vector.reshape(-1))) + "\n")

    def saveSelectionsDict(self):
        pathFile = os.path.join(self.path, "selections_dict.pkl")
        selections_dict_file = open(pathFile, "wb")
        pickle.dump(self.selections, selections_dict_file)
        selections_dict_file.close()

    # ---------------------------------------------------------------------------
    # View functions
    # ---------------------------------------------------------------------------
    def moveViewToPosition(self, position):
        # Update 3D line position
        # for axis, normal in zip(["x", "y", "z"], np.eye(3)):
        #     line3d = getattr(self, 'line_3d_%s' % axis)
        #     grid_size = np.asarray(self.grid.shape)
        #     position_np = np.asarray(position)
        #     point1 = position_np * (1 - normal)
        #     point2 = grid_size * normal + point1
        #     x = [point1[0], point2[0]]
        #     y = [point1[1], point2[1]]
        #     z = [point1[2], point2[2]]
        #     line3d.mlab_source.reset(x=x, y=y, z=z)

        # Update real time conformation
        pos = np.asarray([position[2], position[1], position[0]]).reshape(1, -1)
        _, ind = self.kdtree_data.query(pos, k=1)

        # Get generation function arguments
        if self.mode == "Zernike3D":
            args = {"map": "reference.mrc", "mask": "mask.mrc", "output": "deformed.mrc",
                    "path": self.path, "z_clnm": self.z_space[ind[0], :],
                    "L1": int(self.class_inputs["L1"]), "L2": int(self.class_inputs["L2"]),
                    "Rmax": 32}
        elif self.mode == "CryoDrgn":
            args = {"zValues": self.z_space[ind[0], :], "weights": self.class_inputs["weights"],
                    "config": self.class_inputs["config"], "outdir": self.path,
                    "apix": float(self.class_inputs["sr"]), "flip": False,
                    "downsample": int(self.class_inputs["boxsize"]), "invert": False}
        elif self.mode == "HetSIREN":
            args = {"weigths_file": self.class_inputs["weights"], "x_het": self.z_space[ind[0], :],
                    "outdir": self.path, "step": self.class_inputs["step"]}
        elif self.mode == "NMA":
            args = {"weigths_file": self.class_inputs["weights"], "c_nma": self.z_space[ind[0], :],
                    "outdir": self.path, "sr": self.class_inputs["sr"], "xsize": int(self.class_inputs["boxsize"])}

        # Create worker in separate thread
        self.createThread(**args)
        self.thread_vol.start()

    # ---------------------------------------------------------------------------
    # Worker generation
    # ---------------------------------------------------------------------------
    def createThread(self, **kwargs):
        self.thread_vol = QThread()
        self.worker = GenerateVolumesWorker(self.mode, **kwargs)
        self.worker.moveToThread(self.thread_vol)
        self.thread_vol.started.connect(self.worker.generateVolume)
        self.worker.volume.connect(self.updateEmittedMap)
        self.worker.finished.connect(self.thread_vol.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread_vol.finished.connect(self.thread_vol.deleteLater)
        self.thread_vol.finished.connect(self.removeThreadVol)

    def createThreadChimeraX(self, *args, **kwargs):
        self.thread_chimerax = QThread()
        self.worker = FlexMorphChimeraX(*args, **kwargs)
        self.worker.moveToThread(self.thread_chimerax)
        if self.morphing_choice == "Salesman":
            self.thread_chimerax.started.connect(self.worker.showSalesMan)
        elif self.morphing_choice == "Random walk":
            self.thread_chimerax.started.connect(self.worker.showRandomWalk)
        self.worker.finished.connect(self.thread_chimerax.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread_chimerax.finished.connect(self.thread_chimerax.deleteLater)
        self.thread_chimerax.finished.connect(self.removeThreadChimeraX)

    def removeThreadVol(self):
        self.thread_vol = None

    def removeThreadChimeraX(self):
        self.thread_chimerax = None

    def updateEmittedMap(self, map):
        self.generated_map = map

        volume = getattr(self, 'ipw_map')
        val_max = np.amax(volume.mlab_source.m_data.scalar_data)
        val_min = np.amin(volume.mlab_source.m_data.scalar_data)
        volume.mlab_source.reset(scalars=self.generated_map)
        if val_max == val_min:
            volume.contour.contours = [0.0001]

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
                         ),
                    Item('threshold',
                         editor=RangeEditor(format='%.02f', low_name='op_min', high_name='op_max', mode='slider')
                         ),
                    Group(
                        Item('n_clusters', show_label=False, editor=TextEditor()),
                        Item('compute_kmeans'),
                        show_labels=False, columns=2
                    ),
                    Group(
                        Item('scale_factor', show_label=False, editor=TextEditor()),
                        Item('scale_pts'),
                        show_labels=False, columns=2
                    ),
                    Group(
                        Item('save_selections'),
                        show_labels=False, columns=1
                    ),
                    Group(
                        Item('morphing_choice',
                             editor=EnumEditor(
                                 values={"Salesman": "Salesman",
                                         "Random walk": "Random walk"}
                             )),
                        Item('morph_chimerax', style='custom', show_label=False),
                    columns=2)
                ),
                Group(
                    Item('scene_c',
                         editor=SceneEditor(scene_class=MayaviScene),
                         height=250, width=300),
                    show_labels=False,
                )
            ),
        ),
    ),
        resizable=True,
        title='Point cloud clustering',
        icon=os.path.join(os.path.dirname(flexutils.__file__), "icon_square.png")
    )

if __name__ == '__main__':
    import argparse

    # Input parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, required=True)
    parser.add_argument('--z_space', type=str, required=True)
    parser.add_argument('--interp_val', type=str, required=True)
    parser.add_argument('--path', type=str, required=True)
    parser.add_argument('--mode', type=str, required=True)

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

    # Read and generate data
    data = np.loadtxt(args.data)
    z_space = np.loadtxt(args.z_space)
    interp_val = np.loadtxt(args.interp_val)

    # Input
    input_dict = vars(args)
    input_dict['data'] = data
    input_dict['z_space'] = z_space
    input_dict['interp_val'] = interp_val

    # Initialize volume slicer
    m = PointCloudView(**input_dict)
    m.configure_traits()
