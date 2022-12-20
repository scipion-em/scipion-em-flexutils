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

from sklearn.neighbors import KDTree
from sklearn.cluster import KMeans

import pyvista as pv

from traits.api import HasTraits, Instance, Array, Float, Int, Bool, String,\
    on_trait_change, Callable
from traits.trait_types import Button, Enum
from traitsui.api import View, Item, HGroup, Group, HSplit, VGroup, RangeEditor, TextEditor, EnumEditor

from pyface.image_resource import ImageResource

from tvtk.api import tvtk
from tvtk.pyface.scene import Scene

from mayavi import mlab
from mayavi.core.api import PipelineBase, Source
from mayavi.core.ui.api import SceneEditor, MayaviScene, \
    MlabSceneModel

from pwem.emlib.image import ImageHandler
from pyworkflow.utils import getExt

import xmipp3

import flexutils
from flexutils.viewers.viewer_morph_chimerax import FlexMorphChimeraX


################################################################################
# The object implementing the dialog
class VolumeSlicer(HasTraits):
    # The data to plot
    data = Array()
    grid = Array()

    # The values to interpolate
    interp_val = Array()

    # The 4 views displayed (Zernike space)
    scene3d = Instance(MlabSceneModel, ())
    scene_x = Instance(MlabSceneModel, ())
    scene_y = Instance(MlabSceneModel, ())
    scene_z = Instance(MlabSceneModel, ())

    # Real time conformation view
    scene_c = Instance(MlabSceneModel, ())

    # The data source
    data_src3d = Instance(Source)

    # The image plane widgets of the 3D scene
    ipw_3d_x = Instance(PipelineBase)
    ipw_3d_y = Instance(PipelineBase)
    ipw_3d_z = Instance(PipelineBase)

    # Line cuts for 3d view
    line_3d_x = Instance(PipelineBase)
    line_3d_y = Instance(PipelineBase)
    line_3d_z = Instance(PipelineBase)

    # Point cloud (actor)
    ipw_pc = Instance(PipelineBase)
    ipw_sel = Instance(PipelineBase)

    # Map (actor)
    ipw_map = Instance(PipelineBase)

    # Visualization style
    show_cut_planes = Bool(True)
    opacity = Float(0.0)
    threshold = Float(0.0)
    op_min = Float(0.0)
    op_max = Float(1.0)

    _axis_names = dict(x=0, y=1, z=2)

    # Output path
    path = String()

    # Zernike parameters
    z_space = Array()
    vector_file = String('z_vw.txt')

    # For picking coefficients
    cursor_position = Array()

    # Map
    map = Array()
    generated_map = Array()

    # Mask
    mask_file = String()

    # KMeans
    n_clusters = String("10")

    # Selections
    selection_name = String()
    # show_selections = Bool(True)
    current_sel = String('origin')
    ipw_label = Instance(PipelineBase)
    save_file = String('saved_selections.txt')

    # Save selections
    add_selection = Button("Add conformation")
    save_selections = Button("Save selections")
    remove_selection = Button("Remove conformation")
    rename_selection = Button("Rename selection")
    compute_kmeans = Button("Compute KMeans")

    # Generate map function
    mode = String()
    generate_map = Callable()

    # Points properties
    scale_pts = Button("Scale spheres")
    scale_factor = String("3")

    # ChimeraX morphing
    morphing_choice = Enum('Salesman', 'Random walk')
    morph_chimerax = Button(label="",
                            image=ImageResource(os.path.join(os.path.dirname(flexutils.__file__), "chimerax_logo.png")))

    # ---------------------------------------------------------------------------
    def __init__(self, **traits):
        self.class_inputs = traits
        super(VolumeSlicer, self).__init__(**traits)
        # Force the creation of the image_plane_widgets:
        self.interpolateGrid()
        self.moveDataToGrid()
        self.map
        self.ipw_3d_x
        self.ipw_3d_y
        self.ipw_3d_z
        self.ipw_pc
        self.ipw_map
        self.line_3d_x
        self.line_3d_y
        self.line_3d_z
        self.path

        # Create KDTree
        self.kdtree_data = KDTree(self.data)
        self.kdtree_z_space = KDTree(self.z_space)

        # Generate map
        self.mode
        if self.mode == "Zernike3D":
            from flexutils.protocols.xmipp.utils.utils import applyDeformationField
            self.generate_map = applyDeformationField
        elif self.mode == "CryoDrgn":
            import cryodrgn
            from cryodrgn.utils import generateVolumes
            cryodrgn.Plugin._defineVariables()
            self.generate_map = generateVolumes

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

        self.ipw_sel

    # ---------------------------------------------------------------------------
    # Default values
    # ---------------------------------------------------------------------------
    def _data_src3d_default(self):
        return mlab.pipeline.scalar_field(self.grid, figure=self.scene3d.mayavi_scene, colormap="viridis")

    def make_ipw_3d(self, axis_name):
        ipw = mlab.pipeline.image_plane_widget(self.data_src3d,
                                               figure=self.scene3d.mayavi_scene,
                                               transparent=True,
                                               colormap="black-white",
                                               plane_orientation='%s_axes' % axis_name)
        return ipw

    def make_line_3d(self, axis_name):
        if axis_name == "x":
            normal = np.array([1, 0, 0])
        elif axis_name == "y":
            normal = np.array([0, 1, 0])
        elif axis_name == "z":
            normal = np.array([0, 0, 1])
        white = (1, 1, 1)
        grid_size = np.asarray(self.grid.shape)
        point1 = 0.5 * grid_size * (1 - normal)
        point2 = grid_size * normal + point1
        x = [point1[0], point2[0]]
        y = [point1[1], point2[1]]
        z = [point1[2], point2[2]]
        line = mlab.plot3d(x, y, z, color=white, tube_radius=1., figure=self.scene3d.mayavi_scene)
        return line

    def _ipw_3d_x_default(self):
        return self.make_ipw_3d('x')

    def _ipw_3d_y_default(self):
        return self.make_ipw_3d('y')

    def _ipw_3d_z_default(self):
        return self.make_ipw_3d('z')

    def _line_3d_x_default(self):
        return self.make_line_3d('x')

    def _line_3d_y_default(self):
        return self.make_line_3d('y')

    def _line_3d_z_default(self):
        return self.make_line_3d('z')

    # def _ipw_pc_default(self):
    #     return self.display_space_cloud()
    #
    # def _ipw_sel_default(self):
    #     return self.display_selections()
    #
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
        # Interaction properties can only be changed after the scene
        # has been created, and thus the interactor exists
        for ipw in (self.ipw_3d_x, self.ipw_3d_y, self.ipw_3d_z):
            # Turn the interaction off
            ipw.ipw.interaction = 0
        self.scene3d.scene.background = (0, 0, 0)
        # Keep the view always pointing up
        self.scene3d.scene.interactor.interactor_style = \
            tvtk.InteractorStyleTerrain()

        # Pick event
        self.scene3d.scene.interactor.add_observer('KeyPressEvent', self.on_pick)
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
                                      scale_mode='none', scale_factor=3, mode='sphere', colormap="viridis",
                                      figure=self.scene3d.mayavi_scene)
        scatter.actor.actor.pickable = 0
        setattr(self, 'ipw_pc', scatter)

    @on_trait_change('scene3d.activated')
    def display_selections(self):
        data = self.data[list(self.selections.values())]
        scatter = mlab.points3d(data[:, 2], data[:, 1], data[:, 0],
                                scale_mode='none', scale_factor=3, mode='sphere', color=(1, 1, 1),
                                figure=self.scene3d.mayavi_scene)
        setattr(self, 'ipw_sel', scatter)

    @on_trait_change('scene3d.activated')
    def populateLabels(self):
        self.ipw_label = self.addLabel('origin')

    def addLabel(self, label):
        data = self.data[self.selections[label]]
        text = mlab.text3d(data[2] + 5., data[1] + 5., data[0] + 5., label, scale=3, color=(1, 1, 1),
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

    def make_side_view(self, axis_name):
        scene = getattr(self, 'scene_%s' % axis_name)

        # To avoid copying the data, we take a reference to the
        # raw VTK dataset, and pass it on to mlab. Mlab will create
        # a Mayavi source from the VTK without copying it.
        # We have to specify the figure so that the data gets
        # added on the figure we are interested in.
        outline = mlab.pipeline.outline(
            self.data_src3d.mlab_source.dataset,
            figure=scene.mayavi_scene,
        )
        ipw = mlab.pipeline.image_plane_widget(
            outline,
            colormap="viridis",
            plane_orientation='%s_axes' % axis_name)
        setattr(self, 'ipw_%s' % axis_name, ipw)

        # Synchronize positions between the corresponding image plane
        # widgets on different views.
        ipw.ipw.sync_trait('slice_position',
                           getattr(self, 'ipw_3d_%s' % axis_name).ipw)

        # Make left-clicking create a crosshair
        ipw.ipw.left_button_action = 0

        # Add a callback on the image plane widget interaction to
        # move the others
        def move_view(obj, evt):
            # Update 3D plane position
            position = obj.GetCurrentCursorPosition()
            self.moveViewToPosition(position, axis_name)



        ipw.ipw.add_observer('InteractionEvent', move_view)
        ipw.ipw.add_observer('StartInteractionEvent', move_view)

        # Center the image plane widget
        ipw.ipw.slice_position = 0.5 * self.grid.shape[
            self._axis_names[axis_name]]

        # Position the view for the scene
        views = dict(x=(0, 90),
                     y=(90, 90),
                     z=(0, 0),
                     )
        scene.mlab.view(*views[axis_name])
        # 2D interaction: only pan and zoom
        scene.scene.interactor.interactor_style = \
            tvtk.InteractorStyleImage()
        scene.scene.background = (0, 0, 0)

    @on_trait_change('scene_x.activated')
    def display_scene_x(self):
        return self.make_side_view('x')

    @on_trait_change('scene_y.activated')
    def display_scene_y(self):
        return self.make_side_view('y')

    @on_trait_change('scene_z.activated')
    def display_scene_z(self):
        return self.make_side_view('z')

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

    @on_trait_change("selection_name")
    def change_selection_label(self):
        self.ipw_label.trait_set(text=self.selection_name)
        self.ipw_label.actor.property.color = (0, 0.76, 1)

    @on_trait_change("n_clusters")
    def change_n_clusters(self):
        if not str(self.n_clusters).isdigit():
            self.n_clusters = "10"

    @on_trait_change("show_cut_planes")
    def update_cut_plane_visibility(self):
        self.ipw_3d_x.visible = self.show_cut_planes
        self.ipw_3d_y.visible = self.show_cut_planes
        self.ipw_3d_z.visible = self.show_cut_planes

    # @on_trait_change("show_selections")
    # def update_cut_plane_visibility(self):
    #     self.ipw_sel.visible = self.show_selections
    #     self.ipw_label.visible = self.show_selections

    def on_pick(self, obj, evt):
        if obj.GetKeyCode() == "c":
            pos = np.asarray([self.cursor_position[2],
                              self.cursor_position[1],
                              self.cursor_position[0]]).reshape(1, -1)
            _, ind = self.kdtree_data.query(pos, k=1)
            self.selections.append(self.z_space[ind, :])

    def _save_selections_fired(self):
        pathFile = os.path.join(self.path, self.save_file)
        with open(pathFile, 'w') as fid:
            for key, idc in self.selections.items():
                if "origin" not in key and "class" not in key:
                    vector = self.z_space[idc]
                    fid.write(' '.join(map(str, vector.reshape(-1))) + "\n")
        self.saveSelectionsDict()

    def _add_selection_fired(self):
        pos = np.asarray([self.cursor_position[2],
                          self.cursor_position[1],
                          self.cursor_position[0]]).reshape(1, -1)
        _, ind = self.kdtree_data.query(pos, k=1)
        key = "conf_%d" % (len(self.selections) + 1)
        self.selections[key] = ind[0, 0]

        data = self.data[list(self.selections.values())]
        self.ipw_sel.mlab_source.reset(x=data[:, 2], y=data[:, 1], z=data[:, 0])

    def _compute_kmeans_fired(self):
        # First delete all automatic selections associated to a previous KMeans
        for key in list(self.selections.keys()):
            if 'kmean' in key:
                del self.selections[key]
        # Compute KMeans and save automatic selection
        if int(self.n_clusters) > 0:
            clusters = KMeans(n_clusters=int(self.n_clusters), n_init=1).fit(self.z_space)
            centers = clusters.cluster_centers_
            _, inds = self.kdtree_z_space.query(centers, k=1)
            for idx, ind in enumerate(inds):
                self.selections["kmean_%d" % (idx + 1)] = ind[0]
        data = self.data[list(self.selections.values())]
        self.ipw_sel.mlab_source.reset(x=data[:, 2], y=data[:, 1], z=data[:, 0])

    def _remove_selection_fired(self):
        del self.selections[self.current_sel]
        self.current_sel = ''
        data = self.data[list(self.selections.values())]
        self.ipw_sel.mlab_source.reset(x=data[:, 2], y=data[:, 1], z=data[:, 0])
        self.ipw_label.trait_set(text=self.current_sel)
        self.ipw_label.actor.property.color = (1, 1, 1)

    def _rename_selection_fired(self):
        if self.selection_name not in list(self.selections.keys()):
            self.selections[self.selection_name] = self.selections.pop(self.current_sel)
            self.current_sel = self.selection_name
            self.ipw_label.trait_set(text=self.current_sel)
            self.ipw_label.actor.property.color = (1, 1, 1)
            data = self.data[list(self.selections.values())]
            self.ipw_sel.mlab_source.reset(x=data[:, 2], y=data[:, 1], z=data[:, 0])

    def _scale_pts_fired(self):
        scale_factor = float(self.scale_factor)
        pts, pts_sel = getattr(self, 'ipw_pc'), getattr(self, 'ipw_sel')
        pts.glyph.glyph.scale_factor = scale_factor
        pts_sel.glyph.glyph.scale_factor = scale_factor + 1.
        self.ipw_label.scale = np.array([scale_factor, scale_factor, scale_factor])

    def _morph_chimerax_fired(self):
        # Morph maps in chimerax based on different ordering methods
        idm = np.asarray(list(self.selections.values()))
        sel_names = list(self.selections.keys())
        if len(self.selections) > 1:
            if self.morphing_choice == "Salesman":
                print("Salesman chosen")
                morph_chimerax = FlexMorphChimeraX(self.z_space[idm], sel_names, self.mode, self.path, **self.class_inputs)
                morph_chimerax.showSalesMan()
            elif self.morphing_choice == "Random walk":
                print("Random walk chosen")
                morph_chimerax = FlexMorphChimeraX(self.z_space[idm], sel_names, self.mode, self.path,
                                                   **self.class_inputs)
                morph_chimerax.showRandomWalk()

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

                self.moveViewToPosition([px, py, pz], None)

                # Update label
                self.ipw_label.trait_set(position=[px + 5., py + 5., pz + 5.], text=self.current_sel)
                self.ipw_label.actor.property.color = (1, 1, 1)

    # ---------------------------------------------------------------------------
    # Conversion functions
    # ---------------------------------------------------------------------------
    def interpolateGrid(self):
        mesh = pv.PolyData(self.data)
        mesh["interp_val"] = self.interp_val
        mesh.set_active_scalars("interp_val")
        grid = pv.create_grid(mesh, dimensions=(300, 300, 300))
        self.origin = grid.origin
        self.sr = grid.spacing
        interpolated = grid.interpolate(mesh, radius=0.1, sharpness=1)
        self.grid = interpolated.active_scalars.reshape((300, 300, 300))

    def moveDataToGrid(self):
        self.data[:, 0] = (self.data[:, 0] - self.origin[0]) / self.sr[0]
        self.data[:, 1] = (self.data[:, 1] - self.origin[1]) / self.sr[1]
        self.data[:, 2] = (self.data[:, 2] - self.origin[2]) / self.sr[2]

    def getDataAtThreshold(self, data, level):
        return np.asarray(np.where(data >= level))

    # ---------------------------------------------------------------------------
    # Read functions
    # ---------------------------------------------------------------------------
    def readMap(self, file):
        if getExt(file) == ".mrc":
            map = ImageHandler().read(file + ":mrc").getData()
        else:
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
    def moveViewToPosition(self, position, axis_name):
        setattr(self, 'cursor_position', np.asarray(position))
        for other_axis, axis_number in self._axis_names.items():
            if other_axis == axis_name:
                continue
            ipw3d = getattr(self, 'ipw_3d_%s' % other_axis)
            ipw3d.ipw.slice_position = position[axis_number]

        # Update 3D line position
        for axis, normal in zip(["x", "y", "z"], np.eye(3)):
            line3d = getattr(self, 'line_3d_%s' % axis)
            grid_size = np.asarray(self.grid.shape)
            position_np = np.asarray(position)
            point1 = position_np * (1 - normal)
            point2 = grid_size * normal + point1
            x = [point1[0], point2[0]]
            y = [point1[1], point2[1]]
            z = [point1[2], point2[2]]
            line3d.mlab_source.reset(x=x, y=y, z=z)

        # Update real time conformation
            pos = np.asarray([position[0], position[1], position[2]]).reshape(1, -1)
            _, ind = self.kdtree_data.query(pos, k=1)

            if self.generate_map is not None:
                if self.mode == "Zernike3D":
                    self.generate_map("reference.mrc", "mask.mrc", "deformed.mrc",
                                      self.path, self.z_space[ind[0], :],
                                      int(self.class_inputs["L1"]), int(self.class_inputs["L2"]), 32)
                    self.generated_map = self.readMap(os.path.join(self.path, "deformed.mrc"))
                elif self.mode == "CryoDrgn":
                    self.generate_map(self.z_space[ind[0], :], self.class_inputs["weights"],
                                      self.class_inputs["config"], self.path,
                                      downsample=int(self.class_inputs["boxsize"]),
                                      apix=float(self.class_inputs["sr"]))
                    self.generated_map = self.readMap(os.path.join(self.path, "vol_000.mrc"))

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
                Item('scene_y',
                     editor=SceneEditor(scene_class=Scene),
                     height=250, width=300),
                Item('scene_z',
                     editor=SceneEditor(scene_class=Scene),
                     height=250, width=300),
                show_labels=False,
            ),
            Group(
                Item('scene_x',
                     editor=SceneEditor(scene_class=Scene),
                     height=250, width=300),
                Item('scene3d',
                     editor=SceneEditor(scene_class=MayaviScene),
                     height=250, width=300),
                show_labels=False,
            ),
            VGroup(
                Group(
                    Group(
                        Item('show_cut_planes'),
                        # Item('show_selections'),
                        columns=2
                    ),
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
                        Item('add_selection'),
                        Item('remove_selection'),
                        Item('selection_name', show_label=False, editor=TextEditor()),
                        Item('rename_selection'),
                        show_labels=False, columns=3
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
        title='Volume Slicer',
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
    m = VolumeSlicer(**input_dict)
    m.configure_traits()
