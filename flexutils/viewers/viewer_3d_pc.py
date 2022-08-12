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
    on_trait_change
from traits.trait_types import Button
from traitsui.api import View, Item, HGroup, Group, HSplit, VGroup, RangeEditor, TextEditor

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
    op_min = Float(0.0)
    op_max = Float(1.0)

    _axis_names = dict(x=0, y=1, z=2)

    # Output path
    path = String()

    # Zernike parameters
    z_clnm = Array()
    l1 = Int()
    l2 = Int()
    d = Float()
    zernike_file = String('z_clnm_vw.txt')

    # Volume coefficients
    n_vol = Int()

    # For picking coefficients
    cursor_position = Array()

    # Map
    map = Array()
    map_file = String()
    deformed_map = Array()
    deformed_file = String('deformed.mrc')

    # Mask
    mask_file = String()

    # KMeans
    n_clusters = String("10")

    # Selections
    current_sel = String('reference')
    ipw_label = Instance(PipelineBase)
    save_file = String('saved_selections.txt')

    # Save selections
    save_selections = Button("Save selections")
    compute_kmeans = Button("Compute KMeans")

    # ---------------------------------------------------------------------------
    def __init__(self, **traits):
        super(PointCloudView, self).__init__(**traits)
        self.map
        self.ipw_pc
        self.ipw_map
        # self.line_3d_x
        # self.line_3d_y
        # self.line_3d_z
        self.path

        # Create KDTree
        self.kdtree_data = KDTree(self.data)
        self.kdtree_z_clnm = KDTree(self.z_clnm)

        # Selections
        pathFile = os.path.join(self.path, "selections_dict.pkl")
        if os.path.isfile(pathFile):
            self.readSelectionsDict()
        else:
            self.selections = dict()
            for idx in range(self.n_vol):
                if (idx+1) == self.n_vol:
                    self.selections['reference'] = self.data.shape[0] - (idx + 1)
                else:
                    self.selections['class_%d' % (idx+1)] = self.data.shape[0] - (idx + 1)

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

    def _ipw_pc_default(self):
        return self.display_zernike_space_cloud()

    def _ipw_sel_default(self):
        return self.display_selections()

    def _ipw_map_default(self):
        return self.display_map()

    def _map_default(self):
        map = self.readMap(self.map_file)
        return np.zeros(map.shape)

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
    def display_zernike_space_cloud(self):
        scatter = mlab.pipeline.scalar_scatter(self.data[:, 2], self.data[:, 1], self.data[:, 0], self.interp_val,
                                               figure=self.scene3d.mayavi_scene)
        scatter = mlab.pipeline.glyph(scatter,
                                      opacity=self.opacity,
                                      scale_mode='none', scale_factor=0.1, mode='sphere', colormap="viridis",
                                      figure=self.scene3d.mayavi_scene)
        scatter.actor.actor.pickable = 0
        setattr(self, 'ipw_pc', scatter)

    @on_trait_change('scene3d.activated')
    def display_selections(self):
        data = self.data[list(self.selections.values())]
        scatter = mlab.points3d(data[:, 2], data[:, 1], data[:, 0],
                                scale_mode='none', scale_factor=0.1, mode='sphere', color=(1, 1, 1),
                                figure=self.scene3d.mayavi_scene)
        setattr(self, 'ipw_sel', scatter)

    @on_trait_change('scene3d.activated')
    def populateLabels(self):
        self.ipw_label = self.addLabel('reference')

    def addLabel(self, label):
        data = self.data[self.selections[label]]
        text = mlab.text3d(data[2] + 0.1, data[1] + 0.1, data[0] + 0.1, label, scale=0.1, color=(1, 1, 1),
                           figure=self.scene3d.mayavi_scene)
        return text

    @on_trait_change('scene_c.activated')
    def display_map(self):
        # Possible visualization
        # volume = mlab.pipeline.scalar_field(self.map, figure=self.scene_c.mayavi_scene)
        # volume = mlab.pipeline.volume(volume, figure=self.scene_c.mayavi_scene, color=(1, 1, 1),
        #                               vmin=0.0, vmax=0.05)

        # Chimera visualization (slow)
        volume = mlab.contour3d(self.map, color=(1, 1, 1), figure=self.scene_c.mayavi_scene)
        setattr(self, 'ipw_map', volume)

    @on_trait_change("opacity")
    def change_opacity_point_cloud(self):
        self.ipw_pc.actor.property.opacity = self.opacity

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
    #         self.selections.append(self.z_clnm[ind, :])

    def _save_selections_fired(self):
        pathFile = os.path.join(self.path, self.save_file)
        with open(pathFile, 'w') as fid:
            fid.write(' '.join(map(str, [self.l1, self.l2, 0.5 * self.d])) + "\n")
            for idc in list(self.selections.values()):
                coeff = self.z_clnm[idc]
                fid.write(' '.join(map(str, coeff.reshape(-1))) + "\n")
        self.saveSelectionsDict()
        np.savetxt(os.path.join(self.path, 'deformation.txt'), self.interp_val)

    def _compute_kmeans_fired(self):
        # First delete all automatic selections associated to a previous KMeans
        for key in list(self.selections.keys()):
            if 'kmean' in key:
                del self.selections[key]
        # Compute KMeans and save automatic selection
        if int(self.n_clusters) > 0:
            clusters = KMeans(n_clusters=int(self.n_clusters), n_init=1).fit(self.z_clnm)
            centers = clusters.cluster_centers_
            self.interp_val = clusters.labels_
            _, inds = self.kdtree_z_clnm.query(centers, k=1)
            for idx, ind in enumerate(inds):
                self.selections["kmean_%d" % (idx + 1)] = ind[0]
        data = self.data[list(self.selections.values())]
        self.ipw_sel.mlab_source.reset(x=data[:, 2], y=data[:, 1], z=data[:, 0])
        self.ipw_pc.mlab_source.scalars = self.interp_val

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
        if getExt(self.map_file) == ".mrc":
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
    def writeZernikeFile(self, coeff):
        pathFile = os.path.join(self.path, self.zernike_file)
        with open(pathFile, 'w') as fid:
            fid.write(' '.join(map(str, [self.l1, self.l2, 0.5 * self.map.shape[0]])) + "\n")
            fid.write(' '.join(map(str, coeff.reshape(-1))) + "\n")

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
        pos = np.asarray([position[0], position[1], position[2]]).reshape(1, -1)
        _, ind = self.kdtree_data.query(pos, k=1)
        self.writeZernikeFile((self.map.shape[0] / self.d) * self.z_clnm[ind[0], :])
        pathFile = os.path.join(self.path, self.zernike_file)
        deformedFile = os.path.join(self.path, self.deformed_file)
        params = '-i %s --mask %s --step 1 --blobr 2 -o %s --clnm %s' % \
                 (self.map_file, self.mask_file, deformedFile, pathFile)
        xmipp3.Plugin.runXmippProgram('xmipp_volume_apply_coefficient_zernike3d', params)
        self.deformed_map = self.readMap(deformedFile)
        volume = getattr(self, 'ipw_map')
        volume.mlab_source.reset(scalars=self.deformed_map)
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
                    Group(
                        Item('n_clusters', show_label=False, editor=TextEditor()),
                        Item('compute_kmeans'),
                        show_labels=False, columns=2
                    ),
                    Group(
                        Item('save_selections'),
                        show_labels=False, columns=1
                    ),
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
    parser.add_argument('--coords', type=str, required=True)
    parser.add_argument('--z_clnm', type=str, required=True)
    parser.add_argument('--deformation', type=str, required=True)
    parser.add_argument('--path', type=str, required=True)
    parser.add_argument('--map_file', type=str, required=True)
    parser.add_argument('--mask_file', type=str, required=True)
    parser.add_argument('--l1', type=int, required=True)
    parser.add_argument('--l2', type=int, required=True)
    parser.add_argument('--d', type=int, required=True)
    parser.add_argument('--num_vol', type=int, required=True)

    args = parser.parse_args()

    # Read and generate data
    coords = np.loadtxt(args.coords)
    z_clnm = np.loadtxt(args.z_clnm)
    deformation = np.loadtxt(args.deformation)

    # Initialize volume slicer
    m = PointCloudView(data=coords, z_clnm=z_clnm, interp_val=deformation, path=args.path,
                     map_file=args.map_file, l1=args.l1,
                     l2=args.l2, d=args.d, n_vol=args.num_vol,
                     mask_file=args.mask_file)
    m.configure_traits()