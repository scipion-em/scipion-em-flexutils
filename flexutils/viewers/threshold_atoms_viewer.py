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

from traits.api import HasTraits, Instance, Array, Float, String,\
    on_trait_change
from traits.trait_types import Button
from traitsui.api import View, Item, HGroup, Group, HSplit, VGroup, RangeEditor, TextEditor

from mayavi import mlab
from mayavi.core.api import PipelineBase
from mayavi.core.ui.api import SceneEditor, MayaviScene, \
    MlabSceneModel

import flexutils


################################################################################
# The object implementing the dialog
class ModelThresholdView(HasTraits):
    # Map view
    scene3d = Instance(MlabSceneModel, ())

    # Model (actor)
    ipw_pts = Instance(PipelineBase)
    ipw_pts_sel = Instance(PipelineBase)

    # Visualization style
    threshold = Float(0.0)
    thr_min = Float(0.0)
    thr_max = Float(1.0)

    # Output
    keep_pos = Array()

    # Text
    num_pts_text = String("")

    # Points properties
    scale_pts = Button("Scale spheres")
    scale_factor = String("3")

    # ---------------------------------------------------------------------------
    def __init__(self, **traits):
        self.class_inputs = traits
        super(ModelThresholdView, self).__init__(**traits)
        self.coords = self.class_inputs["coords"]
        self.keep_pos = np.ones(self.coords.shape[0], dtype=bool)
        self.num_pts_text = "Selected pos: %d (number of coords 3n: %d)" % \
                            (self.coords.shape[0], 3 * self.coords.shape[0])

    # ---------------------------------------------------------------------------
    # Default values
    # ---------------------------------------------------------------------------

    # ---------------------------------------------------------------------------
    # Scene activation callbaks
    # ---------------------------------------------------------------------------
    @on_trait_change('scene3d.activated')
    def display_scene3d(self):
        self.scene3d.mlab.view(40, 50)
        self.scene3d.scene.background = (0, 0, 0)

    @on_trait_change('scene3d.activated')
    def display_pts(self):
        # Xmipp convention is ZYX array
        z, y, x = self.coords[:, 2], self.coords[:, 1], self.coords[:, 0]
        pts = mlab.points3d(x, y, z, color=(1., 1., 1.), scale_factor=3., resolution=10, opacity=1.)
        setattr(self, 'ipw_pts', pts)

    @on_trait_change('scene3d.activated')
    def display_pts_sel(self):
        # Xmipp convention is ZYX array
        z, y, x = self.coords[:, 2], self.coords[:, 1], self.coords[:, 0]
        pts = mlab.points3d(x, y, z, color=(0., 0., 1.), scale_factor=4., resolution=10, opacity=1.)
        setattr(self, 'ipw_pts_sel', pts)

    @on_trait_change("threshold")
    def change_threshold(self):
        std_df = self.class_inputs["std_df"]
        quantile_std_pos = np.quantile(std_df, self.threshold)
        self.keep_pos = std_df >= quantile_std_pos

        pts = getattr(self, 'ipw_pts_sel')
        pts.mlab_source.points = self.coords[self.keep_pos, :]

        self.num_pts_text = "Selected pos: %d (number of coords 3n: %d)" % \
                            (np.sum(self.keep_pos), 3 * np.sum(self.keep_pos))

    def _scale_pts_fired(self):
        scale_factor = float(self.scale_factor)
        pts, pts_sel = getattr(self, 'ipw_pts'), getattr(self, 'ipw_pts_sel')
        pts.glyph.glyph.scale_factor = scale_factor
        pts_sel.glyph.glyph.scale_factor = scale_factor + 1.

    # ---------------------------------------------------------------------------
    # Conversion functions
    # ---------------------------------------------------------------------------

    # ---------------------------------------------------------------------------
    # Read functions
    # ---------------------------------------------------------------------------

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
                    Item('threshold',
                         editor=RangeEditor(format='%.02f', low_name='thr_min', high_name='thr_max', mode='slider')
                         )
                ),
                Group(
                    Item('scale_factor', show_label=False, editor=TextEditor()),
                    Item('scale_pts'),
                    show_labels=False, columns=2
                ),
                Group(
                    Item('num_pts_text', show_label=False, editor=TextEditor(read_only=True)),
                    show_labels=False, columns=1
                ),
            ),
        ),
    ),
        resizable=True,
        title='Interactive model thresholding',
        icon=os.path.join(os.path.dirname(flexutils.__file__), "icon_square.png")
    )
