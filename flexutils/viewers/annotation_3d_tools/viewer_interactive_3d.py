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
import shutil
from glob import glob
from xmipp_metadata.image_handler import ImageHandler
from copy import deepcopy

from PyQt5.QtGui import QIntValidator, QIcon
from PyQt5.QtWidgets import QLineEdit, QHBoxLayout, QSizePolicy
from PyQt5.QtCore import QThread

from sklearn.neighbors import KDTree
from sklearn.cluster import KMeans

from qtpy.QtWidgets import (
    QPushButton,
    QSplitter,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)
from packaging.version import parse as parse_version

import napari
from napari.components.viewer_model import ViewerModel
from napari.qt import QtViewer
from napari._qt.layer_controls import QtLayerControlsContainer

import flexutils
from flexutils.viewers.utils.pyqt_worker import GenerateVolumesWorker
from flexutils.viewers.chimera_viewers.viewer_morph_chimerax import FlexMorphChimeraX

NAPARI_GE_4_16 = parse_version(napari.__version__) > parse_version("0.4.16")


class QtViewerWrap(QtViewer):
    def __init__(self, main_viewer, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.main_viewer = main_viewer

    def _qt_open(
            self,
            filenames: list,
            stack: bool,
            plugin: str = None,
            layer_type: str = None,
            **kwargs,
    ):
        """for drag and drop open files"""
        self.main_viewer.window._qt_viewer._qt_open(
            filenames, stack, plugin, layer_type, **kwargs
        )


class MenuWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.save_btn = QPushButton("Save selections")
        self.cluster_btn = QPushButton("Compute KMeans")
        self.cluster_num = QLineEdit()
        self.morph_button = QPushButton("Morph ChimeraX")
        self.morph_button.setIcon(QIcon(os.path.join(os.path.dirname(flexutils.__file__), "chimerax_logo.png")))
        onlyInt = QIntValidator()
        onlyInt.setBottom(1)
        self.cluster_num.setValidator(onlyInt)
        layout_main = QVBoxLayout()
        layout_cluster = QHBoxLayout()
        widget_cluster = QWidget()
        layout_cluster.addWidget(self.cluster_num)
        layout_cluster.addWidget(self.cluster_btn)
        widget_cluster.setLayout(layout_cluster)
        layout_main.addWidget(self.save_btn)
        layout_main.addWidget(widget_cluster)
        layout_main.addWidget(self.morph_button)
        layout_main.addStretch(1)
        self.setLayout(layout_main)


class MultipleViewerWidget(QSplitter):
    """The main widget of the example."""

    def __init__(self, viewer: napari.Viewer):
        super().__init__()
        self.viewer = viewer
        self.viewer_model1 = ViewerModel(title="map_view", ndisplay=3)

        self.qt_viewer1 = QtViewerWrap(self.viewer_model1, self.viewer_model1)

        self.tab_widget = QTabWidget()
        self.menu_widget = MenuWidget()
        w1 = QtLayerControlsContainer(self.viewer_model1)
        self.tab_widget.addTab(w1, "Map view")
        self.tab_widget.addTab(self.menu_widget, "Menu")
        self.tab_widget.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Minimum)

        self.viewer.window.add_dock_widget(self.tab_widget, area="bottom")
        self.viewer.window.add_dock_widget(self.qt_viewer1, area="bottom")


class Annotate3D(object):

    def __init__(self, data, z_space, mode, path=".", **kwargs):
        # Prepare attributes
        self.class_inputs = kwargs
        self.data = data
        self.z_space = z_space
        self.path = path
        self.mode = mode
        self.prev_layers = None  # To keep track of old layers for callbacks
        self.last_selected = set()

        # Scale data to box of side 300
        bounding = np.amax(np.abs(np.amax(data, axis=0) - np.amin(data, axis=0)))
        self.data = (300 / bounding) * self.data

        # Create KDTree
        self.kdtree_data = KDTree(self.data)
        self.kdtree_z_pace = KDTree(self.z_space)

        # Create viewer
        self.view = napari.Viewer(ndisplay=3, title="Flexutils 3D Annotation")
        self.view.window._qt_window.setWindowIcon(QIcon(os.path.join(os.path.dirname(flexutils.__file__),
                                                                     "icon_square.png")))
        self.dock_widget = MultipleViewerWidget(self.view)

        # Set data in viewers
        points_layer = self.dock_widget.viewer.add_points(self.data, size=1, shading='spherical', edge_width=0,
                                                          antialiasing=0,
                                                          blending="additive", name="Landscape")
        points_layer.editable = False

        boxsize = int(self.class_inputs["boxsize"])
        dummy_vol = np.zeros((boxsize, boxsize, boxsize))
        self.dock_widget.viewer_model1.add_image(dummy_vol, name="Map", rendering="iso")

        # Selections layers
        self.reloadView()

        # Add callbacks
        self.dock_widget.menu_widget.save_btn.clicked.connect(self.saveSelections)
        self.dock_widget.menu_widget.cluster_btn.clicked.connect(self._compute_kmeans_fired)
        self.dock_widget.menu_widget.morph_button.clicked.connect(self._morph_chimerax_fired)
        self.dock_widget.viewer.layers.events.inserted.connect(self.on_insert_add_callback)

        # Worker threads
        self.thread_vol = None
        self.thread_chimerax = None

        # Run viewer
        napari.run()

    # ---------------------------------------------------------------------------
    # Read functions
    # ---------------------------------------------------------------------------
    def reloadView(self):
        pathFile = os.path.join(self.path, "selections_layers")

        if os.path.isdir(pathFile):
            for file in glob(os.path.join(pathFile, "*.csv")):
                self.dock_widget.viewer.open(file, layer_type="points", size=5)

        # Add callbacks
        for layer in self.dock_widget.viewer.layers:
            layer.events.data.connect(self.updateConformation)
            layer.events.highlight.connect(self.updateConformation)

    def readMap(self, file):
        map = ImageHandler().read(file).getData()
        return map


    # ---------------------------------------------------------------------------
    # Write functions
    # ---------------------------------------------------------------------------
    def writeVectorFile(self, vector):
        pathFile = os.path.join(self.path, self.vector_file)
        with open(pathFile, 'w') as fid:
            fid.write(' '.join(map(str, vector.reshape(-1))) + "\n")

    def saveSelections(self):
        pathFile = os.path.join(self.path, "selections_layers")
        selected_z = []

        if os.path.isdir(pathFile):
            shutil.rmtree(pathFile)

        os.mkdir(pathFile)

        for layer in self.dock_widget.viewer.layers:
            if layer.name != "Landscape":
                layer.save(os.path.join(pathFile, layer.name + ".csv"))

                points = layer.data
                _, inds = self.kdtree_data.query(points, k=1)
                inds = np.array(inds).flatten()
                selected_z.append(self.z_space[inds])

        selected_z = np.vstack(selected_z)
        np.savetxt(os.path.join(self.path, 'saved_selections.txt'), selected_z, delimiter=" ")


    # ---------------------------------------------------------------------------
    # Callbacks
    # ---------------------------------------------------------------------------
    def _compute_kmeans_fired(self):
        # Compute KMeans and save automatic selection
        n_clusters = int(self.dock_widget.menu_widget.cluster_num.text())
        clusters = KMeans(n_clusters=n_clusters).fit(self.z_space)
        centers = clusters.cluster_centers_
        self.interp_val = clusters.labels_
        _, inds = self.kdtree_z_pace.query(centers, k=1)
        inds = np.array(inds).flatten()
        selected_data = self.data[inds]
        self.dock_widget.viewer.add_points(selected_data, size=5, name="KMeans")

    def _morph_chimerax_fired(self):
        # Morph maps in chimerax based on different ordering methods
        if self.thread_chimerax is None:
            layer = self.dock_widget.viewer.layers.selection._current
            if layer.name != "Landscape":
                points = layer.data
                if points.shape[0] > 0:
                    _, inds = self.kdtree_data.query(points, k=1)
                    inds = np.array(inds).flatten()
                    sel_names = ["vol_%03d" % idx for idx in range(points.shape[0])]

                    args = (self.z_space[inds], sel_names, self.mode, self.path)

                    self.createThreadChimeraX(*args, **self.class_inputs)
                    self.thread_chimerax.start()

    def on_insert_add_callback(self, event):
        layer = event.value
        layer.events.data.connect(self.updateConformation)
        layer.events.highlight.connect(self.updateConformation)

    # ---------------------------------------------------------------------------
    # Update map functions
    # ---------------------------------------------------------------------------
    def updateConformation(self, event):
        if event.type != "highlight":
            # Update real time conformation
            pos = event.value
            layer_idx = event.index

            if self.prev_layers is not None and layer_idx < len(self.prev_layers):
                if self.prev_layers[layer_idx].shape[0] > pos.shape[0]:
                    return
                prev_layer = self.prev_layers[layer_idx]
                nrows, ncols = pos.shape
                dtype = {'names': ['f{}'.format(i) for i in range(ncols)],
                         'formats': ncols * [pos.dtype]}
                pos = np.setdiff1d(pos.view(dtype), prev_layer.view(dtype))
                pos = pos.view(prev_layer.dtype).reshape(-1, ncols)
        else:
            layer = event.source
            if layer.mode == 'select':
                selected = layer.selected_data
                if selected != self.last_selected and selected != set():
                    self.last_selected = selected
                    pos = layer.data[list(selected)[0]].reshape((1, -1))
                else:
                    return
            else:
                return

        _, ind = self.kdtree_data.query(pos, k=1)
        ind = np.array(ind).flatten()[0]

        # Get generation function arguments
        if self.mode == "Zernike3D":
            args = {"map": "reference.mrc", "mask": "mask.mrc", "output": "deformed.mrc",
                    "path": self.path, "z_clnm": self.z_space[ind, :],
                    "L1": int(self.class_inputs["L1"]), "L2": int(self.class_inputs["L2"]),
                    "Rmax": 32}
        elif self.mode == "CryoDrgn":
            args = {"zValues": self.z_space[ind, :], "weights": self.class_inputs["weights"],
                    "config": self.class_inputs["config"], "outdir": self.path,
                    "apix": float(self.class_inputs["sr"]), "flip": False,
                    "downsample": int(self.class_inputs["boxsize"]), "invert": False}
        elif self.mode == "HetSIREN":
            args = {"weigths_file": self.class_inputs["weights"], "x_het": self.z_space[ind, :],
                    "outdir": self.path, "step": self.class_inputs["step"]}
        elif self.mode == "NMA":
            args = {"weigths_file": self.class_inputs["weights"], "c_nma": self.z_space[ind, :],
                    "outdir": self.path, "sr": self.class_inputs["sr"]}

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
        self.thread_chimerax.started.connect(self.worker.showSalesMan)
        self.worker.finished.connect(self.thread_chimerax.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread_chimerax.finished.connect(self.thread_chimerax.deleteLater)
        self.thread_chimerax.finished.connect(self.removeThreadChimeraX)

    def removeThreadVol(self):
        self.thread_vol = None

    def removeThreadChimeraX(self):
        self.thread_chimerax = None

    def updateEmittedMap(self, map):
        self.dock_widget.viewer_model1.layers[0].data = map
        self.prev_layers = [layer.data.copy() for layer in self.dock_widget.viewer.layers]


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
    # interp_val = np.loadtxt(args.interp_val)

    # Input
    input_dict = vars(args)
    input_dict['data'] = data
    input_dict['z_space'] = z_space
    # input_dict['interp_val'] = interp_val

    # Initialize volume slicer
    Annotate3D(**input_dict)
