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
from scipy import signal
from sklearn.decomposition import PCA
from umap import ParametricUMAP
from umap.parametric_umap import load_ParametricUMAP
import os
import shutil
from glob import glob
import psutil
import tensorflow as tf

from xmipp_metadata.image_handler import ImageHandler
from matplotlib.pyplot import get_cmap
from matplotlib.path import Path
import pickle
import warnings

from PyQt5.QtGui import QIntValidator, QIcon
from PyQt5.QtWidgets import QLineEdit, QHBoxLayout, QSizePolicy, QComboBox, QApplication
from PyQt5.QtCore import QThread

from sklearn.neighbors import KDTree
from sklearn.cluster import MiniBatchKMeans

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
from napari._qt.utils import _maybe_allow_interrupt
from napari.utils.notifications import show_warning, notification_manager
from napari.utils import progress

from magicgui.widgets import ComboBox, Container, Slider, Button

from pyworkflow.utils import moveFile

import flexutils
from flexutils.viewers.chimera_viewers.viewer_morph_chimerax import FlexMorphChimeraX
from flexutils.viewers.utils.pyqt_socket_threads import ServerQThread, ClientQThread
from flexutils.socket.server import Server


NAPARI_GE_4_16 = parse_version(napari.__version__) > parse_version("0.4.16")


class PCA_UMAP:
    '''Auxiliar class to align an UMAP cloud along its principal components'''
    def __init__(self, umap):
        self.umap = umap
        self.pca = PCA(n_components=umap.n_components)

    def fit(self, data):
        self.pca.fit(self.umap.transform(data))

    def transform(self, data):
        return self.pca.transform(self.umap.transform(data))

    def inverse_transform(self, data):
        return self.umap.inverse_transform(self.pca.inverse_transform(data))


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
    def __init__(self, ndims):
        super().__init__()
        dims = [f"Dim {dim + 1}" for dim in range(ndims)]
        self.save_btn = QPushButton("Save selections")
        self.cluster_btn = QPushButton("Compute KMeans")
        self.cluster_num = QLineEdit()
        self.dimension_sel = QComboBox()
        _ = [self.dimension_sel.addItem(item) for item in dims]
        self.dimension_sel.setCurrentIndex(0)
        self.dimension_btn = QPushButton("Cluster along PCA dimension")
        self.morph_button = QPushButton("Morph ChimeraX")
        self.morph_button.setIcon(QIcon(os.path.join(os.path.dirname(flexutils.__file__), "chimerax_logo.png")))
        onlyInt = QIntValidator()
        onlyInt.setBottom(1)
        self.cluster_num.setValidator(onlyInt)
        layout_main = QVBoxLayout()
        layout_cluster = QHBoxLayout()
        widget_cluster = QWidget()
        layout_button = QHBoxLayout()
        widget_button = QWidget()
        layout_cluster.addWidget(self.cluster_num)
        layout_cluster.addWidget(self.dimension_sel)
        layout_button.addWidget(self.cluster_btn)
        layout_button.addWidget(self.dimension_btn)
        widget_cluster.setLayout(layout_cluster)
        widget_button.setLayout(layout_button)
        layout_main.addWidget(self.save_btn)
        layout_main.addWidget(widget_cluster)
        layout_main.addWidget(widget_button)
        layout_main.addWidget(self.morph_button)
        layout_main.addStretch(1)
        self.setLayout(layout_main)


class MultipleViewerWidget(QSplitter):
    """The main widget of the example."""

    def __init__(self, viewer: napari.Viewer, ndims, interactive):
        super().__init__()
        self.viewer = viewer

        if interactive:
            self.viewer_model1 = ViewerModel(title="map_view", ndisplay=3)

            self.qt_viewer1 = QtViewerWrap(self.viewer_model1, self.viewer_model1)

            self.tab_widget = QTabWidget()
            self.menu_widget = MenuWidget(ndims)
            dims = [f"Dim {dim + 1}" for dim in range(ndims)]
            items = [("X axis", dims), ("Y axis", dims), ("Z axis", dims)]
            value = ["Dim 1", "Dim 2", "Dim 3"]
            self.right_widgets = [ComboBox(choices=c, label=l, value=val) for [l, c], val in zip(items, value)]
            self.right_widgets.append(Slider(value=20, min=1, max=50, label="Landscape-Vol sigma"))
            self.right_widgets.append(Button(label="Extract selection to layer"))
            self.right_widgets.append(ComboBox(choices=[], label="# layer"))
            self.right_widgets.append(Button(label="Add selection to # layer"))
            self.right_widgets.append(Slider(value=100, min=1, max=300, label="Landscape-Vol-Labels #"))
            self.select_axis_container = Container(widgets=self.right_widgets)
            w1 = QtLayerControlsContainer(self.viewer_model1)
            self.tab_widget.addTab(w1, "Map view")
            self.tab_widget.addTab(self.menu_widget, "Menu")
            self.tab_widget.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Minimum)

            self.viewer.window.add_dock_widget(self.tab_widget, area="bottom")
            self.viewer.window.add_dock_widget(self.qt_viewer1, area="bottom")
            self.viewer.window.add_dock_widget(self.select_axis_container, area="right", name="Landscape controls")


class Annotate3D(object):

    def __init__(self, data, z_space, mode=None, path=".", interactive=True, reduce="umap", **kwargs):
        # Prepare attributes
        self.class_inputs = kwargs
        # self.pca_data = data
        self.data = data
        self.z_space = z_space
        self.path = path
        self.mode = mode
        self.prev_layers = None  # To keep track of old layers for callbacks
        self.last_selected = set()
        self.current_axis = [0, 1, 2]
        boxsize = 128

        # Keyboard attributes
        self.control_pressed = False
        self.alt_pressed = False

        # Scale data to box of side 300
        # bounding = np.amax(np.abs(np.amax(data, axis=0) - np.amin(data, axis=0)))
        # self.data = (300 / bounding) * self.data
        # OldRange = (np.amax(self.data) - np.amin(self.data))
        # NewRange = ((boxsize - 5) - 5)
        # self.data = (boxsize - 1) * (self.data - np.amin(self.data)) / (np.amax(self.data) - np.amin(self.data))
        # self.data = (((self.data - np.amin(self.data)) * NewRange) / OldRange) + 5

        # Create viewer
        self.view = napari.Viewer(ndisplay=3, title="Flexutils 3D Annotation")
        self.view.window._qt_window.setWindowIcon(QIcon(os.path.join(os.path.dirname(flexutils.__file__),
                                                                     "icon_square.png")))
        self.dock_widget = MultipleViewerWidget(self.view, self.data.shape[1], interactive=interactive)

        # Load in view or interactive mode
        self.view.window.qt_viewer.dockLayerControls.setVisible(interactive)

        # PCA for clustering along dimension
        if reduce == "pca":
            self.transformer = PCA(n_components=self.data.shape[1])
            self.transformer.fit(self.z_space)
            self.transformer_data = self.transformer.transform(self.z_space)
        elif reduce == "umap":
            if os.path.isdir(self.class_inputs["umap_weights"]):
                self.transformer = load_ParametricUMAP(self.class_inputs["umap_weights"])
            else:
                self.transformer = ParametricUMAP(n_components=self.data.shape[1],
                                                  autoencoder_loss=False, parametric_reconstruction=True,
                                                  parametric_reconstruction_loss_fcn=tf.keras.losses.MSE,
                                                  global_correlation_loss_weight=1.0)
                self.transformer.fit(self.z_space)
            self.transformer = PCA_UMAP(self.transformer)
            self.transformer.fit(self.z_space)
            self.transformer_data = self.transformer.transform(self.z_space)
            self.data = np.copy(self.transformer_data)

        # Scale data to box of side 300
        self.data = (boxsize - 1) * (self.data - np.amin(self.data)) / (np.amax(self.data) - np.amin(self.data))

        # Create KDTree
        self.kdtree_data = KDTree(self.data[:, :3])
        self.kdtree_z_pace = KDTree(self.z_space)

        # Set data in viewers
        points_layer = self.dock_widget.viewer.add_points(np.copy(self.data[:, :3]), size=1, shading='spherical',
                                                          edge_width=0,
                                                          antialiasing=0,
                                                          blending="additive", name="Landscape")
        points_layer.editable = True

        # Set extra data layer (like priors) in viewer
        if "z_space_vol" in self.class_inputs:
            extra_data = np.loadtxt(self.class_inputs["z_space_vol"])
            _, inds = self.kdtree_z_pace.query(extra_data, k=1)
            inds = np.array(inds).flatten()
            extra_data = np.copy(self.data)[inds, :3]
            extra_layer = self.dock_widget.viewer.add_points(extra_data, size=1, shading='spherical',
                                                             edge_width=0,
                                                             antialiasing=0,
                                                             visible=False,
                                                             blending="additive", name="Priors")
            extra_layer.editable = True

        # Create volume from data points
        indeces = np.round(self.data).astype(int)
        vol = np.zeros((boxsize, boxsize, boxsize))
        vol[indeces[:, 0], indeces[:, 1], indeces[:, 2]] += 1

        # Filter volume
        std = np.pi * np.sqrt(boxsize) / 10.0
        gauss_1d = signal.windows.gaussian(boxsize, std)
        kernel = np.einsum('i,j,k->ijk', gauss_1d, gauss_1d, gauss_1d)
        kernel = np.pad(kernel, (5, 5))
        vol = np.pad(vol, (5, 5))
        ft_vol = np.fft.fftshift(np.fft.fftn(vol))
        ft_vol_real = np.real(ft_vol) * kernel
        ft_vol_imag = np.imag(ft_vol) * kernel
        ft_vol = ft_vol_real + 1j * ft_vol_imag
        vol = np.real(np.fft.ifftn(np.fft.ifftshift(ft_vol)))[5:133, 5:133, 5:133]

        # Label generation
        clusters = MiniBatchKMeans(n_clusters=100).fit(indeces)
        values = clusters.labels_ + 1
        labels = np.zeros((boxsize, boxsize, boxsize))
        labels[indeces[:, 0], indeces[:, 1], indeces[:, 2]] += values
        labels = labels.astype(int)

        # Add volume and labels
        self.view.add_image(vol, rgb=False, colormap="inferno", name="Landscape-Vol", opacity=0.5)
        self.view.add_labels(labels, name='Landscape-Vol-Labels', visible=False)
        # vol_layer.editable = False

        if interactive:
            if "boxsize" in self.class_inputs.keys():
                boxsize = int(self.class_inputs["boxsize"])
            else:
                boxsize = 64
            dummy_vol = np.zeros((boxsize, boxsize, boxsize))
            self.dock_widget.viewer_model1.add_image(dummy_vol, name="Map", rendering="mip")

        # Selections layers
        if interactive:
            self.reloadView()

        # Add callbacks
        if interactive:
            self.dock_widget.viewer.mouse_drag_callbacks.append(self.lassoSelector)
            self.dock_widget.viewer.bind_key("Control", self.control_detection)
            self.dock_widget.viewer.bind_key("Alt", self.alt_detection)
            self.dock_widget.menu_widget.save_btn.clicked.connect(self.saveSelections)
            self.dock_widget.menu_widget.cluster_btn.clicked.connect(self._compute_kmeans_fired)
            self.dock_widget.menu_widget.dimension_btn.clicked.connect(self._compute_dim_cluster_fired)
            self.dock_widget.menu_widget.morph_button.clicked.connect(self._morph_chimerax_fired)
            self.dock_widget.viewer.layers.events.inserted.connect(self.on_insert_add_callback)
            self.dock_widget.right_widgets[0].changed.connect(lambda event: self.selectAxis(0, event))
            self.dock_widget.right_widgets[1].changed.connect(lambda event: self.selectAxis(1, event))
            self.dock_widget.right_widgets[2].changed.connect(lambda event: self.selectAxis(2, event))
            self.dock_widget.right_widgets[3].changed.connect(self.updateVolSigma)
            self.dock_widget.right_widgets[5].choices = self.getLayerChoices
            self.dock_widget.right_widgets[4].changed.connect(self.extractSelectionToLayer)
            self.dock_widget.right_widgets[6].changed.connect(self.addSelectionToLayer)
            self.dock_widget.right_widgets[7].changed.connect(self.updateVolLabels)

            # Worker threads
            self.thread_chimerax = None

            # Volume generation socket
            if self.mode == "Zernike3D":
                flexutils.Plugin._defineVariables()
                metadata = {"mask": os.path.join(self.path, "mask_reference_original.mrc"),
                            "volume": os.path.join(self.path, "reference_original.mrc"),
                            "L1": self.class_inputs["L1"],
                            "L2": self.class_inputs["L2"],
                            "boxSize": ImageHandler(os.path.join(self.path, "reference_original.mrc")).getDimensions()[-1],
                            "outdir": self.path}
                program = flexutils.Plugin.getProgram(os.path.join(flexutils.__path__[0],
                                                                   "socket", "server.py"), python=True)
                env = flexutils.Plugin.getEnviron()
            elif self.mode == "HetSIREN":
                flexutils.Plugin._defineVariables()
                metadata = {"weights": self.class_inputs["weights"],
                            "lat_dim": self.z_space.shape[1],
                            "architecture": self.class_inputs["architecture"],
                            "outdir": self.path}
                program = flexutils.Plugin.getTensorflowProgram(os.path.join(flexutils.__path__[0],
                                                                             "socket", "server.py"), python=True)
                env = flexutils.Plugin.getEnviron()
            elif self.mode == "CryoDrgn":
                import cryodrgn
                cryodrgn.Plugin._defineVariables()
                metadata = {"weights": self.class_inputs["weights"],
                            "config": self.class_inputs["config"], "outdir": self.path}
                program = cryodrgn.Plugin.getActivationCmd() + " && python " + \
                          os.path.join(flexutils.__path__[0], "socket", "server.py")
                env = cryodrgn.Plugin.getEnviron()

            metadata_file = os.path.join(self.path, "metadata.p")
            with open(metadata_file, 'wb') as fp:
                pickle.dump(metadata, fp, protocol=pickle.HIGHEST_PROTOCOL)

            # Start server
            self.port = Server.getFreePort()
            self.server = ServerQThread(program, metadata_file, self.mode, self.port, env)
            self.server.start()

            # Start client
            self.client = ClientQThread(self.port, self.path, self.mode)
            self.client.volume.connect(self.updateEmittedMap)
            self.client.chimera.connect(self.launchChimeraX)

        # Run viewer
        self.app = QApplication.instance()

        if interactive:
            self.app.aboutToQuit.connect(self.on_close_callback)

        with notification_manager, _maybe_allow_interrupt(self.app):
            self.app.exec_()
        # napari.run()

    # ---------------------------------------------------------------------------
    # Read functions
    # ---------------------------------------------------------------------------
    def reloadView(self):
        pathFile = os.path.join(self.path, "selections_layers")

        if os.path.isdir(pathFile):
            for file in glob(os.path.join(pathFile, "*")):
                print(file)
                if "_cluster" in file:
                    metadata = {"needs_closest": False, "save": True}
                elif "KMeans" in file:
                    metadata = {"needs_closest": False, "save": False}
                else:
                    metadata = None

                if ".tif" in file:
                    self.dock_widget.viewer.open(file, layer_type="labels", visible=False, metadata=metadata)
                elif ".csv" in file:
                    self.dock_widget.viewer.open(file, layer_type="points", size=1, visible=False, metadata=metadata)

        # Add callbacks
        for layer in self.dock_widget.viewer.layers:
            if "Landscape-Vol" not in layer.name and len(layer.data.shape) == 2:
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

        if os.path.isdir(pathFile):
            shutil.rmtree(pathFile)

        os.mkdir(pathFile)

        with progress(self.dock_widget.viewer.layers) as pbr:
            for layer in pbr:
                save = True
                name = layer.name
                names = []
                selected_z = []
                if "Landscape" not in layer.name and "Priors" not in layer.name:
                    layer.save(os.path.join(pathFile, layer.name))
                    points = layer.data

                    if "save" in layer.metadata:
                        metadata = layer.metadata
                        if metadata["save"]:
                            # if metadata["needs_closest"]:
                            _, inds = self.kdtree_data.query(points, k=1)
                            inds = np.array(inds).flatten()
                            z_space_points = self.z_space[inds]
                            if not metadata["needs_closest"]:
                                mean = np.mean(z_space_points, axis=0)[None, ...]
                                selected_z.append(np.concatenate([mean, z_space_points], axis=0))
                                names.append(name + "_cluster")
                        else:
                            save = False
                    elif "cluster_ids_in_space" in layer.name:
                        labels = layer.data
                        unique_labels = np.unique(labels)[2:]
                        for label in unique_labels:
                            points = np.asarray(np.where(labels == label)).T
                            _, inds = self.kdtree_data.query(points, k=1)
                            inds = np.array(inds).flatten()
                            selected_z.append(self.z_space[inds])
                            names.append(name + f"_cluster_l{label}")
                    else:
                        _, inds = self.kdtree_data.query(points, k=1)
                        inds = np.array(inds).flatten()
                        selected_z.append(self.z_space[inds])
                        names.append(name)

                    if save:
                        for z, n in zip(selected_z, names):
                            np.savetxt(os.path.join(self.path, f'saved_selections_{n}.txt'), z, delimiter=" ")

    # ---------------------------------------------------------------------------
    # Callbacks
    # ---------------------------------------------------------------------------
    def on_close_callback(self):
        # If viewer is closed, remove socket
        for proc in psutil.process_iter():
            cmdline = " ".join(proc.cmdline())
            if "--port {:d}".format(self.port) in cmdline and "server.py" in cmdline:
                proc.kill()

    def lassoSelector(self, viewer, event):
        # viewer = self.view

        # Lasso is activate with mouse wheel button
        if event.button == 3:

            # Stop warnings for now (due to weird Napari behaviour)
            warnings.filterwarnings('ignore')

            # Initialize the lasso path list
            lasso_path = []

            # Layer to select from
            layer = viewer.layers.selection._current

            if isinstance(layer, napari.layers.Points):
                ndims = layer._view_data.shape[1]

                # Lasso layer
                lasso_layer = viewer.add_shapes(name='lasso', shape_type='path', edge_width=0.01, face_color=[0] * 4,
                                                edge_color="blue")

                # Add first clicked point to lasso path
                cursor = np.asarray(list(viewer.cursor.position))
                # cursor[0] = viewer.dims.current_step[0]
                lasso_path.append(cursor)
                yield

                # Keep on adding points on dragging
                while event.type == 'mouse_move' and not event.type == 'mouse_release':
                    cursor = (np.asarray(list(viewer.cursor.position)) +  # We might need to make scale smaller!
                              np.random.normal(loc=0, scale=0.001, size=3))  # Custom scale based on smallest distance?
                    # cursor[0] = viewer.dims.current_step[0]
                    lasso_path.append(cursor)
                    if len(lasso_path) > 1:
                        try:
                            lasso_layer.data = [np.array(lasso_path)]
                        except AttributeError:  # If we are selecting an empty selection just skip it
                            pass
                    yield

                # Once mouse is released, project points to 2D view if needed
                if ndims == 3:
                    points = layer._view_data
                    # points_proj = np.asarray([layer.world_to_data(x) for x in points])
                    # lasso_path_proj = np.asarray([layer.world_to_data(x) for x in lasso_path])
                    # projection_direction = np.asarray(layer.world_to_data(viewer.camera.view_direction))
                    points_proj = np.asarray(points)
                    lasso_path_proj = np.asarray(lasso_path)
                    projection_direction = np.asarray(viewer.camera.view_direction)
                    rot = rotation_matrix_from_vectors(projection_direction, np.asarray([0, 0, 1]))
                    points_proj = project_points(points_proj, projection_direction).dot(rot.T)[..., :2]
                    lasso_path_proj = project_points(lasso_path_proj, projection_direction).dot(rot.T)[..., :2]
                else:
                    points = layer.data
                    lasso_path_proj = np.asarray(lasso_path)
                    if points.shape[1] == 3:
                        ids = int(lasso_path_proj[0, viewer.dims.not_displayed])
                        ids = np.argwhere(np.logical_not(np.isclose(points[:, viewer.dims.not_displayed],
                                                                    ids, 1e-2)))[..., 0]
                        points_proj = points[..., viewer.dims.displayed]
                        points_proj[ids] *= 10000

                    else:
                        points_proj = points[..., viewer.dims.displayed]
                    lasso_path_proj = lasso_path_proj[..., viewer.dims.displayed]

                # Once mouse is released, create Lasso path and select points
                path = Path(lasso_path_proj)
                inside = path.contains_points(points_proj)
                selected_data = set(np.nonzero(inside)[0])

                if self.control_pressed:
                    layer.selected_data = layer.selected_data | selected_data
                elif self.alt_pressed:
                    layer.selected_data = layer.selected_data - selected_data
                else:
                    layer.selected_data = set(np.nonzero(inside)[0])
                lasso_path.clear()

                # Remove Lasso layer
                viewer.layers.remove(lasso_layer)

                # Keep selected layer active
                viewer.layers.selection.active = layer

            # Bring back warnings
            warnings.filterwarnings('default')

    def control_detection(self, event):
        # On key press
        self.control_pressed = True
        yield

        # On key release
        self.control_pressed = False

    def alt_detection(self, event):
        # On key press
        self.alt_pressed = True
        yield

        # On key release
        self.alt_pressed = False

    def _compute_kmeans_fired(self):
        landscape = self.dock_widget.viewer.layers["Landscape"].data
        self.kmeans_data = []
        self.clusters_data = []

        # Remove previous clusters and kmeans
        layer_names = [layer.name for layer in self.dock_widget.viewer.layers]
        for layer_name in layer_names:
            if "Cluster_" in layer_name or "KMeans" in layer_name:
                self.dock_widget.viewer.layers.remove(layer_name)

        # Compute KMeans and save automatic selection
        n_clusters = int(self.dock_widget.menu_widget.cluster_num.text())
        clusters = MiniBatchKMeans(n_clusters=n_clusters).fit(self.z_space)
        centers = clusters.cluster_centers_
        self.interp_val = clusters.labels_
        _, inds = self.kdtree_z_pace.query(centers, k=1)
        inds = np.array(inds).flatten()
        selected_data = np.copy(landscape[inds])
        self.kmeans_data.append(np.copy(self.data[inds]))
        self.dock_widget.viewer.add_points(selected_data, size=5, name="KMeans", metadata={"needs_closest": False,
                                                                                           "save": False})

        # Add points for each cluster independently with colors
        self.dock_widget.viewer.layers["Landscape"].visible = False
        cm = get_cmap("viridis")
        color_ids = np.linspace(0.0, 1.0, n_clusters)
        for label, color_id in zip(np.unique(self.interp_val), color_ids):
            self.kmeans_data.append(np.copy(self.data[self.interp_val == label]))
            cluster_points = np.copy(landscape[self.interp_val == label])
            color = np.asarray(cm(color_id))
            self.dock_widget.viewer.add_points(cluster_points, size=1, name=f"Cluster_{label}", visible=True,
                                               shading='spherical', edge_width=0, antialiasing=0,
                                               face_color=color, metadata={"needs_closest": False, "save": True})

    def _compute_dim_cluster_fired(self):
        landscape = self.dock_widget.viewer.layers["Landscape"].data
        axis = int(self.dock_widget.menu_widget.dimension_sel.currentText().replace("Dim ", "")) - 1
        self.kmeans_data = []
        self.clusters_data = []

        # Remove previous clusters and kmeans
        layer_names = [layer.name for layer in self.dock_widget.viewer.layers]
        for layer_name in layer_names:
            if "Cluster_" in layer_name or "KMeans" in layer_name:
                self.dock_widget.viewer.layers.remove(layer_name)

        # Determine the range of PCA DIM and divide into X equal intervals
        n_clusters = int(self.dock_widget.menu_widget.cluster_num.text())
        pca_axis = self.transformer_data[..., axis]
        min_pca1, max_pca1 = pca_axis.min(), pca_axis.max()
        intervals = np.linspace(min_pca1, max_pca1, n_clusters + 1)

        # Initialize lists to hold group means and point indices
        group_means = []
        # group_points = []
        labels = np.empty_like(pca_axis)

        # Compute clusters along dimension and save automatic selection
        for i in range(n_clusters):
            # Find points that fall within the current interval
            in_interval = (pca_axis >= intervals[i]) & (pca_axis < intervals[i + 1])
            if i == n_clusters - 1:
                # Ensure the last group includes the max value
                in_interval = (pca_axis >= intervals[i]) & (pca_axis <= intervals[i + 1])
            points_in_group = self.transformer_data[in_interval, axis]

            if len(points_in_group) > 0:
                # Assign labels
                labels[in_interval] = i

                # Compute mean for points in the interval along PCA1
                mean_pca = np.zeros(self.data.shape[-1])
                mean_pca[axis] = points_in_group.mean()

                # Store the mean and the points
                group_means.append(mean_pca)
                # group_points.append(points_in_group)

        group_means = np.vstack(group_means)
        self.interp_val = labels.astype(int)

        # Compute clusters along dimension and save automatic selection
        # sort_ind = np.argsort(landscape[..., axis])
        # sort_ind_split = np.array_split(sort_ind, n_clusters)
        # centers = np.vstack([np.mean(landscape[ind], axis=0) for ind in sort_ind_split])
        # labels = np.empty_like(sort_ind)
        # for idx in range(len(sort_ind_split)):
        #     labels[sort_ind_split[idx]] = idx
        # self.interp_val = labels

        # Cluster always along PCA space
        # _, inds = self.kdtree_data.query(group_means, k=1)
        # inds = np.array(inds).flatten()
        # selected_data = np.copy(landscape[inds])
        z_tr_data = self.transformer.inverse_transform(group_means)
        self.kmeans_data.append(np.copy(z_tr_data))

        group_means = 127 * (group_means - np.amin(self.transformer_data)) / (np.amax(self.transformer_data) - np.amin(self.transformer_data))
        self.dock_widget.viewer.add_points(group_means[..., self.current_axis], size=5, name="KMeans along PCA {:d}".format(axis + 1),
                                           metadata={"needs_closest": False, "save": False})

        # Add points for each cluster independently with colors
        self.dock_widget.viewer.layers["Landscape"].visible = False
        cm = get_cmap("viridis")
        color_ids = np.linspace(0.0, 1.0, n_clusters)
        for label, color_id in zip(np.unique(self.interp_val), color_ids):
            self.kmeans_data.append(np.copy(self.data[self.interp_val == label]))
            cluster_points = np.copy(landscape[self.interp_val == label])
            color = np.asarray(cm(color_id))
            self.dock_widget.viewer.add_points(cluster_points, size=1, name=f"Cluster_{label}", visible=True,
                                               shading='spherical', edge_width=0, antialiasing=0,
                                               face_color=color, metadata={"needs_closest": False, "save": True})


    def _morph_chimerax_fired(self):
        # Morph maps in chimerax based on different ordering methods
        if self.thread_chimerax is None:
            layer = self.dock_widget.viewer.layers.selection._current
            if "Landscape" not in layer.name and "Cluster_" not in layer.name:
                points = layer.data
                if points.shape[0] > 0:
                    _, inds = self.kdtree_data.query(points, k=1)
                    inds = np.array(inds).flatten()
                    sel_names = ["vol_%03d" % (idx + 1) for idx in range(points.shape[0])]
                    z_space = self.z_space[inds]
                    if "along PCA" in layer.name:
                        z_space = self.transformer.inverse_transform(self.transformer.transform(z_space))
                    if z_space.ndim == 1:
                        z_space = z_space[None, ...]

                    if self.client.isRunning():
                        show_warning(
                            "Previous conformation has not being generated yet, current selection will not be generated")
                    else:
                        self.client.z = z_space
                        self.client.file_names = sel_names
                        self.client.start()

    def on_insert_add_callback(self, event):
        layer = event.value
        if layer.name != "lasso":
            if len(layer.data.shape) == 2:
                layer.events.data.connect(self.updateConformation)
                layer.events.highlight.connect(self.updateConformation)

    def selectAxis(self, pos, event):
        axis = int(event.replace("Dim ", "")) - 1

        # Change point layers
        for layer in self.dock_widget.viewer.layers:
            if len(layer.data.shape) == 2:
                current_data = layer.data

                if current_data.shape[0] != self.data.shape[0]:
                    _, inds = self.kdtree_data.query(current_data, k=1)
                    inds = np.array(inds).flatten()
                    layer.data[:, pos] = self.data[inds, axis]
                else:
                    layer.data[:, pos] = self.data[:, axis]

                layer.refresh()

        # Landscape data
        data = self.dock_widget.viewer.layers["Landscape"].data

        # Change volume
        layer = self.dock_widget.viewer.layers["Landscape-Vol"]
        boxsize = layer.data.shape[0]
        indeces = np.round(data).astype(int)
        vol = np.zeros((boxsize, boxsize, boxsize))
        vol[indeces[:, 0], indeces[:, 1], indeces[:, 2]] += 1
        # Filter volume
        std = np.pi * np.sqrt(boxsize) / self.dock_widget.right_widgets[3].get_value()
        gauss_1d = signal.windows.gaussian(boxsize, std)
        kernel = np.einsum('i,j,k->ijk', gauss_1d, gauss_1d, gauss_1d)
        kernel = np.pad(kernel, (5, 5))
        vol = np.pad(vol, (5, 5))
        ft_vol = np.fft.fftshift(np.fft.fftn(vol))
        ft_vol_real = np.real(ft_vol) * kernel
        ft_vol_imag = np.imag(ft_vol) * kernel
        ft_vol = ft_vol_real + 1j * ft_vol_imag
        vol = np.real(np.fft.ifftn(np.fft.ifftshift(ft_vol)))[5:133, 5:133, 5:133]
        layer.data = vol
        layer.refresh()

        # Change labels
        layer = self.dock_widget.viewer.layers["Landscape-Vol-Labels"]
        clusters = MiniBatchKMeans(n_clusters=100).fit(indeces)
        values = clusters.labels_
        labels = np.zeros((boxsize, boxsize, boxsize))
        labels[indeces[:, 0], indeces[:, 1], indeces[:, 2]] += values
        labels = labels.astype(int)
        layer.data = labels
        layer.refresh()

        # Update current axis
        self.current_axis[pos] = axis
        self.kdtree_data = KDTree(self.data[:, self.current_axis])

    def updateVolSigma(self, sigma):
        # Landscape data
        data = self.dock_widget.viewer.layers["Landscape"].data

        # Change volume
        layer = self.dock_widget.viewer.layers["Landscape-Vol"]
        boxsize = layer.data.shape[0]
        indeces = np.round(data).astype(int)
        vol = np.zeros((boxsize, boxsize, boxsize))
        vol[indeces[:, 0], indeces[:, 1], indeces[:, 2]] += 1
        # Filter volume
        std = np.pi * np.sqrt(boxsize) / sigma
        gauss_1d = signal.windows.gaussian(boxsize, std)
        kernel = np.einsum('i,j,k->ijk', gauss_1d, gauss_1d, gauss_1d)
        kernel = np.pad(kernel, (5, 5))
        vol = np.pad(vol, (5, 5))
        ft_vol = np.fft.fftshift(np.fft.fftn(vol))
        ft_vol_real = np.real(ft_vol) * kernel
        ft_vol_imag = np.imag(ft_vol) * kernel
        ft_vol = ft_vol_real + 1j * ft_vol_imag
        vol = np.real(np.fft.ifftn(np.fft.ifftshift(ft_vol)))[5:133, 5:133, 5:133]
        layer.data = vol
        layer.refresh()

    def updateVolLabels(self, num_labels):
        # Landscape data
        data = self.dock_widget.viewer.layers["Landscape"].data

        # Change labels
        layer = self.dock_widget.viewer.layers["Landscape-Vol-Labels"]
        boxsize = layer.data.shape[0]
        indeces = np.round(data).astype(int)
        clusters = MiniBatchKMeans(n_clusters=num_labels).fit(indeces)
        values = clusters.labels_ + 1
        labels = np.zeros((boxsize, boxsize, boxsize))
        labels[indeces[:, 0], indeces[:, 1], indeces[:, 2]] += values
        labels = labels.astype(int)
        layer.data = labels
        layer.refresh()

    def extractSelectionToLayer(self):
        choices = list(self.dock_widget.right_widgets[5].choices)
        total = len(choices)
        layer = self.dock_widget.viewer.layers.selection._current
        layer_name = f"Selected points {total + 1}"
        selected = layer.data[np.asarray(list(layer.selected_data)).astype(int)]
        self.dock_widget.viewer.add_points(selected, size=1, name=layer_name, visible=True,
                                           shading='spherical', edge_width=0, antialiasing=0, blending="additive",
                                           face_color="white", metadata={"needs_closest": False, "save": True,
                                                                         "user_selection": True})
        # choices.append(layer_name)
        # self.dock_widget.right_widgets[5].choices = choices

    def addSelectionToLayer(self):
        layer_name = self.dock_widget.right_widgets[5].current_choice
        layer = self.dock_widget.viewer.layers[layer_name]
        sel_layer = self.dock_widget.viewer.layers.selection._current
        selected = sel_layer.data[np.asarray(list(sel_layer.selected_data)).astype(int)]
        layer.data = np.concatenate([layer.data, selected], axis=0)
        layer.refresh()

    def getLayerChoices(self, extra):
        choices = []
        for layer in self.dock_widget.viewer.layers:
            if "user_selection" in layer.metadata:
                if layer.metadata["user_selection"]:
                    choices.append(layer.name)
        return choices

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
                    pos = layer.data[list(selected)]
                    if pos.ndim == 2 and pos.shape[0] > 1:
                        pos = np.mean(pos, axis=0)
                        pos = pos[None, ...]
                else:
                    return
            else:
                return

        _, ind = self.kdtree_data.query(pos, k=1)
        ind = np.array(ind).flatten()[0]

        if self.client.isRunning():
            show_warning("Previous conformation has not being generated yet, current selection will not be generated")
        else:
            self.client.z = self.z_space[ind, :][None, ...]
            self.client.file_names = [""]
            self.client.start()

    # ---------------------------------------------------------------------------
    # Worker generation
    # ---------------------------------------------------------------------------
    def createThreadChimeraX(self, *args):
        self.thread_chimerax = QThread()
        self.worker = FlexMorphChimeraX(*args)
        self.worker.moveToThread(self.thread_chimerax)
        self.thread_chimerax.started.connect(self.worker.showSalesMan)
        self.worker.finished.connect(self.thread_chimerax.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread_chimerax.finished.connect(self.thread_chimerax.deleteLater)
        self.thread_chimerax.finished.connect(self.removeThreadChimeraX)

    def removeThreadChimeraX(self):
        self.thread_chimerax = None

    def updateEmittedMap(self, map):
        layer = self.dock_widget.viewer_model1.layers[0]
        data_is_empty = np.sum(layer.data[:]) == 0.0
        layer.data = map
        self.prev_layers = [layer.data.copy() for layer in self.dock_widget.viewer.layers]

        if data_is_empty:
            self.dock_widget.viewer_model1.reset_view()
            layer.iso_threshold = 3.0 * np.std(map[map >= 0.0])

        layer.contrast_limits = [0.0, np.amax(map)]

    def launchChimeraX(self):
        sel_names = self.client.file_names
        z = self.client.z
        self.createThreadChimeraX(z, sel_names, self.path)
        self.thread_chimerax.start()

# ---------------------------------------------------------------------------
# Utils functions
# ---------------------------------------------------------------------------
def project_points(points, normal):
    """
    Projects the points with coordinates x, y, z onto the plane
    defined by a*x + b*y + c*z = 1
    """
    a, b, c = normal[0], normal[1], normal[2]
    vector_norm = a * a + b * b + c * c
    normal_vector = np.array([a, b, c]) / np.sqrt(vector_norm)
    point_in_plane = np.array([a, b, c]) / vector_norm

    points_from_point_in_plane = points - point_in_plane
    proj_onto_normal_vector = np.dot(points_from_point_in_plane,
                                     normal_vector)
    proj_onto_plane = (points_from_point_in_plane -
                       proj_onto_normal_vector[:, None] * normal_vector)

    return point_in_plane + proj_onto_plane

def rotation_matrix_from_vectors(vec1, vec2):
    """ Find the rotation matrix that aligns vec1 to vec2
    :param vec1: A 3d "source" vector
    :param vec2: A 3d "destination" vector
    :return mat: A transform matrix (3x3) which when applied to vec1, aligns it with vec2.
    """
    a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (vec2 / np.linalg.norm(vec2)).reshape(3)
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
    return rotation_matrix

if __name__ == '__main__':
    import argparse

    # Input parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, required=True)
    parser.add_argument('--z_space', type=str, required=True)
    parser.add_argument('--interp_val', type=str, required=True)
    parser.add_argument('--path', type=str, required=False)
    parser.add_argument('--mode', type=str, required=False)
    parser.add_argument('--onlyView', action='store_true')

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
    input_dict['interactive'] = not args.onlyView
    # input_dict['interp_val'] = interp_val

    # Initialize volume slicer
    Annotate3D(**input_dict)
