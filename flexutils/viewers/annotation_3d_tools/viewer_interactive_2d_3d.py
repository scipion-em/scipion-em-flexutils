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
import sys
from functools import partial
from xmipp_metadata.image_handler import ImageHandler

import numpy as np
from multiprocessing import Process

from watchdog.observers import Observer
from watchdog.events import LoggingEventHandler

from sklearn.neighbors import KDTree
from sklearn.cluster import KMeans

from bokeh.models import HoverTool, LinearColorMapper, PointDrawTool, ColumnDataSource
from bokeh.plotting import figure
from bokeh.models import Select, Button, TextInput
from bokeh.layouts import column, layout
from bokeh.server.server import Server
from bokeh.util.hex import hexbin
from bokeh.transform import linear_cmap
from bokeh.palettes import all_palettes

from PyQt5 import QtCore, QtWidgets, QtWebEngineWidgets, QtGui
from PyQt5.QtCore import QUrl, QTimer
from PyQt5.QtWidgets import QTabWidget

from traits.api import HasTraits, Instance, Array, Float, String, on_trait_change
from traitsui.api import View, Item, HGroup, Group, HSplit, RangeEditor

from mayavi import mlab
from mayavi.core.api import PipelineBase
from mayavi.core.ui.api import SceneEditor, MayaviScene, MlabSceneModel

from pyworkflow.utils import getExt

import flexutils


class InteractiveAnnotate2D(QtWidgets.QWidget):

    # ---------------------------------------------------------------------------
    # Init
    # ---------------------------------------------------------------------------
    def __init__(self, **kwargs):
        super(InteractiveAnnotate2D, self).__init__()
        # Attributes from kwargs
        self.class_inputs = kwargs
        self.data = kwargs.get("data")
        self.z_space = kwargs.get("z_space")
        self.path = kwargs.get("path")
        self.mode = kwargs.get("mode")
        self.data_tree_3d = KDTree(self.data)
        self.data_tree_2d = KDTree(self.data[:, :-1])
        self.z_space_tree = KDTree(self.z_space)
        self.adding_kmeans = False
        self.renaming_selection = False
        self.updating_2d = False
        # self.hexbin_bins = ColumnDataSource(data=dict(r=[], q=[], counts=[]))

        # Setup main window
        self.fig = figure(title="HexBin flexible landscape",
                          match_aspect=True, tools="wheel_zoom,reset,pan",
                          background_fill_color='black',
                          toolbar_location="below",
                          plot_width=650, plot_height=600)
        self.fig.grid.visible = False

        # Setup tabs Widget
        self.mayavi_tabs = QTabWidget()

        # Setup Mayavi window
        self.createVWInterface(view="volume_viewer", **kwargs)
        self.createVWInterface(view="landscape", **kwargs)

        # Setup server
        self.server = Server({'/': self.createInterface}, num_procs=1)
        self.server.start()
        # server.io_loop.start()

        # Web engine to render plot
        self.window = QtWidgets.QWidget()
        self.web_engine = QtWebEngineWidgets.QWebEngineView()
        self.web_engine.setFixedWidth(1000)
        self.web_engine.load(QUrl("http://localhost:%s/" % self.server.port))

        # PyQt layout
        window_lay = QtWidgets.QHBoxLayout()
        lay_1 = QtWidgets.QVBoxLayout()
        lay_1.addWidget(self.web_engine)
        lay_2 = QtWidgets.QVBoxLayout()
        lay_2.addWidget(self.mayavi_tabs)
        self.mayavi_tabs.addTab(self.mayavi_landscape_view, "Landscape View")
        self.mayavi_tabs.addTab(self.mayavi_vw_view, "Volume Viewer")
        window_lay.addLayout(lay_1)
        window_lay.addLayout(lay_2)
        # self.window.setFixedSize(1600, 650)
        self.window.setLayout(window_lay)
        self.setWindowIcon(QtGui.QIcon(os.path.join(os.path.dirname(flexutils.__file__), "icon_square.png")))

        # Start Mayavi window
        # self.mayavi_view.visualization.configure_traits()

        # Watcher for Mayavi
        class MayaviHandler(LoggingEventHandler):
            def on_moved(cls, event):
                if "z_space_mayavi.txt" in event.dest_path:
                    self.update_mayavi_map()

        class ChimeraxHandler(LoggingEventHandler):
            def on_moved(cls, event):
                if "z_space_chimerax.txt" in event.dest_path:
                    self.morph_selections_chimerax()

        mayavi_watcher = Observer()
        open(os.path.join(self.path, "z_space_mayavi.txt"), 'w').close()
        mayavi_watcher.schedule(MayaviHandler(), self.path, recursive=True)
        mayavi_watcher.start()

        chimerax_watcher = Observer()
        open(os.path.join(self.path, "z_space_chimerax.txt"), 'w').close()
        chimerax_watcher.schedule(ChimeraxHandler(), self.path, recursive=True)
        chimerax_watcher.start()

        # Refresh main window every half a second to avoid Mayavi taking grab
        self.__timer = QTimer()
        self.__timer.timeout.connect(self.web_engine.update)
        self.__timer.start(500)

    # ---------------------------------------------------------------------------
    # Create interface
    # ---------------------------------------------------------------------------
    def createInterface(self, doc):
        self.bokeh_doc = doc

        # Colormap control
        colormaps = {"Viridis": "Viridis", "Cividis": "Cividis", "Plasma": "Plasma",
                     "Inferno": "Inferno", "BuPu": "BuPu", "Oranges": "Oranges",
                     "PuRd": "PuRd"}
        self.colormap_select = Select(value="Viridis", title='Colormap', options=sorted(colormaps.keys()))
        self.colormap_select.on_change('value', self.update_color_map)

        # Save selections button
        button_save_selections = Button(label="Save current selections", button_type="success")
        button_save_selections.on_click(self.save_selections)

        # Rename selections
        self.label_input = TextInput(value="", title="New label:", width=150)
        button_rename_selection = Button(label="Rename selection", button_type="success", width=50, align="end")
        button_rename_selection.on_click(self.rename_selection)

        # Cluster space
        self.n_clusters = TextInput(value="10", title="Number of clusters:", width=150)
        button_cluster = Button(label="Cluster KMeans", button_type="success", width=50, align="end")
        button_cluster.on_click(self.cluster_space_kmeans)

        # ChimeraX morphing
        path_type = {"Salesman": "Salesman", "Random walk": "Random walk"}
        self.path_select = Select(value="Salesman", title='Path type', options=sorted(path_type.keys()),
                                  width=150)
        button_do_morph = Button(label="Morph Chimerax", button_type="success",
                                 width=50, align="end")
        button_do_morph.on_click(self.save_selections_chimerax)

        # Initiate tools
        self.initiatePointDrawTool()

        # Hexbin plot
        bins = hexbin(self.data[:, 0], self.data[:, 1], 0.1)
        self.hexbin_bins = ColumnDataSource(data=dict(r=[], q=[], counts=[]))
        self.hexbin_bins.data = dict(r=bins.r, q=bins.q, counts=bins.counts)

        # Watcher for Mayavi
        class MayaviHandler(LoggingEventHandler):
            def on_moved(cls, event):
                if os.path.join(self.path, "projected_coords.txt") in event.dest_path:
                    self.update_2d_landscape()

        mayavi_landscape_watcher = Observer()
        open(os.path.join(self.path, "projected_coords.txt"), 'w').close()
        mayavi_landscape_watcher.schedule(MayaviHandler(), self.path, recursive=True)
        mayavi_landscape_watcher.start()

        self.hexbin = self.fig.hex_tile(q='q', r='r', source=self.hexbin_bins, size=0.1, level="underlay", line_color=None,
                                        fill_color=linear_cmap('counts', 'Viridis256', 0, max(self.hexbin_bins.data["counts"])))

        # Layout
        controls_colormap = column(self.colormap_select)
        controls_save_selections = column(button_save_selections)

        # Server
        doc.add_root(layout([[self.fig, [controls_colormap,
                                         [self.label_input, button_rename_selection],
                                         [self.n_clusters, button_cluster],
                                         [self.path_select, button_do_morph],
                                         controls_save_selections
                                         ]
                              ]
                             ]
                            ))

    def createVWInterface(self, view, **kwargs):
        if self.mode == "Zernike3D":
            inputs_mayavi = {"path": self.path,
                             "mode": self.mode,
                             "L1": int(kwargs.get("L1")),
                             "L2": int(kwargs.get("L2"))}
        elif self.mode == "CryoDrgn":
            inputs_mayavi = {"path": self.path,
                             "mode": self.mode,
                             "weights": kwargs.get("weights"),
                             "config": kwargs.get("config"),
                             "boxsize": int(kwargs.get("boxsize")),
                             "sr": float(self.class_inputs["sr"])}
        if view == "volume_viewer":
            self.mayavi_vw_view = MayaviQWidget(view=view, **inputs_mayavi)
        elif view == "landscape":
            self.mayavi_landscape_view = MayaviQWidget(view=view, data=kwargs.get("data"), path=kwargs.get("path"))

    # ---------------------------------------------------------------------------
    # Tools
    # ---------------------------------------------------------------------------
    def initiatePointDrawTool(self):
        # Read selections or make default
        # self.seeds = ColumnDataSource({'x': [], 'y': [], 'names': []})
        self.loadSelections()

        # Selection callbacks
        self.seeds.on_change("data", self.set_default_values_selection, self.save_z_mayavi, self.update_selection_indeces)
        self.seeds.selected.on_change('indices', self.save_z_mayavi)

        # Tooltips setup
        TOOLTIPS = [
            ("index", "$index"),
            ("(x,y)", "($x, $y)"),
            ("label", "@names"),
        ]

        # Renderers for selections and tooltips
        self.renderer = self.fig.scatter(x='x', y='y', source=self.seeds, color='white', size=10,
                                        line_color='black', line_width=3)
        draw_tool = PointDrawTool(renderers=[self.renderer])
        self.fig.add_tools(draw_tool)
        self.fig.add_tools(HoverTool(tooltips=TOOLTIPS, renderers=[self.renderer]))

    # ---------------------------------------------------------------------------
    # Callbacks
    # ---------------------------------------------------------------------------
    def update_color_map(self, attrname, old, new):
        palette = all_palettes[new]
        palette = palette[list(palette)[-1]]
        color_mapper = LinearColorMapper(palette=palette, low=min(self.hexbin_bins.data["counts"]),
                                         high=max(self.hexbin_bins.data["counts"]))
        # color_mapper = LinearColorMapper(palette=new, low=np.amin(), high=np.amax())
        self.hexbin.glyph.fill_color['transform'].palette = color_mapper.palette

    def save_selections(self):
        selections = self.seeds.data
        selected_points = np.asarray([selections["x"], selections["y"], selections["z"], selections["idx"]]).T
        selected_z_space = self.z_space[np.asarray(selections["idx"]).reshape(-1), :]

        np.savetxt(os.path.join(self.path, "saved_selections.txt"), selected_z_space)
        np.savetxt(os.path.join(self.path, "selections.txt"), selected_points)
        with open(os.path.join(self.path, 'selection_names.txt'), 'w') as f:
            for line in selections["names"]:
                f.write(line + "\n")

    def rename_selection(self):
        self.renaming_selection = True
        selected = self.seeds.selected.indices
        if len(selected) == 1:
            names = self.seeds.data['names']
            names[selected[0]] = self.label_input.value
            self.seeds.data['names'] = names
        self.renaming_selection = False

    def set_default_values_selection(self, attrname, old, new):
        if not self.adding_kmeans and not self.updating_2d:
            if len(new["names"]) > len(old["names"]):
                names = old["names"]
                names.append("class_%d" % (len(new["names"]) - 1))

                z = old["z"]
                z.append(0.0)

                idx = old["idx"]
                new_point = np.asarray([new["x"][-1], new["y"][-1]]).reshape(1, -1)
                _, ind = self.data_tree_2d.query(new_point, k=1)
                idx.append(ind[0].astype(int).item())

                data = {"x": new["x"], "y": new["y"], "z": z, "names": names,
                        "idx": idx}
                self.seeds.data = data


    def save_z_mayavi(self, attrname, old, new):
        if not self.adding_kmeans and not self.renaming_selection and not self.updating_2d:
            idx = None
            if isinstance(new, dict):
                idx = list(set(new["idx"]) - set(old["idx"]))
                if len(idx) == 0:
                    idx = new["idx"][-1]
                else:
                    idx = idx[-1]
            elif isinstance(new, list) and len(new) == 1:
                idx = self.seeds.data["idx"][new[0]]

                # x, y = self.seeds.data["x"][new[0]], self.seeds.data["y"][new[0]]
                # new_point = np.asarray([x, y]).reshape(1, -1)
                # _, ind = self.data_tree_2d.query(new_point, k=1)
                # idx = ind[0].astype(int).item()

            if idx:
                z_s = self.z_space[int(idx), :]
                np.savetxt(os.path.join(self.path, "temp.txt"), z_s, delimiter=",")
                os.rename(os.path.join(self.path, "temp.txt"),
                          os.path.join(self.path, "z_space_mayavi.txt"))

    def update_selection_indeces(self, attrname, old, new):
        if isinstance(new, dict) and not None in new["x"] and not None in old["x"]:
            x = list(set(new["x"]) - set(old["x"]))
            if len(x) > 0:
                idn = new["x"].index(x[0])
                x, y = new["x"][idn], new["y"][idn]
                new_point = np.asarray([x, y]).reshape(1, -1)
                _, ind = self.data_tree_2d.query(new_point, k=1)
                idx = ind[0].astype(int).item()

                seeds_data = dict(self.seeds.data)
                new_idx = seeds_data["idx"]
                new_idx[idn] = idx
                seeds_data["idx"] = new_idx

                async def update(data):
                    self.seeds.data = data

                self.bokeh_doc.add_next_tick_callback(partial(update, seeds_data))

    def save_selections_chimerax(self):
        selected_data = self.seeds.data
        selected_z_space = self.z_space[np.asarray(selected_data["idx"]).reshape(-1), :]
        sel_names = np.asarray(selected_data["names"])

        with open(os.path.join(self.path, 'selection_names_chimerax.txt'), 'w') as f:
            for line in sel_names:
                f.write(line + "\n")
            f.write(self.path_select.value + "\n")
        np.savetxt(os.path.join(self.path, "temp.txt"), selected_z_space)
        os.rename(os.path.join(self.path, "temp.txt"),
                  os.path.join(self.path, "z_space_chimerax.txt"))

    def update_mayavi_map(self):
        self.mayavi_vw_view.visualization.updateMap()
        # self.web_engine.update()
        # self.mayavi_vw_view.update()

    def cluster_space_kmeans(self):
        n_clusters = int(self.n_clusters.value)
        # First delete all automatic selections associated to a previous KMeans
        new_x, new_y, new_z, new_idx, new_names = [], [], [], [], []
        old_x, old_y, old_z, old_idx, old_names = self.seeds.data["x"], self.seeds.data["y"], \
                                                  self.seeds.data["z"], self.seeds.data["idx"], \
                                                  self.seeds.data["names"]
        for x, y, z, idx, name in zip(old_x, old_y, old_z, old_idx, old_names):
            if not 'kmean' in name:
                new_x.append(x)
                new_y.append(y)
                new_z.append(z)
                new_idx.append(idx)
                new_names.append(name)
        # Compute KMeans and save automatic selection
        if int(n_clusters) > 0:
            clusters = KMeans(n_clusters=int(n_clusters), n_init=1).fit(self.z_space)
            centers = clusters.cluster_centers_
            _, inds = self.z_space_tree.query(centers, k=1)
            for idx, ind in enumerate(inds):
                new_x.append(self.data[ind[0], 0])
                new_y.append(self.data[ind[0], 1])
                new_z.append(self.data[ind[0], 2])
                new_idx.append(ind[0].astype(int).item())
                new_names.append("kmean_%d" % (idx + 1))
        self.adding_kmeans = True
        new_data = {"x": new_x, "y": new_y, "z": new_z, "idx": new_idx, "names": new_names}
        self.seeds.data = new_data
        self.adding_kmeans = False

    def update_2d_landscape(self):
        if os.stat(os.path.join(self.path, "projected_coords.txt")).st_size != 0:
            data = np.loadtxt(os.path.join(self.path, "projected_coords.txt"))
            bins = hexbin(data[:, 0], data[:, 1], 0.5)
            self.data = data
            self.data_tree_3d = KDTree(data)
            self.data_tree_2d = KDTree(data[:, :-1])

            # selected_data = self.seeds.data
            # selected_points = np.asarray([selected_data["x"], selected_data["y"], selected_data["z"]]).T
            # names = selected_data["names"]
            # W = np.ones((selected_points.shape[0], 1))
            # hmgns_world_coords = np.column_stack((selected_points, W))
            # comb_trans_mat = np.loadtxt("comb_trans_mat.txt")
            # view_coords = np.dot(comb_trans_mat, hmgns_world_coords.T).T
            # norm_view_coords = view_coords / (view_coords[:, 3].reshape(-1, 1))
            # view_to_disp_mat = np.loadtxt("view_to_disp_mat.txt")
            # disp_coords = np.dot(view_to_disp_mat, norm_view_coords.T).T

            selected_data = self.seeds.data
            idx = selected_data["idx"]
            idx_np = np.asarray(idx).astype(int)
            new_data = {"x": data[idx_np, 0].tolist(),
                        "y": data[idx_np, 1].tolist(),
                        "z": data[idx_np, 2].tolist(),
                        "idx": idx,
                        "names": selected_data["names"]}

            async def update(data):
                self.updating_2d = True
                self.seeds.data = data
                self.updating_2d = False

            self.bokeh_doc.add_next_tick_callback(partial(update, new_data))


            async def update(r, q, counts):
                self.hexbin_bins.data = dict(r=r, q=q, counts=counts)
                self.fig.renderers.remove(self.hexbin)
                palette = all_palettes[self.colormap_select.value]
                palette = palette[list(palette)[-1]]
                self.hexbin = self.fig.hex_tile(q='q', r='r', source=self.hexbin_bins, size=0.5, level="underlay",
                                                line_color=None,
                                                fill_color=linear_cmap('counts', palette, 0,
                                                                       max(self.hexbin_bins.data["counts"])))

            self.bokeh_doc.add_next_tick_callback(partial(update, r=bins.r, q=bins.q, counts=bins.counts))
            # color_mapper = LinearColorMapper(palette="Viridis256", low=min(self.hexbin_bins.data["counts"]),
            #                                  high=max(self.hexbin_bins.data["counts"]))
            # self.hexbin.glyph.fill_color['transform'].palette = color_mapper.palette

    def morph_selections_chimerax(self):
        # Morph maps in chimerax based on different ordering methods
        from flexutils.viewers.chimera_viewers.viewer_morph_chimerax import FlexMorphChimeraX
        z_space_selected = np.loadtxt(os.path.join(self.path, "z_space_chimerax.txt"))
        with open(os.path.join(self.path, 'selection_names_chimerax.txt'), 'r') as f:
            sel_names = [line.strip("\n") for line in f.readlines()]
        path_select = sel_names[-1]
        sel_names = sel_names[:-1]
        if len(sel_names) > 1:
            if path_select == "Salesman":
                morph_chimerax = FlexMorphChimeraX(z_space_selected, sel_names, self.mode, self.path,
                                                   **self.class_inputs)
                morph_chimerax.showSalesMan()
            elif path_select == "Random walk":
                morph_chimerax = FlexMorphChimeraX(z_space_selected, sel_names, self.mode, self.path,
                                                   **self.class_inputs)
                morph_chimerax.showRandomWalk()

    # ---------------------------------------------------------------------------
    # Read data
    # ---------------------------------------------------------------------------
    def loadSelections(self):
        if os.path.isfile(os.path.join(self.path, "selections.txt")):
            selections = np.loadtxt(os.path.join(self.path, "selections.txt"))
            with open(os.path.join(self.path, 'selection_names.txt'), 'r') as f:
                names = [line.strip("\n") for line in f.readlines()]
            self.seeds = ColumnDataSource({'x': selections[:, 0].tolist(),
                                           'y': selections[:, 1].tolist(),
                                           'z': selections[:, 2].tolist(),
                                           'names': names,
                                           'idx': selections[:, 3].tolist()})
        else:
            x, y, z, names, idx = [], [], [], [], []
            if "n_vol" in self.class_inputs:
                n_vol = int(self.class_inputs["n_vol"])
                x, y, z = self.data[-n_vol:, 0].tolist(), self.data[-n_vol:, 1].tolist(), self.data[-n_vol:, 2].tolist()
                idx = list(range((self.data.shape[0] - n_vol), self.data.shape[0]))
                names.append("reference")
                for idx in range(1, n_vol):
                    names.append("class_%d" % idx)
            data = {'x': x, 'y': y, 'z': z, 'names': names, 'idx': idx}
            self.seeds = ColumnDataSource(data)


    # ---------------------------------------------------------------------------
    # Launch values
    # ---------------------------------------------------------------------------
    def launchViewer(self):
        # Get application
        app = QtWidgets.QApplication.instance()
        if app is None:
            app = QtWidgets.QApplication(sys.argv)

        # Show interface
        self.window.show()

        # Load Bokeh server
        p = Process(target=self.server.io_loop.start, args=())
        p.start()

        # Execute app
        app.exec_()

        # Stop app and sever
        self.server.io_loop.stop()
        self.server.stop()
        p.terminate()


# -------- Mayavi 3D map ----------
class MapView(HasTraits):

    # Map (actor)
    ipw_map = Instance(PipelineBase)

    # Main view
    scene3d = Instance(MlabSceneModel, ())

    # Visualization style
    threshold = Float(0.0)
    op_min = Float(0.0)
    op_max = Float(1.0)

    # Map
    map = Array()
    generated_map = Array()

    # Output path
    path = String()

    # ---------------------------------------------------------------------------
    def __init__(self, **traits):
        self.class_inputs = traits
        super(MapView, self).__init__(**traits)
        self.map
        self.ipw_map

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
        elif self.mode == "HetSIREN":
            from flexutils.utils import generateVolumesHetSIREN
            self.generate_map = generateVolumesHetSIREN
        elif self.mode == "NMA":
            from flexutils.utils import generateVolumesDeepNMA
            self.generate_map = generateVolumesDeepNMA

    # ---------------------------------------------------------------------------
    # Default values
    # ---------------------------------------------------------------------------
    def _map_default(self):
        return np.zeros([64, 64, 64])

    # ---------------------------------------------------------------------------
    # Scene activation callbaks
    # ---------------------------------------------------------------------------
    @on_trait_change('scene3d.activated')
    def display_scene3d(self):
        self.scene3d.mlab.view(40, 50)
        self.scene3d.scene.background = (0, 0, 0)

    @on_trait_change('scene3d.activated')
    def display_map(self):
        volume = mlab.contour3d(self.map, color=(1, 1, 1), contours=1, figure=self.scene3d.mayavi_scene)
        volume.contour.auto_contours = False
        volume.contour.auto_update_range = True
        setattr(self, 'ipw_map', volume)

    @on_trait_change("threshold")
    def change_map_contour(self):
        volume = getattr(self, 'ipw_map')
        val_max = np.amax(volume.mlab_source.m_data.scalar_data)
        val_min = np.amin(volume.mlab_source.m_data.scalar_data)
        if val_max != val_min:
            contour = (val_max - val_min) * self.threshold + val_min
            volume.contour.contours = [contour]

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

    # ---------------------------------------------------------------------------
    # View functions
    # ---------------------------------------------------------------------------
    def updateMap(self):
        z = np.loadtxt(os.path.join(self.path, "z_space_mayavi.txt"), delimiter=",")
        if self.generate_map is not None:
            if self.mode == "Zernike3D":
                self.generate_map("reference.mrc", "mask.mrc", "deformed.mrc",
                                  self.path, z,
                                  int(self.class_inputs["L1"]), int(self.class_inputs["L2"]), 32)
                self.generated_map = self.readMap(os.path.join(self.path, "deformed.mrc"))
            elif self.mode == "CryoDrgn":
                self.generate_map(z, self.class_inputs["weights"],
                                  self.class_inputs["config"], self.path, downsample=int(self.class_inputs["boxsize"]),
                                  apix=float(self.class_inputs["sr"]))
                self.generated_map = self.readMap(os.path.join(self.path, "vol_000.mrc"))
            elif self.mode == "HetSIREN":
                self.generate_map(self.class_inputs["weights"], z,
                                  self.path, step=self.class_inputs["step"])
                self.generated_map = self.readMap(os.path.join(self.path, "decoded_map_class_01.mrc"))
            elif self.mode == "NMA":
                self.generate_map(self.class_inputs["weights"], z,
                                  self.path, sr=float(self.class_inputs["sr"]))
                self.generated_map = self.readMap(os.path.join(self.path, "decoded_map_class_01.mrc"))

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
                Item('threshold',
                     editor=RangeEditor(format='%.02f', low_name='op_min', high_name='op_max', mode='slider'),
                     ),
                show_labels=False
            )
        )
    ),
    resizable=True,
    title='Point cloud clustering',
    icon=os.path.join(os.path.dirname(flexutils.__file__), "icon_square.png")
    )


# -------- Mayavi 3D landscape ----------
class LandscapeView(HasTraits):

    # Map (actor)
    ipw_pc = Instance(PipelineBase)

    # Main view
    scene3d = Instance(MlabSceneModel, ())

    # Visualization style
    opacity = Float(0.0)
    scale_points = Float(0.5)
    op_min = Float(0.0)
    op_max = Float(1.0)
    sc_min = Float(0.0)
    sc_max = Float(10.0)

    # Output path
    path = String()

    # ---------------------------------------------------------------------------
    def __init__(self, **traits):
        self.class_inputs = traits
        super(LandscapeView, self).__init__(**traits)

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
        self.scene3d.scene.interactor.add_observer("EndInteractionEvent", self.get_projected_coords)

    @on_trait_change('scene3d.activated')
    def display_landscape(self):
        data = self.class_inputs["data"]
        scatter = mlab.pipeline.scalar_scatter(data[:, 0], data[:, 1], data[:, 2],
                                               figure=self.scene3d.mayavi_scene)
        scatter = mlab.pipeline.glyph(scatter,
                                      opacity=self.opacity,
                                      scale_mode='none', scale_factor=.5, mode='sphere', colormap="hot",
                                      figure=self.scene3d.mayavi_scene)
        scatter.actor.actor.pickable = 0
        setattr(self, 'ipw_pc', scatter)
        # self.ipw_pc.add_observer('InteractionEvent', self.get_projected_coords)

    @on_trait_change("opacity")
    def change_opacity_point_cloud(self):
        self.ipw_pc.actor.property.opacity = self.opacity

    @on_trait_change("scale_points")
    def change_scale_point_cloud(self):
        scale_factor = float(self.scale_points)
        pts = getattr(self, 'ipw_pc')
        pts.glyph.glyph.scale_factor = scale_factor

    def get_projected_coords(self, obj, evt):
        data = self.class_inputs["data"]
        path = self.class_inputs["path"]
        W = np.ones((data.shape[0], 1))
        hmgns_world_coords = np.column_stack((data, W))
        comb_trans_mat = self.get_world_to_view_matrix()
        view_coords = self.apply_transform_to_points(hmgns_world_coords, comb_trans_mat)
        # norm_view_coords = view_coords / (view_coords[:, 3].reshape(-1, 1))
        # view_to_disp_mat = self.get_view_to_display_matrix()
        # disp_coords = self.apply_transform_to_points(norm_view_coords, view_to_disp_mat)
        coords = view_coords[:, :-1]
        np.savetxt(os.path.join(path, "temp.txt"), coords)
        np.savetxt(os.path.join(path, "comb_trans_mat.txt"), comb_trans_mat)
        # np.savetxt(os.path.join(path, "view_to_disp_mat.txt"), view_to_disp_mat)
        os.rename(os.path.join(path, "temp.txt"),
                  os.path.join(path, "projected_coords.txt"))

    # ---------------------------------------------------------------------------
    # Conversion functions
    # ---------------------------------------------------------------------------

    # ---------------------------------------------------------------------------
    # Read functions
    # ---------------------------------------------------------------------------

    # ---------------------------------------------------------------------------
    # View functions
    # ---------------------------------------------------------------------------
    def get_world_to_view_matrix(self):
        """returns the 4x4 matrix that is a concatenation of the modelview transform and
        perspective transform. Takes as input an mlab scene object."""

        # The VTK method needs the aspect ratio and near and far clipping planes
        # in order to return the proper transform. So we query the current scene
        # object to get the parameters we need.
        scene_size = tuple(self.scene3d.get_size())
        clip_range = self.scene3d.camera.clipping_range
        aspect_ratio = float(scene_size[0]) / float(scene_size[1])

        # this actually just gets a vtk matrix object, we can't really do anything with it yet
        vtk_comb_trans_mat = self.scene3d.camera.get_composite_projection_transform_matrix(
            aspect_ratio, clip_range[0], clip_range[1])

        # get the vtk mat as a numpy array
        np_comb_trans_mat = vtk_comb_trans_mat.to_array()

        return np_comb_trans_mat

    def get_view_to_display_matrix(self):
        """ this function returns a 4x4 matrix that will convert normalized
            view coordinates to display coordinates. It's assumed that the view should
            take up the entire window and that the origin of the window is in the
            upper left corner"""

        # this gets the client size of the window
        x, y = tuple(self.scene3d.get_size())

        # normalized view coordinates have the origin in the middle of the space
        # so we need to scale by width and height of the display window and shift
        # by half width and half height. The matrix accomplishes that.
        view_to_disp_mat = np.array([[x / 2.0, 0., 0., x / 2.0],
                                     [0., -y / 2.0, 0., y / 2.0],
                                     [0., 0., 1., 0.],
                                     [0., 0., 0., 1.]])

        return view_to_disp_mat

    def apply_transform_to_points(self, points, trans_mat):
        """a function that applies a 4x4 transformation matrix to an of
            homogeneous points. The array of points should have shape Nx4"""

        if not trans_mat.shape == (4, 4):
            raise ValueError('transform matrix must be 4x4')

        if not points.shape[1] == 4:
            raise ValueError('point array must have shape Nx4')

        return np.dot(trans_mat, points.T).T

    # ---------------------------------------------------------------------------
    # The layout of the dialog created
    # ---------------------------------------------------------------------------
    view = View(HSplit(
        HGroup(
            Group(
                Item('scene3d',
                     editor=SceneEditor(scene_class=MayaviScene),
                     height=250, width=300),
                Item('opacity',
                     editor=RangeEditor(format='%.02f', low_name='op_min', high_name='op_max', mode='slider'),
                     ),
                Item('scale_points',
                     editor=RangeEditor(format='%.02f', low_name='sc_min', high_name='sc_max', mode='slider'),
                     ),
                show_labels=False
            )
        )
    ),
    resizable=True,
    title='Point cloud clustering',
    icon=os.path.join(os.path.dirname(flexutils.__file__), "icon_square.png")
    )


# -------- Mayavi QWidget -------
class MayaviQWidget(QtWidgets.QWidget):
    def __init__(self, parent=None, view=None, **kwargs):
        QtWidgets.QWidget.__init__(self, parent)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        if view == "volume_viewer":
            self.visualization = MapView(**kwargs)
        elif view == "landscape":
            self.visualization = LandscapeView(**kwargs)

        # The edit_traits call will generate the widget to embed.
        self.ui = self.visualization.edit_traits(parent=self,
                                                 kind='subpanel').control
        layout.addWidget(self.ui)
        self.ui.setParent(self)


# -------- Viewer call -------
if __name__ == '__main__':
    import argparse

    # Input parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, required=True)
    parser.add_argument('--z_space', type=str, required=True)
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

    # Input
    input_dict = vars(args)
    input_dict['data'] = data
    input_dict['z_space'] = z_space

    # Initialize volume slicer
    viewer = InteractiveAnnotate2D(**input_dict)
    viewer.launchViewer()