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
import numpy as np
from multiprocessing import Process
from xmipp_metadata.image_handler import ImageHandler

from watchdog.observers import Observer
from watchdog.events import LoggingEventHandler

from bokeh.palettes import all_palettes
from sklearn.neighbors import KDTree
from sklearn.cluster import KMeans

from bokeh.models import HoverTool, LinearColorMapper, PointDrawTool, \
                         ColumnDataSource
from bokeh.plotting import figure
from bokeh.models import Select, Button, TextInput
from bokeh.layouts import column, layout
from bokeh.server.server import Server

from PyQt5 import QtWidgets, QtWebEngineWidgets, QtGui
from PyQt5.QtCore import QUrl, QTimer

from traits.api import HasTraits, Instance, Array, Float, String, on_trait_change
from traitsui.api import View, Item, HGroup, Group, HSplit, RangeEditor

from mayavi import mlab
from mayavi.core.api import PipelineBase
from mayavi.core.ui.api import SceneEditor, MayaviScene, MlabSceneModel

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
        self.data_tree = KDTree(self.data)
        self.z_space_tree = KDTree(self.z_space)
        self.adding_kmeans = False
        self.renaming_selection = False

        # Setup main window
        self.fig = figure(title="HexBin flexible landscape",
                          match_aspect=True, tools="wheel_zoom,reset,pan",
                          background_fill_color='black',
                          toolbar_location="below",
                          plot_width=650, plot_height=600)
        self.fig.grid.visible = False

        # Setup Mayavi window
        self.createMayaviInterface(**kwargs)

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
        lay_2.addWidget(self.mayavi_view)
        window_lay.addLayout(lay_1)
        window_lay.addLayout(lay_2)
        self.window.setFixedSize(1600, 650)
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
        # Colormap control
        colormaps = {"Viridis": "Viridis", "Cividis": "Cividis", "Plasma": "Plasma",
                     "Inferno": "Inferno", "BuPu": "BuPu", "Oranges": "Oranges",
                     "PuRd": "PuRd"}
        colormap_select = Select(value="Viridis", title='Colormap', options=sorted(colormaps.keys()))
        colormap_select.on_change('value', self.update_color_map)

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
        self.hexbin, self.bins = self.fig.hexbin(self.data[:, 0], self.data[:, 1], size=0.1, level="underlay")

        # Layout
        controls_colormap = column(colormap_select)
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

    def createMayaviInterface(self, **kwargs):
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
        elif self.mode == "HetSIREN":
            inputs_mayavi = {"path": self.path,
                             "mode": self.mode,
                             "weights": kwargs.get("weights"),
                             "step": kwargs.get("step"),
                             "sr": float(self.class_inputs["sr"])}
        elif self.mode == "NMA":
            inputs_mayavi = {"path": self.path,
                             "mode": self.mode,
                             "weights": kwargs.get("weights"),
                             "sr": float(self.class_inputs["sr"])}
        self.mayavi_view = MayaviQWidget(**inputs_mayavi)

    # ---------------------------------------------------------------------------
    # Tools
    # ---------------------------------------------------------------------------
    def initiatePointDrawTool(self):
        # Read selections or make default
        # self.seeds = ColumnDataSource({'x': [], 'y': [], 'names': []})
        self.loadSelections()

        # Selection callbacks
        self.seeds.on_change("data", self.set_default_label)
        self.seeds.on_change("data", self.save_z_mayavi)
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
        color_mapper = LinearColorMapper(palette=palette, low=np.amin(self.bins["counts"]),
                                         high=np.amax(self.bins["counts"]))
        # color_mapper = LinearColorMapper(palette=new, low=np.amin(), high=np.amax())
        self.hexbin.glyph.fill_color['transform'].palette = color_mapper.palette

    def save_selections(self):
        selections = self.seeds.data
        selected_points = np.asarray([selections["x"], selections["y"]]).T
        _, idx = self.data_tree.query(selected_points, k=1)
        selected_z_space = self.z_space[np.asarray(idx).reshape(-1), :]

        np.savetxt(os.path.join(self.path, "saved_selections.txt"), selected_z_space)
        np.savetxt(os.path.join(self.path, "selections.txt"), selected_points, delimiter=",")
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

    def set_default_label(self, attrname, old, new):
        if not self.adding_kmeans:
            if len(new["names"]) > len(old["names"]):
                names = old["names"]
                names.append("class_%d" % (len(new["names"]) - 1))
                self.seeds.data['names'] = names

    def save_z_mayavi(self, attrname, old, new):
        if not self.adding_kmeans and not self.renaming_selection:
            x, y = None, None
            if isinstance(new, dict):
                x = list(set(new["x"]) - set(old["x"]))
                y = list(set(new["y"]) - set(old["y"]))
                if len(x) == 0 and len(y) == 0:
                    x = new["x"][-1]
                    y = new["y"][-1]
            elif isinstance(new, list) and len(new) == 1:
                x = self.seeds.data["x"][new[0]]
                y = self.seeds.data["y"][new[0]]

            if x and y:
                selected_data = np.asarray([x, y]).reshape(1, -1)
                _, idx = self.data_tree.query(selected_data, k=1)
                z = self.z_space[idx[0], :]
                np.savetxt(os.path.join(self.path, "temp.txt"), z, delimiter=",")
                os.rename(os.path.join(self.path, "temp.txt"),
                          os.path.join(self.path, "z_space_mayavi.txt"))

    def save_selections_chimerax(self):
        selected_data = self.seeds.data
        selected_points = np.asarray([selected_data["x"], selected_data["y"]]).T
        _, idx = self.data_tree.query(selected_points, k=1)
        selected_z_space = self.z_space[np.asarray(idx).reshape(-1), :]
        sel_names = np.asarray(selected_data["names"])

        with open(os.path.join(self.path, 'selection_names_chimerax.txt'), 'w') as f:
            for line in sel_names:
                f.write(line + "\n")
            f.write(self.path_select.value + "\n")
        np.savetxt(os.path.join(self.path, "temp.txt"), selected_z_space, delimiter=",")
        os.rename(os.path.join(self.path, "temp.txt"),
                  os.path.join(self.path, "z_space_chimerax.txt"))

    def update_mayavi_map(self):
        self.mayavi_view.visualization.updateMap()
        # self.web_engine.update()
        # self.mayavi_view.update()

    def cluster_space_kmeans(self):
        n_clusters = int(self.n_clusters.value)
        # First delete all automatic selections associated to a previous KMeans
        new_x, new_y, new_names = [], [], []
        old_x, old_y, old_names = self.seeds.data["x"], self.seeds.data["y"], self.seeds.data["names"]
        for x, y, name in zip(old_x, old_y, old_names):
            if not 'kmean' in name:
                new_x.append(x)
                new_y.append(y)
                new_names.append(name)
        # Compute KMeans and save automatic selection
        if int(n_clusters) > 0:
            clusters = KMeans(n_clusters=int(n_clusters), n_init=1).fit(self.z_space)
            centers = clusters.cluster_centers_
            _, inds = self.z_space_tree.query(centers, k=1)
            for idx, ind in enumerate(inds):
                if not self.data[ind[0], 0] in old_x and not self.data[ind[0], 1] in old_y:
                    new_x.append(self.data[ind[0], 0])
                    new_y.append(self.data[ind[0], 1])
                    new_names.append("kmean_%d" % (idx + 1))
        self.adding_kmeans = True
        new_data = {"x": new_x, "y": new_y, "names": new_names}
        self.seeds.data = new_data
        self.adding_kmeans = False

    def morph_selections_chimerax(self):
        # Morph maps in chimerax based on different ordering methods
        from flexutils.viewers.chimera_viewers.viewer_morph_chimerax import FlexMorphChimeraX
        z_space_selected = np.loadtxt(os.path.join(self.path, "z_space_chimerax.txt"), delimiter=",")
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
            selections = np.loadtxt(os.path.join(self.path, "selections.txt"), delimiter=",")
            with open(os.path.join(self.path, 'selection_names.txt'), 'r') as f:
                names = [line.strip("\n") for line in f.readlines()]
            self.seeds = ColumnDataSource({'x': selections[:, 0].tolist(),
                                           'y': selections[:, 1].tolist(),
                                           'names': names})
        else:
            x, y, names = [], [], []
            # if "n_vol" in self.class_inputs:
            #     n_vol = int(self.class_inputs["n_vol"])
            #     x, y = self.data[-n_vol:, 0].tolist(), self.data[-n_vol:, 1].tolist()
            #     names.append("reference")
            #     for idx in range(1, n_vol):
            #         names.append("class_%d" % idx)
            data = {'x': x, 'y': y, 'names': names}
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
                self.generated_map = self.readMap(os.path.join(self.path, "decoded_map_class_1.mrc"))
            elif self.mode == "NMA":
                self.generate_map(self.class_inputs["weights"], z,
                                  self.path, sr=self.class_inputs["sr"])
                self.generated_map = self.readMap(os.path.join(self.path, "decoded_map_class_1.mrc"))
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


# -------- Mayavi QWidget -------
class MayaviQWidget(QtWidgets.QWidget):
    def __init__(self, parent=None, **kwargs):
        QtWidgets.QWidget.__init__(self, parent)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        self.visualization = MapView(**kwargs)

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
