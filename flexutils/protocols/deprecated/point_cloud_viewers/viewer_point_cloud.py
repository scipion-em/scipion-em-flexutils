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
import pyvista as pv
import pyvistaqt as pvqt


class pointCloudPlot(object):

    def __init__(self, points=None, deformation=None):
        self.points = points[:, :3]
        self.deformation = deformation

        # Define actors
        self.points_actor = None

        # Define PyVista objects
        self.pv_points = pv.PolyData(self.points)
        self.pv_points["deformation"] = self.deformation
        self.pv_points["clusters"] = np.zeros(self.points.shape[0])

        self.plt = pvqt.BackgroundPlotter(title='Embedding 3D interactive viewer')
        self.plt.main_menu.clear()

        # Slider
        def changeOpacity(value):
            if self.points_actor is not None:
                self.points_actor.GetProperty().SetOpacity(value / 100)

        self.plt.add_slider_widget(changeOpacity, [0, 100], value=50,
                                   pointa=(.6, .9), pointb=(.9, .9),
                                   title='Embedding opacity', style='modern')

    # Display PC
    def plotPoints(self):
        self.points_actor = self.plt.add_mesh(self.pv_points, show_scalar_bar=False, scalars="deformation",
                                              cmap="gist_rainbow_r", opacity=0.5,
                                              render_points_as_spheres=True, reset_camera=True)

    def initializePlot(self):
        self.plotPoints()
        self.plt.app.exec_()


if __name__ == '__main__':
    import argparse

    # Input parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--coords', type=str, required=True)
    parser.add_argument('--deformation', type=str, required=True)

    args = parser.parse_args()

    # Read and generate data
    coords = np.loadtxt(args.coords)
    deformation = np.loadtxt(args.deformation)

    # Initialize volume slicer
    m = pointCloudPlot(points=coords, deformation=deformation)
    m.initializePlot()
