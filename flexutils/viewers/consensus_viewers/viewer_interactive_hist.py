

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
import seaborn as sns
import pandas as pd

from pyworkflow import Config

sns.set_theme()


class InteractiveHist(object):

    def __init__(self, data, protocol):
        self.data = pd.Series(data, name="data")
        self.protocol = protocol
        self.threshold = None
        self.idx_selected = None
        self.total_selected = None

    def update(self, val):
        # Update threshold value
        self.threshold = val

        # Update images selected
        self.idx_selected = np.argwhere(self.data.values <= self.threshold)

        # Update total selected
        self.total_selected = self.idx_selected.size

        # Update the position of the vertical lines
        self.limit_line.set_xdata([val, val])

        # Interactive title
        self.axs.set_title("Total images selected: %d" % self.total_selected)

        # Redraw the figure to ensure it updates
        self.fig.canvas.draw_idle()

    def _plotButton(self):
        axcreateSubset = plt.axes([0.75, 0.02, 0.2, 0.050])
        # Button does not allow to define text color so
        # I write it directly
        color = Config.SCIPION_MAIN_COLOR if hasattr(Config, "SCIPION_MAIN_COLOR") else "maroon"
        axcreateSubset.text(0.5, 0.5, 'Subset Particles',
                            verticalalignment='center',
                            horizontalalignment='center',
                            transform=axcreateSubset.transAxes, color='white')
        bcreateSubset = Button(axcreateSubset, '',  # leave label empty
                               color=color,
                               hovercolor='maroon')
        bcreateSubset.on_clicked(self.createSubset)
        return bcreateSubset

    def createSubset(self, event):
        # Save selected indexes
        np.savetxt(self.protocol._getExtraPath("selected_idx.txt"), self.idx_selected)

        # Save output in protocol
        self.protocol._createOutput()

    def show(self):
        self.fig, self.axs = plt.subplots(1, 1, figsize=(10, 5))
        self.fig.subplots_adjust(bottom=0.25)

        # Create histogram
        sns.kdeplot(self.data, ax=self.axs, fill=True, linewidth=0, color="green")

        # Set axis labels
        self.axs.set_xlabel("RMSE")
        self.axs.set_ylabel("Density")

        # Create the Slider
        slider_ax = self.fig.add_axes([0.20, 0.1, 0.60, 0.03])
        self.slider = Slider(slider_ax, "Threshold", self.data.min(), self.data.max())

        # Create the Vertical lines on the histogram
        self.limit_line = self.axs.axvline(self.slider.val, color='k')

        # Update threshold value
        self.threshold = self.slider.val

        # Update images selected
        self.idx_selected = np.argwhere(self.data.values <= self.threshold)

        # Update total selected
        self.total_selected = self.idx_selected.size

        # Interactive title
        self.axs.set_title("Total images selected: %d" % self.total_selected)

        # Subset button
        self.bcreateSubset = self._plotButton()  # If not assign to something it might be deleted

        # Slider callback
        self.slider.on_changed(self.update)

        plt.show()

