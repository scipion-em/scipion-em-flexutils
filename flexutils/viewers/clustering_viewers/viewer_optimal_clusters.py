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

from pyworkflow.protocol.params import LabelParam, EnumParam
from pyworkflow.viewer import DESKTOP_TKINTER
from pwem.viewers import EmProtocolViewer

from flexutils.protocols.protocol_find_optimal_clusters import ProtFlexOptimalClusters


class FlexShowOptimalClustersViewer(EmProtocolViewer):
    """ Visualization of optimal cluster results. """

    _environments = [DESKTOP_TKINTER]
    _targets = [ProtFlexOptimalClusters]
    _label = 'analyze results'
    _method = ["Best clusters", "Calinski Harabasz", "Davies Bouldin",
               "Elbow", "Gap Statistic", "Silhouette"]

    def __init__(self, **kwargs):
        EmProtocolViewer.__init__(self, **kwargs)

    def _createFilenameTemplates(self):
        """ Centralize how files are called. """

        def _out(f):
            return self.protocol._getExtraPath(f)

        self._updateFilenamesDict({
            'Best clusters': _out('auto_clustering_results.png'),
            'Calinski Harabasz': _out('calinski_harabasz.png'),
            'Davies Bouldin': _out('davies_bouldin.png'),
            'Elbow': _out('elbow.png'),
            'Gap Statistic': _out('gap_statistic.png'),
            'Silhouette': _out('silhouette.png')
        })

    def _defineParams(self, form):
        form.addSection(label='Visualization')
        form.addParam('method', EnumParam,
                      choices=self._method,
                      default=0,
                      display=EnumParam.DISPLAY_COMBO,
                      label='Display results for analysis',
                      help="- Best clusters: Shows a scatter plot with the optimal clusters"
                           "found by each algorithm\n"
                           "- Calinski Harabasz: Shows the Calinski Harabasz plot (look for maxima)\n"
                           "- Davies Bouldin: Shows the Davies Bouldin plot (look for maxima)\n"
                           "- Elbow: Shows the Elbow plot (look for points close to the elbow)\n"
                           "- Gap Statistic: Shows the Gap Statistic plot (look for maxima)\n"
                           "- Silhouette: Shows the Silhouette plot (look for maxima)")
        form.addParam('doShowAnalysis', LabelParam,
                      label="Show analysis result")

    def _getVisualizeDict(self):
        self._createFilenameTemplates()

        visDict = {
            'doShowAnalysis': self._doShowAnalysis
        }

        return visDict

    def _doShowAnalysis(self, param=None):
        import matplotlib.image as mpimg
        import matplotlib.pyplot as plt
        fn = self._getFileName(self._method[self.method.get()])
        if os.path.exists(fn):
            img = mpimg.imread(fn)
            imgplot = plt.imshow(img)
            plt.axis('off')
            plt.show()
            return [imgplot]
        else:
            self.showError('File %s not found! Have you run analysis?' % fn)
