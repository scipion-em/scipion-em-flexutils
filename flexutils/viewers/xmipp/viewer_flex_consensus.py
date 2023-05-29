# **************************************************************************
# *
# * Authors:  David Herreros Calero (dherreros@cnb.csic.es)
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

from pyworkflow.viewer import DESKTOP_TKINTER, WEB_DJANGO, ProtocolViewer
import pyworkflow.protocol.params as params

from flexutils.protocols.xmipp.protocol_interactive_flex_consensus import TensorflowProtInteractiveFlexConsensus
from flexutils.viewers.consensus_viewers.viewer_interactive_hist import InteractiveHist


class XmippFlexConsensusView(ProtocolViewer):
    """ Interactive FlexConsensus filtering """
    _label = 'viewer FlexConsensus'
    _targets = [TensorflowProtInteractiveFlexConsensus]
    _environments = [DESKTOP_TKINTER, WEB_DJANGO]
    _choices = ["Mean matched", "Entropy matched"]

    def __init__(self, **kwargs):
        ProtocolViewer.__init__(self, **kwargs)

    def _defineParams(self, form):
        form.addSection(label='Show deformation')
        form.addParam('histChoice', params.EnumParam,
                      choices=self._choices, default=0,
                      label='Error histogram to display', display=params.EnumParam.DISPLAY_COMBO,
                      help="\t Mean matched: Error distribution matched to the lowest error distribution " \
                           "found by FlexConsensus\n" \
                           "\t Entropy matched: Error distribution matched to the distribution with highest " \
                           "entropy found by FlexConsensus")
        form.addParam('doShowHist', params.LabelParam,
                      label="Display the selected histogram in interactive mode")

    def _getVisualizeDict(self):
        self.chosen = self._choices[self.histChoice.get()]
        return {'doShowHist': self._doShowHist}

    # ------------------- Interactive histogram method -------------------
    def _doShowHist(self, param=None):
        # Load data
        if self.chosen == self._choices[0]:
            data = np.loadtxt(self.protocol._getExtraPath("error_matched_mean.txt"))
        elif self.chosen == self._choices[1]:
            data = np.loadtxt(self.protocol._getExtraPath("error_matched_entropy.txt"))

        # Interactive histogram
        hist = InteractiveHist(data, self.protocol)
        hist.show()

    # ------------------- ------------------- -------------------
