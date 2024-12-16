# **************************************************************************
# *
# * Authors:  David Herreros (dherreros@cnb.csic.es)
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
from pathos.multiprocessing import ProcessingPool as Pool
import subprocess

from pyworkflow.viewer import DESKTOP_TKINTER, WEB_DJANGO
from pyworkflow.utils.process import buildRunCommand
from pyworkflow.gui.dialog import showError

from pwem.viewers import DataViewer

from flexutils.protocols.xmipp.protocol_angular_align_zernike3deep import TensorflowProtAngularAlignmentZernike3Deep
from flexutils.protocols.xmipp.protocol_angular_align_het_siren import TensorflowProtAngularAlignmentHetSiren
from flexutils.protocols.xmipp.protocol_angular_align_reconsiren import TensorflowProtAngularAlignmentReconSiren

import flexutils.constants as const
import flexutils


class TensorboardViewer(DataViewer):
    """ Tensorboard visualization of neural networks """
    _label = 'viewer Tensorboard'
    _targets = [TensorflowProtAngularAlignmentZernike3Deep,
                TensorflowProtAngularAlignmentHetSiren,
                TensorflowProtAngularAlignmentReconSiren]
    _environments = [DESKTOP_TKINTER, WEB_DJANGO]

    def __init__(self, **kwargs):
        DataViewer.__init__(self, **kwargs)
        self._data = None

    def _visualize(self, obj, **kwargs):
        logdir_path = self.protocol._getExtraPath(os.path.join("network", "logs"))

        def launchViewerNonBlocking(logdir_path):

            # Run tensorboard
            args = f"--logdir_path {logdir_path}"
            program = os.path.join(const.VIEWERS, "tensorboard", "setup_tensorboard.py")
            program = flexutils.Plugin.getProgram(program, python=True, activateScipion=True)

            command = buildRunCommand(program, args, 1)
            subprocess.Popen(command, shell=True)

        if not os.path.isdir(logdir_path):
            if hasattr(self.protocol, "tensorboard"):
                if not self.protocol.tensorboard.get():
                    msg = ("Tensorboard cannot be opened because the option \"Allow Tensorboard visualization\" "
                           "in the protocol form has been set to \"No\". If you want to visualize one of the "
                           "outputs generated by this protocol, please, right click on the deried output from the "
                           "output list and choose the desired viewer to open it.")
            else:
                msg = "Tensorboard log files have not been properly generated."
            showError(title="Tensorboard log files not found", msg=msg, parent=self.getTkRoot())
            return []

        launchViewerNonBlocking(logdir_path)

        return []
