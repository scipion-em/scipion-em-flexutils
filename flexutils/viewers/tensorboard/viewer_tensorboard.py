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

from pyworkflow.viewer import Viewer
from pyworkflow.utils.process import buildRunCommand

from flexutils.protocols.xmipp.protocol_angular_align_zernike3deep import TensorflowProtAngularAlignmentZernike3Deep
from flexutils.protocols.xmipp.protocol_angular_align_het_siren import TensorflowProtAngularAlignmentHetSiren
from flexutils.protocols.xmipp.protocol_angular_align_deep_pose import TensorflowProtAngularAlignmentDeepPose
from flexutils.protocols.xmipp.protocol_angular_align_homo_siren import TensorflowProtAngularAlignmentHomoSiren

import flexutils.constants as const
import flexutils


class TensorboardViewer(Viewer):
    """ Tensorboard visualization of neural networks """
    _label = 'viewer conformational landscape'
    _targets = [TensorflowProtAngularAlignmentZernike3Deep,
                TensorflowProtAngularAlignmentHetSiren,
                TensorflowProtAngularAlignmentDeepPose,
                TensorflowProtAngularAlignmentHomoSiren]

    def __init__(self, **kwargs):
        Viewer.__init__(self, **kwargs)
        self._data = None

    def _visualize(self, obj, **kwargs):
        logdir_path = self.protocol._getExtraPath(os.path.join("network", "logs"))

        def launchViewerNonBlocking(args):
            (logdir_path, ) = args

            if not os.path.isdir(logdir_path):
                raise NotADirectoryError("Logs directory has not been generated for this network. "
                                         "Please, execute the training again with an updated version "
                                         "of the Flexutils-Tensorflow toolkit to generate the logs.")

            # Run tensorboard
            args = f"--logdir_path {logdir_path}"
            program = os.path.join(const.VIEWERS, "tensorboard", "setup_tensorboard.py")
            program = flexutils.Plugin.getProgram(program)

            command = buildRunCommand(program, args, 1)
            subprocess.Popen(command, shell=True)

        # Launch with Pathos
        p = Pool()
        # p.restart()
        p.apipe(launchViewerNonBlocking, args=(logdir_path, ))
        # p.join()
        # p.close()

        return []