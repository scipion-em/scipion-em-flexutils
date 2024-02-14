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


import numpy as np
import os
from pathos.multiprocessing import ProcessingPool as Pool
import subprocess

from pyworkflow.viewer import DESKTOP_TKINTER, WEB_DJANGO, Viewer
from pyworkflow.utils.process import buildRunCommand

from flexutils.protocols.xmipp.protocol_structure_landscape import XmippProtStructureLanscapes
from flexutils.protocols.protocol_dimred import ProtFlexDimRedSpace

import flexutils.constants as const
from flexutils.utils import computeNormRows
import flexutils


class XmippReducedSpaceViewer(Viewer):
    """ Visualize reduced conformational space """
    _label = 'viewer reduced space'
    _targets = [XmippProtStructureLanscapes, ProtFlexDimRedSpace]
    _environments = [DESKTOP_TKINTER, WEB_DJANGO]

    def __init__(self, **kwargs):
        Viewer.__init__(self, **kwargs)
        self._data = None

    def getData(self):
        if self._data is None:
            self._data = self.loadData()
        return self._data

    def _visualize(self, obj, **kwargs):
        file_red_space = self.protocol._getExtraPath("red_coords.txt")
        file_z_space = self.protocol._getExtraPath("z_space.txt")
        file_interp = self.protocol._getExtraPath("interp_values.txt")
        particles = self.protocol.outputParticles

        def launchViewerNonBlocking(args):
            (particles, file_red_space, file_z_space, file_interp) = args
            red_space = []
            z_clnm = []
            for particle in particles.iterItems():
                z_clnm.append(particle.getZFlex())
                red_space.append(particle.getZRed())
            z_clnm = np.asarray(z_clnm)
            red_space = np.asarray(red_space)
            if red_space.shape[1] < 3:
                raise Exception("Visualization of spaces with dimension smaller than 3 is not yet implemented. Exiting...")

            # Generate files to call command line
            np.savetxt(file_red_space, red_space)
            # np.savetxt(file_z_space, z_clnm)

            if particles.getFlexInfo().getProgName() == const.ZERNIKE3D:
                deformation = computeNormRows(z_clnm)
            else:
                deformation = np.zeros(z_clnm.shape)

            # Generate files to call command line
            np.savetxt(file_interp, deformation)

            # Run slicer
            args = "--data %s --z_space %s --interp_val %s --onlyView" \
                   % (file_red_space, file_z_space, file_interp)
            program = os.path.join(const.VIEWERS, "annotation_3d_tools", "viewer_interactive_3d.py")
            program = flexutils.Plugin.getProgram(program)

            command = buildRunCommand(program, args, 1)
            subprocess.Popen(command, shell=True)

        # Launch with Pathos
        p = Pool()
        # p.restart()
        p.apipe(launchViewerNonBlocking, args=(particles, file_red_space, file_z_space, file_interp))
        # p.join()
        # p.close()

        return []
