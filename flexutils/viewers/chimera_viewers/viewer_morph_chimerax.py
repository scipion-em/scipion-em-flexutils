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


import os
import numpy as np
from subprocess import Popen
from sklearn.neighbors import KDTree
from xmipp_metadata.image_handler import ImageHandler

from PyQt5.QtCore import QObject, pyqtSignal

from pyworkflow.utils.process import runJob

from pwem.viewers import Chimera

import flexutils
import flexutils.constants as const


class FlexMorphChimeraX(QObject):
    """ Morphing along different paths """
    OPEN_FILE = "open %s\n"
    VOXEL_SIZE = "volume #%d voxelSize %s\n"
    VOL_HIDE = "vol #%d hide\n"
    VIEW = "view\n"
    finished = pyqtSignal()

    def __init__(self, _z_space, _file_names, _path):
        super(QObject, self).__init__()
        self.z_space = _z_space
        self.file_names = _file_names
        self.path = _path

    def showSalesMan(self, param=None):
        # Get shortest path
        self.computeSalesManPath()
        path = self.readPath()[0] - 1
        path = path.astype(int)
        self.file_names = [self.file_names[i] for i in path]

        # # Useful parameters
        # smprt = self.protocol.reference.get().getSamplingRate()

        scriptFile = os.path.join(self.path, 'morph_orig_ref_chimera.cxc')
        fhCmd = open(scriptFile, 'w')

        for chid, file_name in enumerate(self.file_names):
            volFile = os.path.abspath(os.path.join(self.path, file_name + ".mrc"))
            fhCmd.write(self.OPEN_FILE % volFile)
            # fhCmd.write(self.VOXEL_SIZE % ((chid + 1), str(smprt)))
            fhCmd.write(self.VOL_HIDE % (chid + 1))

        num_vol = len(path)
        frames = 20 * num_vol
        step = 1 / frames
        fhCmd.write("volume morph #1-%d frames %d playStep %f \n" % (num_vol, frames, step))
        fhCmd.write(self.VIEW)
        fhCmd.close()

        self.openChimeraX(scriptFile)
        self.finished.emit()

    def showRandomWalk(self):
        path = self.computeRandomWalkPath()
        self.file_names = self.file_names[np.unique(path)]

        scriptFile = os.path.join(self.path, 'morph_orig_ref_chimera.cxc')
        fhCmd = open(scriptFile, 'w')

        for chid, file_name in enumerate(self.file_names):
            volFile = os.path.abspath(os.path.join(self.path, file_name + ".mrc"))
            fhCmd.write(self.OPEN_FILE % volFile)
            # fhCmd.write(self.VOXEL_SIZE % ((chid + 1), str(smprt)))
            fhCmd.write(self.VOL_HIDE % (chid + 1))

        num_vol = path.size
        frames = 20 * num_vol
        step = 1 / frames
        path_str = np.char.mod('%d', path)
        path_str = ",".join(path_str)
        print(path_str, frames, step)
        fhCmd.write("volume morph #%s frames %d playStep %f \n" % (path_str, frames, step))
        fhCmd.write(self.VIEW)
        fhCmd.close()

        self.openChimeraX(scriptFile)

    def computeSalesManPath(self):
        coordsPath = os.path.join(self.path, "current_selections.txt")
        outPath = os.path.join(self.path, "path.txt")

        np.savetxt(coordsPath, self.z_space)

        # Run salesman's solver
        program = "python " + os.path.join(os.path.dirname(flexutils.__file__), "viewers", "path_finder_tools", "viewer_salesman_solver.py")
        args = "--coords %s --outpath %s --num_vol 0 " \
               % (coordsPath, outPath)
        runJob(None, program, args)

    def computeRandomWalkPath(self):
        step_n = int(self.z_space.shape[0] * 10)
        origin = self.z_space[0].reshape(1, -1)
        tree = KDTree(self.z_space)
        path = [0]

        for _ in range(step_n):
            _, ids = tree.query(origin, k=3)
            ids = np.asarray(ids[0])
            new_origin_id = np.random.choice(a=ids, size=1)
            origin = self.z_space[new_origin_id].reshape(1, -1)
            path.append(new_origin_id[0])

        return np.asarray(path)

    def readPath(self):
        path = []
        with open(os.path.join(self.path, "path.txt")) as f:
            lines = f.readlines()
            for line in lines:
                path.append(np.fromstring(line, dtype=float, sep=','))
        return path

    def openChimeraX(self, scriptFile):
        # view = ChimeraView(scriptFile)
        # view.show()

        chimera_home = os.environ.get("CHIMERA_HOME")
        program = os.path.join(chimera_home, 'bin', os.path.basename("ChimeraX"))
        cmd = program + ' "%s"' % scriptFile
        Popen(cmd, shell=True, env=Chimera.getEnviron(), cwd=os.getcwd())