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
from xmipp_metadata.image_handler import ImageHandler

from pyworkflow.utils import runJob, moveFile

from PyQt5.QtCore import QThread, pyqtSignal

from flexutils.socket.client import Client


class ServerQThread(QThread):
    def __init__(self, program, metadata_file, mode, port, env):
        super().__init__()
        self.program = program
        self.metadata_file = metadata_file
        self.mode = mode
        self.port = port
        self.env = env

    def run(self):
        runJob(None, self.program,
               f"--metadata_file {self.metadata_file} --mode {self.mode} --port {self.port}",
               env=self.env)

class ClientQThread(QThread):
    finished = pyqtSignal()
    volume = pyqtSignal(object)
    chimera = pyqtSignal()

    def __init__(self, port, path, mode):
        super().__init__()
        self.client = Client(port)
        self.path = path
        self.mode = mode
        self.z = None
        self.file_names = None

    def readMap(self, file):
        map = ImageHandler().read(file).getData()
        return map

    def run(self):
        for z, file in zip(self.z, self.file_names):
            self.client.sendDataToSever(z[None, ...])

            # Read generated volume
            if self.mode == "Zernike3D":
                vol_file = os.path.join(self.path, "deformed.mrc")
            elif self.mode == "CryoDrgn":
                vol_file = os.path.join(os.path.join(self.path, "vol_000.mrc"))
            elif self.mode == "HetSIREN":
                vol_file = os.path.join(os.path.join(self.path, "decoded_map_class_01.mrc"))
            elif self.mode == "NMA":
                vol_file = os.path.join(os.path.join(self.path, "decoded_map_class_01.mrc"))

            # Emit signals
            if self.z.shape[0] == 1:
                generated_map = self.readMap(vol_file)
                self.volume.emit(generated_map)
            else:
                new_path = os.path.join(self.path, file + ".mrc")
                if new_path != vol_file:
                    moveFile(vol_file, new_path)

        # Emit signals
        if self.z.shape[0] > 1:
            self.chimera.emit()
        self.finished.emit()
