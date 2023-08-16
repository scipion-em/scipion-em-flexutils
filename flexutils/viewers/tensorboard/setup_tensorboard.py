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


import sys
import os
import subprocess
import time

import psutil
import socket
from contextlib import closing

from PyQt5 import QtWidgets, QtWebEngineWidgets, QtGui
from PyQt5.QtCore import QUrl

from pyworkflow.utils import runJob

import flexutils


class ViewTensorboard(QtWidgets.QWidget):

    # ---------------------------------------------------------------------------
    # Init
    # ---------------------------------------------------------------------------
    def __init__(self, logdir_path):
        super(ViewTensorboard, self).__init__()
        # Setup Tensorboard
        port = self.launchTensorboard(logdir_path)

        # Web engine to render plot
        self.window = QtWidgets.QWidget()
        self.web_engine = QtWebEngineWidgets.QWebEngineView()
        # self.web_engine.setFixedWidth(1000)
        self.web_engine.load(QUrl(f"http://localhost:{port}/"))

        # PyQt layout
        window_lay = QtWidgets.QHBoxLayout()
        lay_1 = QtWidgets.QVBoxLayout()
        lay_1.addWidget(self.web_engine)
        window_lay.addLayout(lay_1)
        self.window.setBaseSize(1600, 650)
        self.window.setLayout(window_lay)
        self.setWindowIcon(QtGui.QIcon(os.path.join(os.path.dirname(flexutils.__file__), "icon_square.png")))

    # ---------------------------------------------------------------------------
    # Create interface
    # ---------------------------------------------------------------------------
    def launchTensorboard(self, logdir_path):
        def is_port_in_use(port: int) -> bool:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                return s.connect_ex(('localhost', port)) == 0

        with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
            s.bind(('', 0))
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            port = s.getsockname()[1]
        program = flexutils.Plugin.getTensorflowProgram("tensorboard", python=False)
        args = f" --logdir {logdir_path} --port {port}"
        self.tensorboard_process = subprocess.Popen(program + args, shell=True, stdout=subprocess.DEVNULL,
                                                    stderr=subprocess.DEVNULL)

        while not is_port_in_use(port):
            time.sleep(0.1)
        return port

    # ---------------------------------------------------------------------------
    # Tools
    # ---------------------------------------------------------------------------

    # ---------------------------------------------------------------------------
    # Callbacks
    # ---------------------------------------------------------------------------

    # ---------------------------------------------------------------------------
    # Read data
    # ---------------------------------------------------------------------------

    # ---------------------------------------------------------------------------
    # Launch values
    # ---------------------------------------------------------------------------
    def launchViewer(self):
        # Show interface
        self.window.show()

def kill(proc_pid):
    process = psutil.Process(proc_pid)
    for proc in process.children(recursive=True):
        proc.kill()
    process.kill()

# -------- Viewer call -------
if __name__ == '__main__':
    import argparse

    # Input parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--logdir_path', type=str, required=True)

    # Input
    args = parser.parse_args()
    input_dict = vars(args)

    # Get application
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication(sys.argv)

    # Initialize volume slicer
    viewer = ViewTensorboard(**input_dict)

    # Execute app
    viewer.launchViewer()
    app.exec_()

    # Close tensorboard process
    kill(viewer.tensorboard_process.pid)
