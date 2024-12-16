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


import subprocess
import time

import socket
from contextlib import closing
import webbrowser

import flexutils


class ViewTensorboard:

    # ---------------------------------------------------------------------------
    # Init
    # ---------------------------------------------------------------------------
    def __init__(self, logdir_path):
        super(ViewTensorboard, self).__init__()
        # Setup Tensorboard
        self.port = self.launchTensorboard(logdir_path)

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
        # Web engine to render plot
        webbrowser.open_new(f"http://localhost:{self.port}/")


# -------- Viewer call -------
if __name__ == '__main__':
    import argparse

    # Input parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--logdir_path', type=str, required=True)

    # Input
    args = parser.parse_args()
    input_dict = vars(args)

    # Initialize volume slicer
    viewer = ViewTensorboard(**input_dict)

    # Execute Tensorboard
    viewer.launchViewer()
