# **************************************************************************
# *
# * Authors:     David Herreros Calero (dherreos@cnb.csic.es)
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


import socket
import pickle
import struct


class Client:

    def __init__(self, port, verbose=False):
        self.host = socket.gethostname()
        self.port = port
        self.verbose = verbose

        # Socket initialization
        self.createSocket()
        self.connectToServer()

    def createSocket(self):
        self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    def connectToServer(self):
        connecting = True
        while connecting:
            try:
                self.client_socket.connect((self.host, self.port))
                connecting = False
            except:
                connecting = True
        if self.verbose:
            print(f"Connected to {self.host}:{self.port}")

    def sendDataToSever(self, message):
        serialized_message = pickle.dumps(message, protocol=2)
        serialized_message = struct.pack(">I", len(serialized_message)) + serialized_message
        self.client_socket.sendall(serialized_message)
        return self.client_socket.recv(4096).decode()

    def closeConnection(self):
        self.client_socket.close()
