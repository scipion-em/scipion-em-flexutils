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


import os
import numpy as np
import socket
from contextlib import closing
import pickle
import types
import struct


class Server:

    def __init__(self, mode, metadata, port=None, verbose=False):
        self.host = socket.gethostname()
        self.port = port if port is not None else self.getFreePort()
        self.verbose = verbose
        self.metadata = metadata
        self.mode = mode

        # Socket initialization
        self.createSocket()
        self.bindSocket()

        # Prepare map generation
        self.prepareMapGeneration()

        # Listen to client
        self.addListener()

    @classmethod
    def getFreePort(cls):
        with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
            s.bind(('', 0))
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            port = s.getsockname()[1]
        return port

    def createSocket(self):
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    def bindSocket(self):
        self.server_socket.bind((self.host, self.port))

    def addListener(self):
        # threading.Thread(target=self.listenToClient).start()
        self.listenToClient()
        if self.verbose:
            print(f"Server listening on {self.host}:{self.port}")

    def listenToClient(self):
        self.server_socket.listen(1)
        self.allowConnection()
        while True:
            try:
                # data = self.client_socket.recv(4096, socket.MSG_DONTWAIT | socket.MSG_PEEK)
                # if data:
                self.generateMap()
            except ConnectionResetError:
                return False

    def allowConnection(self):
        self.client_socket, addr = self.server_socket.accept()
        if self.verbose:
            print(f"Got a connection from {addr}")

    def prepareMapGeneration(self):
        if self.mode == "Zernike3D":
            pass
        elif self.mode == "CryoDrgn":
            import torch
            from cryodrgn.models import HetOnlyVAE
            from cryodrgn import config
            self.outPath = os.path.join(self.metadata["outdir"], "vol_{:03d}.mrc".format(0))
            use_cuda = torch.cuda.is_available()
            device = torch.device("cuda" if use_cuda else "cpu")
            args = types.SimpleNamespace()
            args.norm = None
            args.D = None
            args.l_extent = None
            args.vol_start_index = 0
            args.Apix = 1.0
            args.flip = False
            args.invert = False
            args.downsample = None
            args.qlayers = None
            args.qdim = None
            args.zdim = None
            args.players = None
            args.pdim = None
            args.enc_mask = None
            args.pe_type = None
            args.feat_sigma = None
            args.pe_dim = None
            args.domain = None
            args.encode_mode = None
            args.activation = "relu"
            cfg = config.overwrite_config(self.metadata["config"], args)
            self.norm = [float(x) for x in cfg["dataset_args"]["norm"]]
            self.model, self.lattice = HetOnlyVAE.load(cfg, self.metadata["weights"], device=device)
            self.model.eval()
        elif self.mode == "HetSIREN":
            import h5py
            from pathlib import Path
            from tensorflow_toolkit.generators.generator_het_siren import Generator
            from tensorflow_toolkit.networks.het_siren import AutoEncoder
            md_file = Path(Path(self.metadata["weights"]).parent.parent, "input_particles.xmd")
            self.outPath = os.path.join(self.metadata["outdir"], "decoded_map_class_{:02d}.mrc".format(1))

            # Get xsize from weights file
            f = h5py.File(self.metadata["weights"], 'r')
            xsize = int(np.sqrt(f["encoder"]["dense"]["kernel:0"].shape[0]))

            # Create data generator
            generator = Generator(md_file=md_file, step=1, shuffle=False,
                                  xsize=xsize)

            # Load model
            self.autoencoder = AutoEncoder(generator, het_dim=self.metadata["lat_dim"],
                                           architecture=self.metadata["architecture"])
            if generator.mode == "spa":
                self.autoencoder.build(input_shape=(None, generator.xsize, generator.xsize, 1))
            elif generator.mode == "tomo":
                self.autoencoder.build(input_shape=[(None, generator.xsize, generator.xsize, 1),
                                                    [None, generator.sinusoid_table.shape[1]]])
            self.autoencoder.load_weights(self.metadata["weights"])

        elif self.mode == "NMA":
            pass

    def generateMap(self):
        raw_msglen = self.recMsg(4)
        msglen = struct.unpack('>I', raw_msglen)[0]
        z = self.recMsg(msglen)
        z = pickle.loads(z)

        if self.mode == "Zernike3D":
            pass
        elif self.mode == "CryoDrgn":
            from cryodrgn.mrc import write
            for zz in z:
                vol = self.model.decoder.eval_volume(
                    self.lattice.coords, self.lattice.D, self.lattice.extent, self.norm, zz
                )
                write(
                    self.outPath, np.array(vol).astype(np.float32), Apix=1.0
                )
        elif self.mode == "HetSIREN":
            from xmipp_metadata.image_handler import ImageHandler
            decoded_maps = self.autoencoder.eval_volume_het(z, allCoords=True, filter=self.metadata["filter"])
            ImageHandler().write(decoded_maps, self.outPath, overwrite=True)
        elif self.mode == "NMA":
            pass

        self.client_socket.sendall("Map generated".encode())

    def recMsg(self, n):
        data = bytearray()
        while len(data) < n:
            packet = self.client_socket.recv(n - len(data))
            if not packet:
                return None
            data.extend(packet)
        return data

    def closeConnection(self):
        self.client_socket.close()
        self.server_socket.close()


if __name__ == "__main__":
    import argparse

    # Input parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--metadata_file', type=str, required=True)
    parser.add_argument('--port', type=int, required=True)
    parser.add_argument('--mode', type=str, required=True)

    args = parser.parse_args()

    with open(args.metadata_file, 'rb') as fp:
        metadata = pickle.load(fp)

    server = Server(args.mode, metadata, port=args.port)
