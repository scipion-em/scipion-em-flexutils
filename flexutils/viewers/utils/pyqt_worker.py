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

from PyQt5.QtCore import QObject, pyqtSignal


class GenerateVolumesWorker(QObject):
    """
    PyQt worker for non-blocking execution of volume generation in viewers
    """
    finished = pyqtSignal()
    volume = pyqtSignal(object)

    def __init__(self, mode, **kwargs):
        super(QObject, self).__init__()
        self.mode = mode
        self.kwargs = kwargs

    def readMap(self, file):
        map = ImageHandler().read(file).getData()
        return map

    def generateVolume(self):
        """
        Volume generation parameters:
            fn : function
                 volume generation function
            path : str
                 folder where volume are being generated
            mode : str
                 generation function program name
            kwargs :
                 fn parameters
        """
        # Get generation function
        if self.mode == "Zernike3D":
            from flexutils.protocols.xmipp.utils.utils import applyDeformationField
            fn = applyDeformationField
        elif self.mode == "CryoDrgn":
            import cryodrgn
            from cryodrgn.utils import generateVolumes
            cryodrgn.Plugin._defineVariables()
            fn = generateVolumes
        elif self.mode == "HetSIREN":
            from flexutils.utils import generateVolumesHetSIREN
            fn = generateVolumesHetSIREN
        elif self.mode == "NMA":
            from flexutils.utils import generateVolumesDeepNMA
            fn = generateVolumesDeepNMA

        # Generate volume
        fn(**self.kwargs)

        # Read generated volume
        if self.mode == "Zernike3D":
            path = self.kwargs.get("path")
            generated_map = self.readMap(os.path.join(path, "deformed.mrc"))
        elif self.mode == "CryoDrgn":
            path = self.kwargs.get("outdir")
            generated_map = self.readMap(os.path.join(path, "vol_000.mrc"))
        elif self.mode == "HetSIREN":
            path = self.kwargs.get("outdir")
            generated_map = self.readMap(os.path.join(path, "decoded_map_class_01.mrc"))
        elif self.mode == "NMA":
            path = self.kwargs.get("outdir")
            generated_map = self.readMap(os.path.join(path, "decoded_map_class_01.mrc"))

        # Emit signals
        self.volume.emit(generated_map)
        self.finished.emit()
