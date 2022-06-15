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

from pyworkflow.viewer import DESKTOP_TKINTER, WEB_DJANGO, ProtocolViewer
import pyworkflow.protocol.params as params
from pyworkflow.utils.process import runJob

from xmipp3.protocols.protocol_angular_alignment_zernike3d import XmippProtAngularAlignmentZernike3D

import flexutils.constants as const
from flexutils.utils import computeNormRows
import flexutils


class XmippAngularAlignmentZernike3DViewer(ProtocolViewer):
    """ Visualize Zernike3D coefficient space """
    _label = 'viewer angular align - Zernike3D'
    _targets = [XmippProtAngularAlignmentZernike3D]
    _environments = [DESKTOP_TKINTER, WEB_DJANGO]

    def __init__(self, **kwargs):
        ProtocolViewer.__init__(self, **kwargs)
        self._data = None

    def getData(self):
        if self._data is None:
            self._data = self.loadData()
        return self._data


    def _defineParams(self, form):
        form.addSection(label='Show Zernike3D coefficient space')
        form.addParam('mode', params.EnumParam, choices=['UMAP', 'PCA'],
                      default=0, display=params.EnumParam.DISPLAY_HLIST,
                      label="Dimensionality reduction method",
                      help="\t * UMAP: usually leads to more meaningfull spaces, although execution "
                           "is higher\n"
                           "\t * PCA: faster but less meaningfull spaces \n"
                           "UMAP and PCA are only computed the first time the are used. Afterwards, they "
                           "will be reused to increase performance")
        form.addParam('nb_umap', params.IntParam, label="UMAP neighbors",
                      default=5, condition="mode==0",
                      help="Number of neighbors to associate to each point in the space when computing "
                           "the UMAP space. The higher the number of neighbors, the more predominant "
                           "global in the original space features will be")
        form.addParam('epochs_umap', params.IntParam, label="Number of UMAP epochs",
                      default=1000, condition="mode==0",
                      help="Increasing the number of epochs will lead to more accurate UMAP spaces at the cost "
                           "of larger execution times")
        form.addParam('densmap_umap', params.BooleanParam, label="Compute DENSMAP?",
                      default=False, condition="mode==0",
                      help="DENSMAP will try to bring densities in the UMAP space closer to each other. Execution time "
                           "will increase when computing a DENSMAP")
        form.addParam('doShowSpace', params.LabelParam,
                      label="Display the Zernike3D coefficient space")

    def _getVisualizeDict(self):
        self.protocol._createFilenameTemplates()
        return {'doShowSpace': self._doShowSpace}

    def _doShowSpace(self, param=None):
        z_clnm = []
        for particle in self.protocol.outputParticles.iterItems():
            z_clnm.append(np.fromstring(particle._xmipp_sphCoefficients.get(), sep=","))
        z_clnm = np.asarray(z_clnm)

        # Generate files to call command line
        file_z_clnm = self.protocol._getExtraPath("z_clnm.txt")
        file_deformation = self.protocol._getExtraPath("deformation.txt")
        np.savetxt(file_z_clnm, z_clnm)

        # Compute/Read UMAP or PCA
        mode = self.mode.get()
        if mode == 0:
            file_coords = self.protocol._getExtraPath("umap_coords.txt")
            if not os.path.isfile(file_coords):
                args = "--input %s --umap --output %s --n_neighbors %d --n_epochs %d " \
                       % (file_z_clnm, file_coords, self.nb_umap.get(), self.epochs_umap.get())
                if self.densmap_umap.get():
                    args += " --densmap"
                program = os.path.join(const.XMIPP_SCRIPTS, "dimensionality_reduction.py")
                program = flexutils.Plugin.getProgram(program)
                runJob(None, program, args)
        elif mode == 1:
            file_coords = self.protocol._getExtraPath("pca_coords.txt")
            if not os.path.isfile(file_coords):
                args = "--input %s --pca --output %s" % (file_z_clnm, file_coords)
                program = os.path.join(const.XMIPP_SCRIPTS, "dimensionality_reduction.py")
                program = flexutils.Plugin.getProgram(program)
                runJob(None, program, args)
        deformation = computeNormRows(z_clnm)

        # Generate files to call command line
        np.savetxt(file_deformation, deformation)

        # Run slicer
        args = "--coords %s --deformation %s" \
               % (file_coords, file_deformation)
        program = os.path.join(const.VIEWERS, "viewer_point_cloud.py")
        program = flexutils.Plugin.getProgram(program)
        runJob(None, program, args)
