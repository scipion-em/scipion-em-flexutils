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

from pyworkflow.viewer import DESKTOP_TKINTER, WEB_DJANGO, ProtocolViewer
import pyworkflow.protocol.params as params
from pyworkflow.utils.process import buildRunCommand

from flexutils.protocols.xmipp.protocol_angular_alignment_zernike3d import XmippProtAngularAlignmentZernike3D
from flexutils.protocols.xmipp.protocol_focus_zernike3d import XmippProtFocusZernike3D
from flexutils.protocols.xmipp.protocol_reassign_reference_zernike3d import XmippProtReassignReferenceZernike3D
from flexutils.protocols.xmipp.protocol_predict_zernike3deep import TensorflowProtPredictZernike3Deep
from flexutils.protocols.xmipp.protocol_predict_het_siren import TensorflowProtPredictHetSiren
from flexutils.protocols.protocol_score_landscape import ProtFlexScoreLandscape

import flexutils.constants as const
from flexutils.utils import computeNormRows
import flexutils


class XmippLandscapeViewer(ProtocolViewer):
    """ Visualize conformational lanscapes """
    _label = 'viewer conformational landscape'
    _targets = [XmippProtAngularAlignmentZernike3D, XmippProtFocusZernike3D,
                XmippProtReassignReferenceZernike3D, TensorflowProtPredictZernike3Deep,
                TensorflowProtPredictHetSiren, ProtFlexScoreLandscape]
    _environments = [DESKTOP_TKINTER, WEB_DJANGO]

    def __init__(self, **kwargs):
        ProtocolViewer.__init__(self, **kwargs)
        self._data = None

    def getData(self):
        if self._data is None:
            self._data = self.loadData()
        return self._data


    def _defineParams(self, form):
        form.addSection(label='Show conformational landscape')
        form.addParam('mode', params.EnumParam, choices=['UMAP', 'PCA'],
                      default=0, display=params.EnumParam.DISPLAY_HLIST,
                      label="Dimensionality reduction method",
                      help="\t * UMAP: usually leads to more meaningful spaces, although execution "
                           "is higher\n"
                           "\t * PCA: faster but less meaningfull spaces \n"
                           "UMAP and PCA are only computed the first time the are used. Afterwards, they "
                           "will be reused to increase performance")
        form.addParam('nb_umap', params.IntParam, label="UMAP neighbors",
                      default=15, condition="mode==0",
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
        form.addParam('threads', params.IntParam, label="Number of threads",
                      default=4, condition="mode==0")
        form.addParam('doShowSpace', params.LabelParam,
                      label="Display the conformational space")

    def _getVisualizeDict(self):
        # self.protocol._createFilenameTemplates()
        return {'doShowSpace': self._doShowSpace}

    def _doShowSpace(self, param=None):
        particles = self.protocol.outputParticles
        file_z_space = self.protocol._getExtraPath("z_space.txt")
        file_interp = self.protocol._getExtraPath("interp_values.txt")
        mode = self.mode.get()

        if mode == 0:
            file_coords = self.protocol._getExtraPath("umap_coords.txt")
            mode = [mode, self.nb_umap.get(), self.epochs_umap.get(), self.threads.get(), self.densmap_umap.get()]
        elif mode == 1:
            file_coords = self.protocol._getExtraPath("pca_coords.txt")
            mode = [mode, ]

        def launchViewerNonBlocking(args):
            (particles, file_z_space, file_interp, file_coords, mode) = args

            z_clnm = []
            for particle in particles.iterItems():
                z_clnm.append(particle.getZFlex())
            z_clnm = np.asarray(z_clnm)
            if z_clnm.shape[1] < 3:
                raise Exception("Visualization of spaces with dimension smaller than 3 is not yet implemented. Exiting...")

            # Generate files to call command line
            np.savetxt(file_z_space, z_clnm)

            # Compute/Read UMAP or PCA
            if mode[0] == 0:
                if not os.path.isfile(file_coords):
                    args = "--input %s --umap --output %s --n_neighbors %d " \
                           "--n_epochs %d --n_components 3 --thr %d" \
                           % (file_z_space, file_coords, mode[1], mode[2], mode[3])
                    if mode[4]:
                        args += " --densmap"
                    program = os.path.join(const.XMIPP_SCRIPTS, "dimensionality_reduction.py")
                    program = flexutils.Plugin.getProgram(program)
                    command = buildRunCommand(program, args, 1)
                    p = subprocess.Popen(command, shell=True)
                    p.wait()
            elif mode[0] == 1:
                if not os.path.isfile(file_coords):
                    args = "--input %s --pca --output %s --n_components 3" % (file_z_space, file_coords)
                    program = os.path.join(const.XMIPP_SCRIPTS, "dimensionality_reduction.py")
                    program = flexutils.Plugin.getProgram(program)
                    command = buildRunCommand(program, args, 1)
                    p = subprocess.Popen(command, shell=True)
                    p.wait()

            if particles.getFlexInfo().getProgName() == const.ZERNIKE3D:
                deformation = computeNormRows(z_clnm)
            else:
                deformation = np.zeros(z_clnm.shape)

            # Generate files to call command line
            np.savetxt(file_interp, deformation)

            # Run slicer
            args = "--data %s --z_space %s --interp_val %s --onlyView" \
                   % (file_coords, file_z_space, file_interp)
            program = os.path.join(const.VIEWERS, "annotation_3d_tools", "viewer_interactive_3d.py")
            program = flexutils.Plugin.getProgram(program)

            command = buildRunCommand(program, args, 1)
            subprocess.Popen(command, shell=True)

        # Launch with Pathos
        p = Pool()
        # p.restart()
        p.apipe(launchViewerNonBlocking, args=(particles, file_z_space, file_interp, file_coords, mode))
        # p.join()
        # p.close()

        return []
