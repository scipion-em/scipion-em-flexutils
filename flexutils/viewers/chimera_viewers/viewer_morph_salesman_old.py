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

import pyworkflow.protocol.params as params
from pyworkflow.viewer import DESKTOP_TKINTER, WEB_DJANGO, ProtocolViewer
from pyworkflow.utils.process import runJob
import pyworkflow.utils as pwutils

from pwem.viewers import ChimeraView
from pwem.objects import SetOfClasses3D

import flexutils
from flexutils.protocols.protocol_annotate_space import ProtFlexAnnotateSpace
from flexutils.protocols.protocol_cluster_space import ProtFlexClusterSpace
from flexutils.utils import getOutputSuffix

import xmipp3


class FlexMorphSalesmanViewer(ProtocolViewer):
    """ Morphing along shortest path defined by the salesman's algorithm """
    _label = 'viewer morph path'
    _targets = [ProtFlexClusterSpace, ProtFlexAnnotateSpace]
    _environments = [DESKTOP_TKINTER, WEB_DJANGO]
    OPEN_FILE = "open %s\n"
    VOXEL_SIZE = "volume #%d voxelSize %s\n"
    VOL_HIDE = "vol #%d hide\n"
    VIEW = "view\n"

    def _defineParams(self, form):
        form.addSection(label='Show deformation')
        form.addParam('doShowMorph', params.LabelParam,
                      label="Display the morphing of the maps along the shortest path")

    def _getVisualizeDict(self):
        return {'doShowMorph': self._doShowMorph}

    def _doShowMorph(self, param=None):
        # Get shortest path
        self.computePath()
        path = self.readPath()

        # Useful parameters
        smprt = self.protocol.reference.get().getSamplingRate()

        scriptFile = self.protocol._getPath('morph_orig_ref_chimera.cxc')
        fhCmd = open(scriptFile, 'w')

        for chid, idx in enumerate(path[0]):
            volFile = os.path.abspath(self.protocol._getExtraPath('path_deformed_%d.mrc' % (idx - 1)))
            fhCmd.write(self.OPEN_FILE % volFile)
            fhCmd.write(self.VOXEL_SIZE % ((chid + 1), str(smprt)))
            fhCmd.write(self.VOL_HIDE % (chid + 1))

        num_vol = len(path[0])
        frames = 20 * num_vol
        step = 1 / frames
        fhCmd.write("volume morph #1-%d frames %d playStep %f \n" % (chid + 1, frames, step))
        fhCmd.write(self.VIEW)
        fhCmd.close()

        view = ChimeraView(scriptFile)
        return [view]

    def computePath(self):
        # Get protocol last output
        last_suffix = int(getOutputSuffix(self.protocol, SetOfClasses3D)) - 1
        if last_suffix == 1:
            last_suffix = ""
        else:
            last_suffix = str(last_suffix)
        output = self.protocol.OUTPUT_PREFIX + last_suffix
        output = getattr(self.protocol, output)

        # Get needed inputs
        reference = self.protocol.reference.get().getFileName()
        mask = self.protocol.mask.get().getFileName()
        if pwutils.getExt(reference) == ".mrc":
            reference += ":mrc"
        if pwutils.getExt(mask) == ".mrc":
            mask += ":mrc"

        # Get Zernike3D coefficients and deformed maps
        for idx, rep in enumerate(output.iterRepresentatives()):
            z_clnm = np.fromstring(rep._xmipp_sphCoefficients.get(), sep=",")

            # Write Zernike3D coefficient
            zernike_file = self.protocol._getExtraPath("z_clnm_vw.txt")
            self.writeZernikeFile(z_clnm, zernike_file)

            # Deform volume
            deformedFile = self.protocol._getExtraPath('path_deformed_%d.mrc' % idx)
            params = '-i %s --mask %s --step 1 --blobr 2 -o %s --clnm %s' % \
                     (reference, mask, deformedFile, zernike_file)
            xmipp3.Plugin.runXmippProgram('xmipp_volume_apply_coefficient_zernike3d', params)

        # Get priors
        num_vol = 1
        volumes = self.protocol.volumes.get()
        if volumes:
            num_vol += volumes.getSize()

        # Run salesman's solver
        program = "python " + os.path.join(os.path.dirname(flexutils.__file__), "viewers", "path_finder_tools", "viewer_salesman_solver.py")
        args = "--coords %s --outpath %s --num_vol %d " \
               % (self.protocol._getExtraPath("saved_selections.txt"), self.protocol._getExtraPath("path.txt"),
                  num_vol)
        runJob(None, program, args)

    def writeZernikeFile(self, coeff, outpath):
        particles = self.protocol.particles.get()
        reference = self.protocol.reference.get()
        l1 = particles.L1.get()
        l2 = particles.L2.get()
        dim = reference.getDim()[0]
        with open(outpath, 'w') as fid:
            fid.write(' '.join(map(str, [l1, l2, 0.5 * dim])) + "\n")
            fid.write(' '.join(map(str, coeff.reshape(-1))) + "\n")

    def readPath(self):
        path = []
        with open(self.protocol._getExtraPath("path.txt")) as f:
            lines = f.readlines()
            for line in lines:
                path.append(np.fromstring(line, dtype=float, sep=','))
        return path