
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


import glob

import numpy as np
import os

from xmipp_metadata.image_handler import ImageHandler

from pyworkflow.object import String, Integer, Float
from pyworkflow import VERSION_2_0
from pyworkflow.utils import removeExt, moveFile, removeBaseExt, getExt
import pyworkflow.protocol.params as params

from pwem.protocols import ProtAnalysis3D, ProtFlexBase
from pwem.objects import SetOfVolumes, Volume, VolumeFlex

import xmipp3

import flexutils.constants as const
from flexutils.utils import readZernikeFile, getXmippFileName
import flexutils


class XmippProtComputeHeterogeneityPriorsZernike3D(ProtAnalysis3D, ProtFlexBase):
    """ Compute Zernike3D priors and assign them to a SetOfVolumes """
    _label = 'compute heterogeneity priors - Zernike3D'
    _lastUpdateVersion = VERSION_2_0
    OUTPUT_SUFFIX = '_%d_crop.mrc'
    ALIGNED_VOL = 'vol%dAligned.mrc'
    OUTPUT_PREFIX = "zernikeVolumes"

    def __init__(self, **args):
        ProtAnalysis3D.__init__(self, **args)
        self.stepsExecutionMode = params.STEPS_PARALLEL

    # --------------------------- DEFINE param functions --------------------------------------------
    def _defineParams(self, form):
        form.addSection(label='Input')
        form.addParam('inputVolumes', params.MultiPointerParam,
                      pointerClass='SetOfVolumes,Volume',
                      label="Input volume(s)", important=True,
                      help='Volumes used to compute the Zernike3D priors')
        form.addParam('reference', params.PointerParam,
                      pointerClass='Volume',
                      label="Zernike3D reference map", important=True,
                      help='Priors computed will be refered to this reference map')
        form.addParam('mask', params.PointerParam,
                      pointerClass='VolumeMask',
                      label="Reference map mask",
                      help='Mask determining where to compute the deformation field in the '
                           'reference volume. The tightest the mask, the higher the '
                           'performance boost')
        form.addParam('boxSize', params.IntParam, default=128,
                      label='Downsample particles to this box size', expertLevel=params.LEVEL_ADVANCED,
                      help='In general, downsampling the volumes will increase performance without compromising '
                           'the estimation the deformation field for each particle. Note that output particles will '
                           'have the original box size, and Zernike3D coefficients will be modified to work with the '
                           'original size volumes')
        form.addParam('l1', params.IntParam, default=3,
                      label='Zernike Degree',
                      expertLevel=params.LEVEL_ADVANCED,
                      help='Degree Zernike Polynomials of the deformation=1,2,3,...')
        form.addParam('l2', params.IntParam, default=2,
                      label='Harmonical Degree',
                      expertLevel=params.LEVEL_ADVANCED,
                      help='Degree Spherical Harmonics of the deformation=1,2,3,...')

    # --------------------------- INSERT steps functions --------------------------------------------
    def _insertAllSteps(self):
        self._insertFunctionStep(self.convertInputStep)
        self._insertFunctionStep(self.computePriorsStep)
        self._insertFunctionStep(self.createOutputStep)

    # --------------------------- STEPS functions ---------------------------------------------------
    def convertInputStep(self):
        reference_file = self.reference.get().getFileName()
        dim = self.reference.get().getDim()[0]
        boxSize = self.boxSize.get()

        # Get input files
        input_files = []
        for pointer in self.inputVolumes:
            obj = pointer.get()
            if isinstance(obj, Volume):
                input_files.append(obj.getFileName())
            elif isinstance(obj, SetOfVolumes):
                for input_map in obj.iterItems():
                    input_files.append(input_map.getFileName())

        # Rigid fitting of maps to reference
        for input_file in input_files:
            output_file = self._getExtraPath(removeBaseExt(input_file) + "_aligned.mrc")
            args = "--i %s --r %s --o %s" % (input_file, reference_file, output_file)
            program = os.path.join(const.XMIPP_SCRIPTS, "align_maps.py")
            program = flexutils.Plugin.getProgram(program)
            self.runJob(program, args, numberOfMpi=1)

            # Resize inputs if needed
            if dim != boxSize:
                self.runJob("xmipp_image_resize",
                            "-i %s -o %s --dim %d "
                            % (output_file + ":mrc", output_file, boxSize), numberOfMpi=1,
                            env=xmipp3.Plugin.getEnviron())

        input_mask = getXmippFileName(self.mask.get().getFileName())
        out_mask = self._getExtraPath("ref_mask.mrc")
        out_reference = self._getExtraPath("ref.mrc")
        ih = ImageHandler()
        ih.convert(input_mask, out_mask)
        ih.convert(getXmippFileName(reference_file), out_reference)
        if dim != boxSize:
            self.runJob("xmipp_image_resize",
                        "-i %s --dim %d --interp nearest" % (out_mask, boxSize), numberOfMpi=1,
                        env=xmipp3.Plugin.getEnviron())
            self.runJob("xmipp_image_resize",
                        "-i %s --dim %d" % (out_reference, boxSize), numberOfMpi=1,
                        env=xmipp3.Plugin.getEnviron())

    def computePriorsStep(self):
        reference_file = self._getExtraPath("ref.mrc")
        input_files = glob.glob(self._getExtraPath("*_aligned.mrc"))
        L1 = self.l1.get()
        L2 = self.l2.get()

        for idf, input_file in enumerate(input_files):
            output_file = self._getExtraPath("deformed_%d.mrc" % (idf + 1))
            args = "--i %s --r %s --o %s --l1 %d --l2 %d" \
                   % (reference_file, input_file, output_file, L1, L2)
            program = os.path.join(const.XMIPP_SCRIPTS, "find_z_clnm_map.py")
            program = flexutils.Plugin.getProgram(program)
            self.runJob(program, args, numberOfMpi=1)

            z_clnm_file = self._getExtraPath("z_clnm.txt")
            rmsd_def_file = self._getExtraPath("rmsd_def.txt")

            moveFile(z_clnm_file, removeExt(z_clnm_file) + "_%d.txt" % (idf + 1))
            moveFile(rmsd_def_file, removeExt(rmsd_def_file) + "_%d.txt" % (idf + 1))

    def createOutputStep(self):
        reference = self.reference.get()
        mask = self.mask.get()
        dim = reference.getDim()[0]
        sr = reference.getSamplingRate()
        L1 = Integer(self.l1.get())
        L2 = Integer(self.l2.get())
        Rmax = Integer(0.5 * reference.getDim()[0])
        reference_filename = String(reference.getFileName())
        mask_filename = String(mask.getFileName()) if mask else String("")

        zernikeVols = self._createSetOfVolumesFlex(progName=const.ZERNIKE3D)
        zernikeVols.setSamplingRate(sr)
        zernikeVols.getFlexInfo().L1 = L1
        zernikeVols.getFlexInfo().L2 = L2
        zernikeVols.getFlexInfo().Rmax = Rmax
        zernikeVols.getFlexInfo().refMap = reference_filename
        zernikeVols.getFlexInfo().refMask = mask_filename

        input_files = len(glob.glob(self._getExtraPath("*_aligned.mrc")))
        for idf in range(input_files):
            z_clnm_file = self._getExtraPath("z_clnm_%d.txt" % (idf + 1))
            rmsd_def_file = self._getExtraPath("rmsd_def_%d.txt" % (idf + 1))

            basis_params, z_clnm = readZernikeFile(z_clnm_file)
            rmsd_def = np.loadtxt(rmsd_def_file)

            deformation = Float(rmsd_def)
            factor = dim / self.boxSize.get()
            z_clnm = factor * z_clnm
            # Rmax = Float(factor * basis_params[2])

            zernikeVol = VolumeFlex(progName=const.ZERNIKE3D)
            zernikeVol.setFileName(reference.getFileName())
            zernikeVol.setSamplingRate(sr)
            zernikeVol.setZFlex(z_clnm.reshape(-1))
            zernikeVol.getFlexInfo().deformation = deformation
            zernikeVol.getFlexInfo().L1 = L1
            zernikeVol.getFlexInfo().L2 = L2
            zernikeVol.getFlexInfo().Rmax = Rmax
            zernikeVol.getFlexInfo().refMap = reference_filename
            zernikeVol.getFlexInfo().refMask = mask_filename

            zernikeVols.append(zernikeVol)

        # Save new output
        name = self.OUTPUT_PREFIX
        args = {}
        args[name] = zernikeVols
        self._defineOutputs(**args)
        self._defineSourceRelation(reference, zernikeVols)

    # ------------------------- VALIDATE functions -----------------------------
    def validate(self):
        """ Try to find errors on define params. """
        errors = []
        l1 = self.l1.get()
        l2 = self.l2.get()
        if (l1 - l2) < 0:
            errors.append('Zernike degree must be higher than '
                          'SPH degree.')
        return errors




