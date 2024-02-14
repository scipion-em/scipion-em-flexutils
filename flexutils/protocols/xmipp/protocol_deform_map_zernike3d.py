
# **************************************************************************
# *
# * Authors:     Amaya Jimenez Moreno (ajimenez@cnb.csic.es)
# *              David Herreros Calero (dherreros@cnb.csic.es)
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

from pwem.protocols import ProtAnalysis3D
import pyworkflow.protocol.params as params
from pwem.objects import Volume
from pyworkflow import VERSION_2_0
from pyworkflow.object import Integer, String

import xmipp3

from pwem.objects import VolumeFlex

from flexutils.utils import readZernikeFile
import flexutils.constants as const
import flexutils


class ProtFlexVolumeDeformZernike3D(ProtAnalysis3D):
    """ Protocol for volume deformation based on Zernike3D. """
    _label = 'volume deform - Zernike3D'
    _lastUpdateVersion = VERSION_2_0

    # --------------------------- DEFINE param functions --------------------------------------------
    def _defineParams(self, form):
        form.addSection(label='Input')
        form.addHidden(params.USE_GPU, params.BooleanParam, default=True,
                       label="Use GPU for execution",
                       help="This protocol has both CPU and GPU implementation.\
                           Select the one you want to use.")
        form.addHidden(params.GPU_LIST, params.StringParam, default='0',
                       expertLevel=params.LEVEL_ADVANCED,
                       label="Choose GPU IDs",
                       help="Add a list of GPU devices that can be used")
        form.addParam('inputVolume', params.PointerParam, label="Input volume",
                      pointerClass='Volume',
                      help="Volume to be deformed")
        form.addParam('refVolume', params.PointerParam, label="Reference volume",
                      pointerClass='Volume',
                      help="Target volume input volume will be deformed to")
        form.addParam('sigma', params.NumericListParam, label="Multiresolution", default="1 2",
                      help="Perform the analysys comparing different filtered versions of the volumes")
        form.addParam('boxSize', params.IntParam, label="Downsample maps to this boxsize",
                      default=128,
                      help="Internally downsample the input and reference maps to speed up computations. "
                           "Downsampling is only applied internally, resulting volumes will be resampled to "
                           "the original box size.")
        form.addParam('l1', params.IntParam, default=3,
                      label='Zernike Degree',
                      expertLevel=params.LEVEL_ADVANCED,
                      help='Degree Zernike Polynomials of the deformation=1,2,3,...')
        form.addParam('l2', params.IntParam, default=2,
                      label='Harmonical Degree',
                      expertLevel=params.LEVEL_ADVANCED,
                      help='Degree Spherical Harmonics of the deformation=1,2,3,...')
        form.addParam('penalization', params.FloatParam, default=0.00025, label='Regularization',
                      expertLevel=params.LEVEL_ADVANCED,
                      help='Penalization to deformations (higher values penalize more the deformation).')
        form.addParallelSection(threads=4, mpi=0)


    def _createFilenameTemplates(self):
        """ Centralize how files are called """
        myDict = {
            'fnRefVol': self._getExtraPath('ref_volume.mrc'),
            'fnInputVol': self._getExtraPath('input_volume.mrc'),
            'fnInputFilt': self._getExtraPath('input_volume_filt.mrc'),
            'fnRefFilt': self._getExtraPath('ref_volume_filt.mrc'),
            'fnOutVol': self._getExtraPath('vol1DeformedTo2.mrc')
                 }
        self._updateFilenamesDict(myDict)

    # --------------------------- INSERT steps functions -----------------------
    def _insertAllSteps(self):
        self._createFilenameTemplates()
        self._insertFunctionStep(self.convertInputStep)
        self._insertFunctionStep(self.deformStep)
        self._insertFunctionStep(self.convertOutputStep)
        self._insertFunctionStep(self.createOutputStep)

    # --------------------------- STEPS functions ------------------------------
    def convertInputStep(self):
        fnInputVol = self._getFileName('fnInputVol')
        fnRefVol = self._getFileName('fnRefVol')

        XdimI = self.inputVolume.get().getDim()[0]
        XdimR = self.refVolume.get().getDim()[0]
        newXdim = self.boxSize.get()

        ih = ImageHandler()
        ih.convert(self.inputVolume.get().getFileName(), fnInputVol)
        if XdimI != newXdim:
            ImageHandler().scaleSplines(inputFn=fnInputVol, outputFn=fnInputVol, finalDimension=newXdim,
                                        overwrite=True)


        ih.convert(self.refVolume.get().getFileName(), fnRefVol)
        if XdimR != newXdim:
            ImageHandler().scaleSplines(inputFn=fnRefVol, outputFn=fnRefVol, finalDimension=newXdim,
                                        overwrite=True)


    def deformStep(self):
        fnRefVol = self._getFileName('fnRefVol')
        fnOutVol = self._getFileName('fnOutVol')
        fnRefMask = self._getExtraPath("ref_mask.mrc")
        fnOutMask = self._getExtraPath("ref_mask.mrc")
        fnZclnm = self._getExtraPath('Volumes_clnm.txt')

        self.alignMaps()

        mask_input = ImageHandler(fnOutVol).generateMask(iterations=50, boxsize=64, smoothStairEdges=False)
        ImageHandler().write(mask_input, fnOutMask, overwrite=True)
        mask_reference = ImageHandler(fnOutVol).generateMask(iterations=50, boxsize=64, smoothStairEdges=False)
        ImageHandler().write(mask_reference, fnRefMask, overwrite=True)

        params = ' -i %s -r %s -o %s --analyzeStrain --l1 %d --l2 %d --sigma "%s" --oroot %s --regularization %f' % \
                 (fnOutVol, fnRefVol, fnOutVol, self.l1.get(), self.l2.get(), self.sigma.get(),
                  self._getExtraPath('Volumes'), self.penalization.get())

        if self.useGpu.get():
            self.runJob("xmipp_cuda_volume_deform_sph", params, env=xmipp3.Plugin.getEnviron())
        else:
            if self.numberOfThreads.get() != 0:
                params = params + ' --thr %d' % self.numberOfThreads.get()
            self.runJob("xmipp_volume_deform_sph", params, env=xmipp3.Plugin.getEnviron())

        # Invert deformation field
        args = "--i %s --r %s --z_clnm %s --o %s" % \
               (fnRefMask, fnOutMask, fnZclnm, fnZclnm)
        program = os.path.join(const.XMIPP_SCRIPTS, "invert_zernike_field.py")
        program = flexutils.Plugin.getProgram(program)
        self.runJob(program, args)

    def convertOutputStep(self):
        inputVolume = self.inputVolume.get()
        xDim = inputVolume.getXDim()
        sr_i = inputVolume.getSamplingRate()
        newXDim = self.boxSize.get()
        fnOutVol = self._getFileName('fnOutVol')
        if newXDim != xDim:
            ImageHandler().scaleSplines(inputFn=fnOutVol, outputFn=fnOutVol,
                                        finalDimension=xDim,
                                        overwrite=True)
        ImageHandler().setSamplingRate(fnOutVol, sr_i)

    def createOutputStep(self):
        inputVolume = self.inputVolume.get()
        xDim = inputVolume.getXDim()
        newXDim = self.boxSize.get()
        sr_i = inputVolume.getSamplingRate()
        basis_params, z_clnm = readZernikeFile(self._getExtraPath('Volumes_clnm.txt'))
        z_clnm *= xDim / newXDim

        # Create input map mask
        zernike_mask = self._getExtraPath("zernike_mask.mrc")
        mask_input = ImageHandler(inputVolume.getFileName()).generateMask(iterations=50, boxsize=64, smoothStairEdges=False)
        ImageHandler().write(mask_input, zernike_mask, overwrite=True)

        # Deformed volume
        out_vol = Volume()
        out_vol.setLocation(self._getFileName('fnOutVol'))
        out_vol.setSamplingRate(sr_i)

        # Flex volume
        vol_flex = VolumeFlex(progName=const.ZERNIKE3D)
        vol_flex.setSamplingRate(sr_i)
        vol_flex.setFileName(inputVolume.getFileName())
        vol_flex.getFlexInfo().L1 = Integer(self.l1.get())
        vol_flex.getFlexInfo().L2 = Integer(self.l2.get())
        vol_flex.getFlexInfo().Rmax = Integer(0.5 * xDim)
        vol_flex.getFlexInfo().refMap = String(inputVolume.getFileName())
        vol_flex.getFlexInfo().refMask = String(zernike_mask)
        vol_flex.setZFlex(z_clnm[0])

        self._defineOutputs(deformed=out_vol, flexVolume=vol_flex)
        self._defineSourceRelation(self.inputVolume, out_vol)
        self._defineSourceRelation(self.inputVolume, vol_flex)

    def alignMaps(self):
        fnInputVol = self._getFileName('fnInputVol')
        fnInputFilt = self._getFileName('fnInputFilt')
        fnRefVol = self._getFileName('fnRefVol')
        fnRefFilt = self._getFileName('fnRefFilt')
        fnOutVol = self._getFileName('fnOutVol')

        # Filter the volumes to improve alignment quality
        params = " -i %s -o %s --fourier real_gaussian 2" % (fnInputVol, fnInputFilt)
        self.runJob("xmipp_transform_filter", params, env=xmipp3.Plugin.getEnviron())
        params = " -i %s -o %s --fourier real_gaussian 2" % (fnRefVol, fnRefFilt)
        self.runJob("xmipp_transform_filter", params, env=xmipp3.Plugin.getEnviron())

        # Find transformation needed to align the volumes
        params = ' --i1 %s --i2 %s --local --dontScale ' \
                 '--copyGeo %s' % \
                 (fnRefFilt, fnInputFilt, self._getExtraPath("geo.txt"))
        self.runJob("xmipp_volume_align", params, env=xmipp3.Plugin.getEnviron())

        # Apply transformation of filtered volume to original volume
        with open(self._getExtraPath("geo.txt"), 'r') as file:
            geo_str = file.read().replace('\n', ',')
        params = " -i %s -o %s --matrix %s" % (fnInputVol, fnOutVol, geo_str)
        self.runJob("xmipp_transform_geometry", params, env=xmipp3.Plugin.getEnviron())

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