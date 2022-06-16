# **************************************************************************
# *
# * Authors:     David Herreros (dherreros@cnb.csic.es)
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
# import pyworkflow.protocol.constants as cons

from pwem import emlib
from pwem.objects import Volume
from pwem.protocols import ProtReconstruct3D

from xmipp3.convert import writeSetOfImages, imageToRow, coordinateToRow
# from xmipp3.base import isXmippCudaPresent

import flexutils
import flexutils.constants as const


class XmippProtReconstructZART(ProtReconstruct3D):
    """    
    Reconstruct a volume using ZART algorithm from a given SetOfParticles.
    """
    _label = 'reconstruct ZART'
    
    #--------------------------- DEFINE param functions --------------------------------------------   
    def _defineParams(self, form):

        # form.addHidden(params.USE_GPU, params.BooleanParam, default=True,
        #                label="Use GPU for execution",
        #                help="This protocol has both CPU and GPU implementation.\
        #                Select the one you want to use.")
        #
        # form.addHidden(params.GPU_LIST, params.StringParam, default='0',
        #                expertLevel=cons.LEVEL_ADVANCED,
        #                label="Choose GPU IDs",
        #                help="Add a list of GPU devices that can be used")

        form.addSection(label='Input')

        form.addParam('inputParticles', params.PointerParam, pointerClass='SetOfParticles',
                      pointerCondition='hasAlignmentProj',
                      label="Input particles",  
                      help='Select the input images from the project.')
        form.addParam('mask', params.PointerParam, pointerClass='VolumeMask',
                      allowsNull=True,
                      label="Reconstruction mask",
                      help="Mask used to restrict the reconstruction space to increase performance. "
                           "Note that here the mask can be tight, as internally the protocol will process "
                           "it to make it wider")
        form.addParam('ctfCorrected', params.BooleanParam, default=False,
                      label="Are particles CTF corrected?",
                      help="If particles are not CTF corrected, set to 'No' to perform "
                           "a Weiner filter based corerction")
        form.addParam('useZernike', params.BooleanParam, default=False,
                      label="Correct motion blurred artifacts?",
                      help="Correct the conformation of the particles during the reconstruct process "
                           "to reduce motion blurred artifacts and increase resolution. Note that this "
                           "option requires that the particles have a set of Zernike3D coefficients associated. "
                           "Otherwise, the parameter should be set to 'No'")
        form.addParam('niter', params.IntParam, default=13,
                      label="Number of ZART iterations to perform",
                      help="In general, the bigger the number the sharper the volume. We recommend "
                           "to run at least 8 iteration for better results")
        form.addParam('save_iter', params.IntParam, default=1000,
                      label="Save partial reconstruction every # images")

        form.addParallelSection(threads=4, mpi=1)

    #--------------------------- INSERT steps functions --------------------------------------------
    def _insertAllSteps(self):
        self._insertFunctionStep(self.convertInputStep)
        self._insertFunctionStep(self.reconstructStep)
        self._insertFunctionStep(self.createOutputStep)
        
    def reconstructStep(self):
        particlesMd = self._getTmpPath('corrected_particles.xmd')

        params =  ' -i %s' % particlesMd
        params += ' -o final_reconstruction.mrc'
        params += ' --odir %s' % self._getExtraPath()
        params += ' --step 2 --sigma 2 --regularization 0.0001'
        params += ' --niter %d' % self.niter.get()
        params += ' --save_iter %d' % self.save_iter.get()

        if self.useZernike.get():
            params += " --useZernike"

        if self.mask.get():
            mask_zart = self._getTmpPath('mask_zart.vol')
            params += ' --mask %s' % mask_zart

        self.runJob('xmipp_forward_art_zernike3d', params)

        # if self.useGpu.get():
        #     if self.numberOfMpi.get()>1:
        #         self.runJob('xmipp_cuda_reconstruct_fourier', params, numberOfMpi=len((self.gpuList.get()).split(','))+1)
        #     else:
        #         self.runJob('xmipp_cuda_reconstruct_fourier', params)
        # else:
        #     if self.legacy.get():
        #         self.runJob('xmipp_reconstruct_fourier', params)
        #     else:
        #         self.runJob('xmipp_reconstruct_fourier_accel', params)

    #--------------------------- STEPS functions --------------------------------------------
    def convertInputStep(self):
        particlesMd = self._getTmpPath('corrected_particles.xmd')
        imgSet = self.inputParticles.get()

        def zernikeRow(part, partRow, **kwargs):
            imageToRow(part, partRow, emlib.MDL_IMAGE, **kwargs)
            coord = part.getCoordinate()
            if coord is not None:
                coordinateToRow(coord, partRow, copyId=False)
            if part.hasMicId():
                partRow.setValue(emlib.MDL_MICROGRAPH_ID, int(part.getMicId()))
                partRow.setValue(emlib.MDL_MICROGRAPH, str(part.getMicId()))
            if hasattr(part, '_xmipp_sphCoefficients') and self.useZernike.get():
                z_clnm = np.fromstring(part._xmipp_sphCoefficients.get(), sep=",")
                partRow.setValue(emlib.MDL_SPH_COEFFICIENTS, z_clnm.tolist())

        writeSetOfImages(imgSet, particlesMd, zernikeRow)

        # Correct CTF of particles if needed
        if not self.ctfCorrected.get():
            sr = imgSet.getSamplingRate()
            corrected_stk = self._getTmpPath('corrected_particles.stk')
            args = "-i %s -o %s --save_metadata_stack --keep_input_columns --sampling_rate %f --correct_envelope" \
                   % (particlesMd, corrected_stk, sr)
            program = 'xmipp_ctf_correct_wiener2d'
            self.runJob(program, args, numberOfMpi=self.numberOfMpi.get())

        # Mask preprocessing (if provided)
        if self.mask.get():
            mask_file = self.mask.get().getFileName()
            mask_zart_file = self._getTmpPath('mask_zart.vol')
            args = "--input %s --output %s" \
                   % (mask_file, mask_zart_file)
            program = os.path.join(const.XMIPP_SCRIPTS, "flood_fill_mask.py")
            program = flexutils.Plugin.getProgram(program)
            self.runJob(program, args)

    def createOutputStep(self):
        imgSet = self.inputParticles.get()
        volume = Volume()
        volume.setFileName(self._getExtraPath("final_reconstruction.mrc"))
        volume.setSamplingRate(imgSet.getSamplingRate())
        
        self._defineOutputs(outputVolume=volume)
        self._defineSourceRelation(self.inputParticles, volume)
    
    #--------------------------- INFO functions -------------------------------------------- 
    def _summary(self):
        """ Should be overriden in subclasses to 
        return summary message for NORMAL EXECUTION. 
        """
        return []
    
    #--------------------------- UTILS functions --------------------------------------------
