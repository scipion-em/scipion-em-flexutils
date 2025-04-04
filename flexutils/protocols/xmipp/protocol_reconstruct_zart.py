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
import glob
import numpy as np
from scipy import stats
from scipy import signal
import re
from xmipp_metadata.image_handler import ImageHandler

import pyworkflow.protocol.params as params
import pyworkflow.utils as pwutils
# import pyworkflow.protocol.constants as cons

from pwem import emlib
from pwem.objects import Volume
from pwem.protocols import ProtReconstruct3D
from pwem.objects import ParticleFlex, SetOfParticlesFlex

from xmipp3.convert import writeSetOfImages, imageToRow, coordinateToRow
# from xmipp3.base import isXmippCudaPresent
import xmipp3

import flexutils
import flexutils.constants as const
from flexutils.utils import getXmippFileName


def filterVol(volume, mode="bspline"):
    size = volume.shape[1]

    # Create kernel
    if mode == "bspline":
        # Create a bspline kernel that will be used to blur the original acquisition
        b_spline_1d = np.asarray([0.0, 0.5, 1.0, 0.5, 0.0])
        pad_before = (size - len(b_spline_1d)) // 2
        pad_after = size - pad_before - len(b_spline_1d)
        kernel = np.einsum('i,j,k->ijk', b_spline_1d, b_spline_1d, b_spline_1d)
        kernel = np.pad(kernel, (pad_before, pad_after), 'constant', constant_values=(0.0,))
        ft_kernel = np.abs(np.fft.fftshift(np.fft.fftn(kernel)))
    elif mode == "gaussian":
        # Create a gaussian kernel that will be used to blur the original acquisition
        std = 2.0
        gauss_1d = signal.windows.gaussian(volume.shape[1], std)
        kernel = np.einsum('i,j,k->ijk', gauss_1d, gauss_1d, gauss_1d)
    ft_kernel = np.abs(np.fft.fftshift(np.fft.fftn(kernel)))

    def applyKernelFourier(x):
        ft_x = np.fft.fftshift(np.fft.fftn(x))
        ft_x_real = ft_x.real * ft_kernel
        ft_x_imag = ft_x.imag * ft_kernel
        ft_x = ft_x_real + ft_x_imag * 1j
        return np.fft.ifftn(np.fft.fftshift(ft_x)).real

    volume = applyKernelFourier(volume)
    thr = 1e-6
    volume = volume - (volume > thr) * thr + (volume < -thr) * thr - (volume == thr) * volume

    return volume


class XmippProtReconstructZART(ProtReconstruct3D):
    """    
    Reconstruct a volume using ZART algorithm from a given SetOfParticles.
    """
    _label = 'reconstruct ZART'

    def __init__(self, **args):
        ProtReconstruct3D.__init__(self, **args)
        self.stepsExecutionMode = params.STEPS_PARALLEL

    #--------------------------- DEFINE param functions --------------------------------------------   
    def _defineParams(self, form):
        form.addSection(label='Input')

        form.addHidden(params.GPU_LIST, params.StringParam, default='0',
                       expertLevel=params.LEVEL_ADVANCED,
                       label="Choose GPU IDs",
                       help="Add a list of GPU devices that can be used")

        group = form.addGroup("Data")
        group.addParam('inputParticles', params.PointerParam, pointerClass='SetOfParticles, SetOfParticlesFlex',
                      pointerCondition='hasAlignmentProj',
                      label="Input particles",  
                      help='Select the input images from the project.')
        group.addParam('initialMap', params.PointerParam, pointerClass='Volume',
                      label="Initial map",
                      allowsNull=True,
                      help='If provided, this map will be used as the initialization of the reconstruction '
                           'process. Otherwise, an empty volume will be used')
        group.addParam('recMask', params.PointerParam, pointerClass='VolumeMask',
                      allowsNull=True,
                      label="Reconstruction mask",
                      help="Mask used to restrict the reconstruction space to increase performance.")
        group = form.addGroup("CTF")
        group.addParam('ctfCorrected', params.BooleanParam, default=False,
                      label="Are particles CTF corrected?",
                      help="If particles are not CTF corrected, set to 'No' to perform "
                           "a Weiner filter based correction")
        group = form.addGroup("Motion correction")
        group.addParam('useZernike', params.BooleanParam, default=False,
                      condition="inputParticles and isinstance(inputParticles, SetOfParticlesFlex)",
                      label="Correct motion blurred artifacts?",
                      help="Correct the conformation of the particles during the reconstruct process "
                           "to reduce motion blurred artifacts and increase resolution. Note that this "
                           "option requires that the particles have a set of Zernike3D coefficients associated. "
                           "Otherwise, the parameter should be set to 'No'")
        group = form.addGroup("Symmetry")
        group.addParam('symmetryGroup', params.StringParam, default="c1",
                      label='Symmetry group',
                      help='If no symmetry is present, give c1')
        group = form.addGroup("Reconstruction modes")
        group.addParam('mode', params.EnumParam, choices=['Reconstruct', 'Gold standard', 'Multiresolution'],
                      default=0, display=params.EnumParam.DISPLAY_HLIST,
                      label="Reconstruction mode",
                      help="\t * Reconstruct: usual reconstruction of a single volume using all the images "
                           "in the dataset\n"
                           "\t * Gold standard: volumes halves reconstruction for FSC and local resolution "
                           "computations\n"
                           "\t * Multiresolution: local resolution analysis during reconstruction to determine "
                           "which areas of the map can be improved further with the ZART algorithm")
        group.addParam('levels', params.IntParam, default=3, condition="mode==2",
                      label="Number of multiresolution levels")

        form.addSection(label='Optimization')
        group = form.addGroup("Preparation")
        group.addParam('niter', params.IntParam, default=2,
                      label="Number of ZART iterations to perform",
                      help="In general, the bigger the number the sharper the volume. We recommend "
                           "to run at least 8 iteration for better results")
        group.addParam('sorting', params.EnumParam, choices=["Orthogonal", "Random"], default=0,
                       display=params.EnumParam.DISPLAY_HLIST,
                       label="How to sort particles?",
                       help="Orthogonal: Particles will be sorted in an orthogonal way based on their angular "
                            "information. Sorting the particles orthogonally usually leads to a faster convergence, "
                            "but it might become slow for large datasets."
                            "Random: Particles are randomly sorted. This is very fast compared to orthogonal sorting.")
        group.addParam('reg', params.FloatParam, default=1e-4,
                      label='ART lambda',
                      help="This parameter determines how fast ZART will converge to the reconstruction. "
                           "Note that larger values may lead to divergence.")
        group = form.addGroup("Regularizations: Denoising")
        group.addParam('lst', params.FloatParam, default=0.0001,
                     label="Positive L1 regularization",
                     expertLevel=params.LEVEL_ADVANCED,
                     help="L1 based penalization on the positive values of the reconstructed volumes. Larger values will "
                          "have a stronger denoising effect on the reconstructed map.")
        group.addParam('ltv', params.FloatParam, default=0.0001,
                      label="Total variation regularization",
                      expertLevel=params.LEVEL_ADVANCED,
                      help="Total variation based regularization on the edges of the reconstructed volumes. Larger "
                           "values will lead to a more enhance representation of the edges and reduction of noise.")
        group.addParam('ltk', params.FloatParam, default=0.0001,
                      expertLevel=params.LEVEL_ADVANCED,
                      label="Tikhonov regularization",
                      help=" Tikhonov based regularization on the edges of the reconstructed volume. Larger values will "
                           "lead softer density transitions in the reconstruction")
        group = form.addGroup("Regularizations: Negative value mitigation")
        group.addParam('ll1', params.FloatParam, default=0.0001,
                      label="Negative L1 regularization",
                      expertLevel=params.LEVEL_ADVANCED,
                      help="L1 based penalization on the negative values of the reconstructed volumes. Larger values "
                           "will decrease more strongly the presence of negative values on the reconstructed map.")

        form.addSection(label='Output')
        group = form.addGroup("Intermediate results")
        group.addParam('save_pr', params.BooleanParam, default=False, expertLevel=params.LEVEL_ADVANCED,
                      label="Save partial reconstructions for every ZART iteration?")
        group = form.addGroup("Output volume appereance")
        group.addParam('onlyPositive', params.BooleanParam, default=False, expertLevel=params.LEVEL_ADVANCED,
                      label="Remove negative values from reconstructed volume?")

        form.addParallelSection(threads=4, mpi=1)

    #--------------------------- INSERT steps functions --------------------------------------------
    def _insertAllSteps(self):
        depsConvert = []
        depsReconstruct = []
        convert = self._insertFunctionStep(self.convertInputStep, prerequisites=[])
        depsConvert.append(convert)
        refFile = self.initialMap.get().getFileName() if self.initialMap.get() else None
        recMask = self.recMask.get().getFileName() if self.recMask.get() else None
        if self.mode.get() == 0:
            particlesMd = self._getTmpPath('corrected_particles.xmd')
            reconstruct = self._insertFunctionStep(self.reconstructStep, particlesMd,
                                                   "final_reconstruction.mrc", refFile, 2,
                                                   self.niter.get(), recMask,
                                                   prerequisites=depsConvert)
            depsReconstruct.append(reconstruct)
        elif self.mode.get() == 1:
            particlesHalvesMd = [self._getTmpPath('corrected_particles_half_000001.xmd'),
                                 self._getTmpPath('corrected_particles_half_000002.xmd')]
            for idx, fileMd in enumerate(particlesHalvesMd):
                outFile = "final_reconstruction_%d.mrc" % (idx + 1)
                reconstruct = self._insertFunctionStep(self.reconstructStep, fileMd, outFile, refFile,
                                                       2, self.niter.get(), recMask,
                                                       prerequisites=depsConvert)
                depsReconstruct.append(reconstruct)
        elif self.mode.get() == 2:
            depsMask = []
            current_iter = 0
            niter = self.niter.get()
            level = 1
            particlesHalvesMd = [self._getTmpPath('corrected_particles_half_000001.xmd'),
                                 self._getTmpPath('corrected_particles_half_000002.xmd')]

            for idx, fileMd in enumerate(particlesHalvesMd):
                outFile = "final_reconstruction_%d_level_%d.mrc" % (idx + 1, level)
                reconstruct = self._insertFunctionStep(self.reconstructStep, fileMd, outFile, refFile,
                                                       2, 2, recMask,
                                                       prerequisites=depsConvert)
                depsReconstruct.append(reconstruct)

            resMask = self._insertFunctionStep(self.resolutionMaskStep, level,
                                               prerequisites=depsReconstruct)
            depsMask.append(resMask)

            resMaskFile = self._getTmpPath("resMask.mrc")
            while current_iter < niter:
                depsReconstruct = []
                level += 1
                current_iter += 2
                for idx, fileMd in enumerate(particlesHalvesMd):
                    outFile = "final_reconstruction_%d_level_%d.mrc" % (idx + 1, level)
                    refFile = self._getExtraPath("final_reconstruction_%d_level_%d.mrc" % (idx + 1, level - 1))
                    reconstruct = self._insertFunctionStep(self.reconstructStep, fileMd, outFile, refFile,
                                                           1, 2, resMaskFile,
                                                           prerequisites=depsMask)
                    depsReconstruct.append(reconstruct)

                depsMask = []
                resMask = self._insertFunctionStep(self.resolutionMaskStep, level,
                                                   prerequisites=depsReconstruct)
                depsMask.append(resMask)

        self._insertFunctionStep(self.createOutputStep, prerequisites=depsReconstruct)

        if not self.save_pr.get():
            pwutils.cleanPattern(self._getExtraPath("*_iter_*.mrc"))
            pwutils.cleanPattern(self._getExtraPath("meanMap.mrc"))
            pwutils.cleanPattern(self._getExtraPath("monoresResolutionChimera.mrc"))
            pwutils.cleanPattern(self._getExtraPath("monoresResolutionMap.mrc"))
            pwutils.cleanPattern(self._getExtraPath("refinedMask.mrc"))
        
    def reconstructStep(self, inputMd, outFile, refFile, step, niter, mask):
        params = self.defineZARTArgs(inputMd, outFile, niter, step, mask)

        if refFile:
            params += " --ref %s" % getXmippFileName(refFile)

        params += " --debug_iter"
        env = xmipp3.Plugin.getEnviron()
        env["CUDA_VISIBLE_DEVICES"] = ','.join([str(elem) for elem in self.getGpuList()])
        self.runJob('xmipp_cuda_forward_art_zernike3d', params, env=env)

    def resolutionMaskStep(self, level):
        half_1 = self._getExtraPath("final_reconstruction_1_level_%d.mrc" % level)
        half_2 = self._getExtraPath("final_reconstruction_2_level_%d.mrc" % level)
        sr = self.inputParticles.get().getSamplingRate()
        low = 2*sr
        high = 2*sr*(self.levels.get() + 1)

        params = "--vol %s --vol2 %s --sampling_rate %f --noiseonlyinhalves " \
                 "--minRes %f --maxRes %f --step 0.5 -o %s --significance 0.95 --threads %d" % \
                 (half_1, half_2, sr, low, high, self._getExtraPath(), self.numberOfThreads.get())

        # if mask:
        #     params += ' --mask %s' % mask
        # else:
        dim = self.inputParticles.get().getXDim()
        mask = self._getTmpPath("sphere_mask.mrc")
        r = int(0.5 * dim)
        ImageHandler().write(np.zeros([dim, dim, dim]), mask, overwrite=True)
        mask_params = "-i %s --mask circular -%d --create_mask %s" % (mask, r, mask)
        self.runJob('xmipp_transform_mask', mask_params, numberOfMpi=1, env=xmipp3.Plugin.getEnviron())
        params += ' --mask %s' % mask

        self.runJob('xmipp_resolution_monogenic_signal', params, numberOfMpi=1, env=xmipp3.Plugin.getEnviron())

        resMap = ImageHandler().read(self._getExtraPath("monoresResolutionMap.mrc")).getData()
        oriMask = ImageHandler().read(mask).getData()

        # _, edges = np.histogram(resMap, bins=3, range=[2*sr, 12.0])

        levels = self.levels.get()
        values = resMap.flatten()
        edges = stats.mstats.mquantiles(values[values != 0.0], np.linspace(0.0, 1.0, num=levels+1))
        mid_points = 0.5 * (edges[1:] + edges[:-1])
        sigmas = mid_points / (2*sr)
        # sigmas *= np.sqrt(1 / (2 * np.log(2)))
        # sigmas *= np.sqrt(np.log(2) / (2 * np.pi * np.pi))
        sigmas = np.floor(sigmas).astype(int)
        # sigmas[sigmas > 1] -= 1
        print(mid_points)
        print(sigmas)

        resMask = np.zeros(resMap.shape)
        # sigmas = [1, 2, 3]
        for idx in range(0, edges.size - 1):
            sigma = sigmas[idx]
            mask = ((resMap >= edges[idx]) * (resMap < edges[idx + 1])).astype(int)
            aux = np.zeros(resMap.shape)
            aux[::sigma, ::sigma, ::sigma] = mask[::sigma, ::sigma, ::sigma]
            mask = sigma * aux
            # mask *= sigma
            resMask += mask
        resMask *= oriMask

        ImageHandler().write(resMask, self._getTmpPath("resMask.mrc"), overwrite=True)

        self.sigmas = ' '.join(map(str, np.unique(sigmas)))
        print(self.sigmas)

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
            if isinstance(part, ParticleFlex) and self.useZernike.get():
                z_clnm = part.getZFlex()
                partRow.setValue(emlib.MDL_SPH_COEFFICIENTS, z_clnm.tolist())

        writeSetOfImages(imgSet, particlesMd, zernikeRow)

        # Correct CTF of particles if needed
        if not self.ctfCorrected.get():
            sr = imgSet.getSamplingRate()
            corrected_stk = self._getTmpPath('corrected_particles.stk')
            args = "-i %s -o %s --save_metadata_stack --keep_input_columns --sampling_rate %f --correct_envelope" \
                   % (particlesMd, corrected_stk, sr)
            program = 'xmipp_ctf_correct_wiener2d'
            self.runJob(program, args, numberOfMpi=self.numberOfThreads.get(), env=xmipp3.Plugin.getEnviron())

        # Mask preprocessing (if provided)
        # if self.mask.get():
        #     mask_file = self.mask.get().getFileName()
        #     mask_zart_file = self._getTmpPath('mask_zart.vol')
        #     args = "--input %s --output %s" \
        #            % (mask_file, mask_zart_file)
        #     program = os.path.join(const.XMIPP_SCRIPTS, "flood_fill_mask.py")
        #     program = flexutils.Plugin.getProgram(program)
        #     self.runJob(program, args, numberOfMpi=1)

        # Prepare data (reconstruction mode)
        if self.mode.get() != 0:
            halvesMd = self._getTmpPath("corrected_particles_half_")
            self.runJob("xmipp_metadata_split",
                        "-i %s --oroot %s" % (particlesMd, halvesMd), env=xmipp3.Plugin.getEnviron())

    def createOutputStep(self):
        imgSet = self.inputParticles.get()
        volume, volume_filtered = Volume(), Volume()
        volume.setFileName(self._getExtraPath("final_reconstruction.mrc"))
        volume_filtered.setFileName(self._getExtraPath("final_reconstruction_filtered.mrc"))
        volume.setSamplingRate(imgSet.getSamplingRate())
        volume_filtered.setSamplingRate(imgSet.getSamplingRate())

        if self.mode.get() != 0:
            halves = self.volumeRestoration()
            volume.setHalfMaps(halves)

            halves_filtered = [f"final_reconstruction_filtered_half_1.mrc", f"final_reconstruction_filtered_half_2.mrc"]
            for half, half_file in zip(halves, halves_filtered):
                volume_data = ImageHandler(half).getData()
                volume_data_filtered = filterVol(volume_data, mode="bspline")
                ImageHandler().write(volume_data_filtered, self._getExtraPath(half_file))
                ImageHandler().setSamplingRate(half, imgSet.getSamplingRate())
                ImageHandler().setSamplingRate(self._getExtraPath(half_file), imgSet.getSamplingRate())
            volume.setHalfMaps(halves)

        # Filter volume -> Real reconstruction
        volume_data = ImageHandler(self._getExtraPath("final_reconstruction.mrc")).getData()
        volume_data_filtered = filterVol(volume_data, mode="bspline")
        ImageHandler().write(volume_data_filtered, self._getExtraPath("final_reconstruction_filtered.mrc"))

        # Set correct sampling rate in volume header
        ImageHandler().setSamplingRate(self._getExtraPath("final_reconstruction.mrc"), imgSet.getSamplingRate())
        ImageHandler().setSamplingRate(self._getExtraPath("final_reconstruction_filtered.mrc"), imgSet.getSamplingRate())

        self._defineOutputs(outputVolume=volume)
        self._defineOutputs(outputVolumeFiltered=volume_filtered)
        self._defineSourceRelation(self.inputParticles, volume)
        self._defineSourceRelation(self.inputParticles, volume_filtered)
    
    #--------------------------- INFO functions -------------------------------------------- 
    def _summary(self):
        """ Should be overriden in subclasses to 
        return summary message for NORMAL EXECUTION. 
        """
        return []

    def _validate(self):
        errors = []

        mask = self.recMask.get()
        if mask:
            data = ImageHandler(mask.getFileName()).getData()
            if not np.all(np.logical_or(data == 0, data == 1)):
                errors.append("Mask provided is not binary. Please, provide a binary mask")

        return errors

    #--------------------------- UTILS functions --------------------------------------------
    def defineZARTArgs(self, inputMd, outFile, niter, step, mask):
        params = ' -i %s' % inputMd
        params += ' -o %s' % outFile
        params += ' --odir %s' % self._getExtraPath()
        # params += ' --step %d' % step
        params += ' --step 1'
        params += ' --sym %s' % self.symmetryGroup.get()
        if "_level_" in outFile:
            reg = self.reg.get()
            level = float(re.findall(r'\d+', outFile)[1]) - 2
            if level > 0:
                params += " --regularization %f" % (reg * 0.9**level)
            else:
                params += " --regularization %f" % (reg)
        else:
            params += " --regularization %f" % self.reg.get()
        if hasattr(self, 'sigmas'):
            params += ' --sigma "' + self.sigmas + '"'
        else:
            params += ' --sigma "1"'
        params += ' --niter %d' % niter

        if isinstance(self.inputParticles.get(), SetOfParticlesFlex) and self.useZernike.get():
            particles = self.inputParticles.get()
            params += " --useZernike --l1 %d --l2 %d" % (particles.getFlexInfo().L1.get(),
                                                         particles.getFlexInfo().L2.get())
            # zernikeMask = particles.recMask.get() if hasattr(particles, "recMask") else self.mask.get()
            # if zernikeMask:
            #     params += " --mask %s" % zernikeMask

        if mask:
            params += ' --maskf %s --maskb %s' % (mask, mask)

        # Regularizations
        params += (' --ll1 %f --lst %f --ltv %f --ltk %f'
                   % (self.ll1.get(), self.lst.get(), self.ltv.get(), self.ltk.get()))

        if self.onlyPositive.get():
            params += " --onlyPositive"

        if self.sorting.get() == 1:
            params += " --sort_random"

        return params

    def volumeRestoration(self):
        half_1 = glob.glob(self._getExtraPath("final_reconstruction_1*"))
        half_2 = glob.glob(self._getExtraPath("final_reconstruction_2*"))
        half_1 = max(half_1, key=os.path.getctime)
        half_2 = max(half_2, key=os.path.getctime)

        params = "--i1 %s --i2 %s --oroot %s" % (half_1, half_2, self._getExtraPath("volume"))
        self.runJob('xmipp_volume_halves_restoration', params, numberOfMpi=1, env=xmipp3.Plugin.getEnviron())

        ih = ImageHandler()
        ih.convert(self._getExtraPath("volume_restored1.vol"), self._getExtraPath("final_reconstruction.mrc"),
                   overwrite=True)

        return [half_1, half_2]
