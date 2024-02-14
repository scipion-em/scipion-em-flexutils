# -*- coding: utf-8 -*-
# **************************************************************************
# *
# * Authors:     David Herreros   (dherreros@cnb.csic.es)
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


import pyworkflow.utils as pwutils
from pyworkflow.object import Float
from pyworkflow.protocol import params

from pwem.objects import Volume, SetOfParticlesFlex
from pwem.protocols import ProtFlexBase

from xmipp3.protocols.protocol_preprocess import XmippResizeHelper
from xmipp3.protocols.protocol_preprocess.protocol_preprocess import XmippProcessParticles, XmippProcessVolumes

import flexutils.constants as const

import xmipp3


def _getSize(imgSet):
    """ get the size of an object"""
    if isinstance(imgSet, Volume):
        Xdim = imgSet.getDim()[0]
    else:
        Xdim = imgSet.getDimensions()[0]
    return Xdim

def _getSampling(imgSet):
    """ get the sampling rate of an object"""
    samplingRate = imgSet.getSamplingRate()
    return samplingRate


class XmippProtCropResizeZernikeParticles(XmippProcessParticles, ProtFlexBase):
    """ Crop or resize a set of particles with Zernike3D coefficients associated """
    _label = 'crop/resize zernike particles'
    _inputLabel = 'particles'

    def __init__(self, **kwargs):
        XmippProcessParticles.__init__(self, **kwargs)

    # --------------------------- DEFINE param functions --------------------------------------------
    def _defineParams(self, form):
        form.addSection(label=pwutils.Message.LABEL_INPUT)

        form.addParam('inputParticles', params.PointerParam,
                      pointerClass='SetOfParticlesFlex',
                      label=pwutils.Message.LABEL_INPUT_PART, important=True)
        # Hook that should be implemented in subclasses
        self._defineProcessParams(form)

        __threads, __mpi = self._getDefaultParallel()

        form.addParallelSection(threads=__threads, mpi=__mpi)

    def _defineProcessParams(self, form):
        XmippResizeHelper._defineProcessParams(self, form)
        form.addParallelSection(threads=0, mpi=8)

    def _insertProcessStep(self):
        XmippResizeHelper._insertProcessStep(self)

    # --------------------------- STEPS functions ---------------------------------------------------
    def filterStep(self, isFirstStep, args):
        XmippResizeHelper.filterStep(self, self._ioArgs(isFirstStep) + args)

    def resizeStep(self, isFirstStep, args):
        XmippResizeHelper.resizeStep(self, self._ioArgs(isFirstStep) + args)

    def windowStep(self, isFirstStep, args):
        XmippResizeHelper.windowStep(self, self._ioArgs(isFirstStep) + args)

    def _preprocessOutput(self, output):
        """ We need to update the sampling rate of the
        particles if the Resize option was used.
        """
        inputParticles = self.inputParticles.get()
        self.inputHasAlign = inputParticles.hasAlignment()

        output.L1 = inputParticles.getFlexInfo().L1
        output.L2 = inputParticles.getFlexInfo().L2
        if self.doResize:
            output.setSamplingRate(self.samplingRate)
            output.getFlexInfo().Rmax = Float(self.factor * self.inputParticles.get().getFlexInfo().Rmax.get())

    def _updateItem(self, item, row):
        """ Update also the sampling rate and
        the alignment if needed.
        """
        XmippProcessParticles._updateItem(self, item, row)
        if self.doResize:
            if item.hasCoordinate():
                item.scaleCoordinate(self.factor)
            item.setSamplingRate(self.samplingRate)
            if self.inputHasAlign:
                item.getTransform().scaleShifts(self.factor)

            # Scale Zernike3D information
            z_clnm = self.factor * item.getZFlex()
            item.setZFlex(z_clnm)

            if hasattr(item.getFlexInfo(), "deformation"):
                deformation = item.getFlexInfo().deformation.get()
                item.getFlexInfo().deformation = Float(self.factor * deformation)


    # --------------------------- INFO functions ----------------------------------------------------
    def _summary(self):
        summary = []

        if not hasattr(self, 'outputParticles'):
            summary.append("Output images not ready yet.")
        else:
            sampling = _getSampling(self.outputParticles)
            size = _getSize(self.outputParticles)
            if self.doResize:
                summary.append(u"Output particles have a different sampling "
                               u"rate (pixel size): *%0.3f* Å/px" % sampling)
                summary.append("Resizing method: *%s*" %
                               self.getEnumText('resizeOption'))
            if self.doWindow:
                if self.getEnumText('windowOperation') == "crop":
                    summary.append("The particles were cropped.")
                else:
                    summary.append("The particles were windowed.")
                summary.append("New size: *%s* px" % size)
        return summary

    def _methods(self):

        if not hasattr(self, 'outputParticles'):
            return []

        methods = ["We took input particles %s of size %d " % (
        self.getObjectTag('inputParticles'), len(self.inputParticles.get()))]
        if self.doWindow:
            if self.getEnumText('windowOperation') == "crop":
                methods += ["cropped them"]
            else:
                methods += ["windowed them"]
        if self.doResize:
            outputParticles = getattr(self, 'outputParticles', None)
            if outputParticles is None or outputParticles.getDim() is None:
                methods += ["Output particles not ready yet."]
            else:
                methods += ['resized them to %d px using the "%s" method%s' %
                            (outputParticles.getDim()[0],
                             self.getEnumText('resizeOption'),
                             " in Fourier space" if self.doFourier else "")]
        if not self.doResize and not self.doWindow:
            methods += ["did nothing to them"]
        str = "%s and %s. Output particles: %s" % (", ".join(methods[:-1]),
                                                   methods[-1],
                                                   self.getObjectTag('outputParticles'))
        return [str]

    def _validate(self):
        errors = XmippResizeHelper._validate(self)

        inputParticles = self.inputParticles.get()

        inputParticles = self.inputParticles.get()
        if isinstance(inputParticles, SetOfParticlesFlex):
            if inputParticles.getFlexInfo().getProgName() != const.ZERNIKE3D:
                errors.append("The flexibility information associated with the particles is not "
                              "coming from the Zernike3D algorithm. Please, provide a set of particles "
                              "with the correct flexibility information.")

        return errors

    # --------------------------- UTILS functions ---------------------------------------------------
    def _ioArgs(self, isFirstStep):
        if isFirstStep:
            return "-i %s -o %s --save_metadata_stack %s --keep_input_columns " % (
            self.inputFn, self.outputStk, self.outputMd)
        else:
            return "-i %s " % self.outputStk

    def _filterArgs(self):
        return XmippResizeHelper._filterCommonArgs(self)

    def _resizeArgs(self):
        return XmippResizeHelper._resizeCommonArgs(self)

    def _windowArgs(self):
        return XmippResizeHelper._windowCommonArgs(self)

    def _getSetSize(self):
        """ get the size of SetOfParticles object"""
        imgSet = self.inputParticles.get()
        return _getSize(imgSet)

    def _getSetSampling(self):
        """ get the sampling rate of SetOfParticles object"""
        imgSet = self.inputParticles.get()
        return _getSampling(imgSet)

    def _getDefaultParallel(self):
        """ Return the default value for thread and MPI
        for the parallel section definition.
        """
        return (0, 1)


class XmippProtCropResizeZernikeVolumes(XmippProcessVolumes):
    """ Crop or resize a set of volumes with Zernike3D coefficients associated """
    _label = 'crop/resize zernike volumes'
    _inputLabel = 'volumes'

    def __init__(self, **kwargs):
        XmippProcessVolumes.__init__(self, **kwargs)

    # --------------------------- DEFINE param functions --------------------------------------------
    def _defineParams(self, form):
        form.addSection(label=pwutils.Message.LABEL_INPUT)

        form.addParam('inputVolumes', params.PointerParam, important=True,
                      label=pwutils.Message.LABEL_INPUT_VOLS, pointerClass='VolumeFlex, SetOfVolumesFlex',
                      help='Can be a density volume or a SetOfVolumesFlex')
        # Hook that should be implemented in subclasses
        self._defineProcessParams(form)
        form.addParallelSection(threads=2, mpi=1)

    def _defineProcessParams(self, form):
        XmippResizeHelper._defineProcessParams(self, form)

    def _insertProcessStep(self):
        XmippResizeHelper._insertProcessStep(self)

    # --------------------------- STEPS functions ---------------------------------------------------
    def filterStep(self, isFirstStep, args):
        self.runJob("xmipp_transform_filter", self._ioArgs(isFirstStep) + args, env=xmipp3.Plugin.getEnviron())

    def resizeStep(self, isFirstStep, args):
        self.runJob("xmipp_image_resize", self._ioArgs(isFirstStep) + args, env=xmipp3.Plugin.getEnviron())

    def windowStep(self, isFirstStep, args):
        self.runJob("xmipp_transform_window", self._ioArgs(isFirstStep) + args, env=xmipp3.Plugin.getEnviron())

    def _preprocessOutput(self, volumes):
        # We use the preprocess only whne input is a set
        # we do not use postprocess to setup correctly
        # the samplingRate before each volume is added
        if not self._isSingleInput():
            if self.doResize:
                volumes.setSamplingRate(self.samplingRate)

    def _postprocessOutput(self, volumes):
        # We use the postprocess only when input is a volume
        if self._isSingleInput():
            if self.doResize:
                volumes.setSamplingRate(self.samplingRate)
                # we have a new sampling so origin need to be adjusted
                iSampling = self.inputVolumes.get().getSamplingRate()
                oSampling = self.samplingRate
                xdim_i, ydim_i, zdim_i = self.inputVolumes.get().getDim()
                xdim_o, ydim_o, zdim_o = volumes.getDim()

                xOrig, yOrig, zOrig = \
                    self.inputVolumes.get().getShiftsFromOrigin()
                xOrig += (xdim_i * iSampling - xdim_o * oSampling) / 2.
                yOrig += (ydim_i * iSampling - ydim_o * oSampling) / 2.
                zOrig += (zdim_i * iSampling - zdim_o * oSampling) / 2.
                volumes.setShiftsInOrigin(xOrig, yOrig, zOrig)
                volumes.setSamplingRate(oSampling)

                # Scale Zernike3D information
                # FIXME: Esto HAY que arreglarlo (no guarda los valores)
                for item in volumes.iterItems(iterate=False):
                    item.getFlexInfo().Rmax = Float(self.factor * item.getFlexInfo().Rmax.get())
                    z_clnm = self.factor * item.getZFlex()
                    item.setZFlex(z_clnm)

    # --------------------------- INFO functions ----------------------------------------------------
    def _summary(self):
        summary = []

        if not hasattr(self, 'outputVol'):
            summary.append("Output volume(s) not ready yet.")
        else:
            sampling = _getSampling(self.outputVol)
            size = _getSize(self.outputVol)
            if self.doResize:
                summary.append(u"Output volume(s) have a different sampling "
                               u"rate (pixel size): *%0.3f* Å/px" % sampling)
                summary.append("Resizing method: *%s*" %
                               self.getEnumText('resizeOption'))
            if self.doWindow.get():
                if self.getEnumText('windowOperation') == "crop":
                    summary.append("The volume(s) were cropped.")
                else:
                    summary.append("The volume(s) were windowed.")
                summary.append("New size: *%s* px" % size)
        return summary

    def _methods(self):
        if not hasattr(self, 'outputVol'):
            return []

        if self._isSingleInput():
            methods = ["We took one volume"]
            pronoun = "it"
        else:
            methods = ["We took %d volumes" % self.inputVolumes.get().getSize()]
            pronoun = "them"
        if self.doWindow.get():
            if self.getEnumText('windowOperation') == "crop":
                methods += ["cropped %s" % pronoun]
            else:
                methods += ["windowed %s" % pronoun]
        if self.doResize:
            outputVol = getattr(self, 'outputVol', None)
            if outputVol is None or self.outputVol.getDim() is None:
                methods += ["Output volume not ready yet."]
            else:
                methods += ['resized %s to %d px using the "%s" method%s' %
                            (pronoun, self.outputVol.getDim()[0],
                             self.getEnumText('resizeOption'),
                             " in Fourier space" if self.doFourier else "")]
        if not self.doResize and not self.doWindow:
            methods += ["did nothing to %s" % pronoun]
            # TODO: does this case even work in the protocol?
        return ["%s and %s." % (", ".join(methods[:-1]), methods[-1])]

    def _validate(self):
        return XmippResizeHelper._validate(self)

    # --------------------------- UTILS functions ---------------------------------------------------
    def _ioArgs(self, isFirstStep):
        if isFirstStep:
            if self._isSingleInput():
                return "-i %s -o %s " % (self.inputFn, self.outputStk)
            else:
                return "-i %s -o %s --save_metadata_stack %s --keep_input_columns " % (
                self.inputFn, self.outputStk, self.outputMd)
        else:
            return "-i %s" % self.outputStk

    def _filterArgs(self):
        return XmippResizeHelper._filterCommonArgs(self)

    def _resizeArgs(self):
        return XmippResizeHelper._resizeCommonArgs(self)

    def _windowArgs(self):
        return XmippResizeHelper._windowCommonArgs(self)

    def _getSetSize(self):
        """ get the size of either Volume or SetOfVolumes objects"""
        imgSet = self.inputVolumes.get()
        size = _getSize(imgSet)
        return size

    def _getSetSampling(self):
        """ get the sampling rate of either Volume or SetOfVolumes objects"""
        imgSet = self.inputVolumes.get()
        samplingRate = _getSampling(imgSet)
        return samplingRate