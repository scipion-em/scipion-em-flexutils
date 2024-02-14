# -*- coding: utf-8 -*-
# **************************************************************************
# *
# * Authors: Martín Salinas Antón (ssalinasmartin@gmail.com)
# * Authors: David Herreros (dherreros@cnb.csic.es)
# *
# *  BCU, Centro Nacional de Biotecnologia, CSIC
# *
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

# General imports
import os
import numpy as np
from xmipp_metadata.metadata import XmippMetaData
from xmipp_metadata.image_handler import ImageHandler


# Scipion em imports
from pwem.protocols import ProtAnalysis3D, ProtFlexBase
from pwem.objects import SetOfParticles, Volume, Integer
from pyworkflow import NEW
from pyworkflow.protocol import params
import pyworkflow.utils as pwutils

# External plugin imports
import xmipp3
from xmipp3.convert import readSetOfImages, rowToParticle


# Protocol output variable name
OUTPUTATTRIBUTE = 'volumeProjections'


def rowToParticleWithLabel(partRow, **kwargs):
    img = rowToParticle(partRow, **kwargs)
    img._xmipp_subtomo_labels = Integer(float(partRow.getValue("subtomo_labels")))
    img.setObjId(None)
    return img


class XmippProtPrepareZernikeVolumes(ProtAnalysis3D, ProtFlexBase):
    """Extracts projections from a set of volumes (keeping a volume id)"""

    # Protocol constants
    _label = 'prepare volumes - Zernike3Deep'
    _devStatus = NEW
    _possibleOutputs = {OUTPUTATTRIBUTE: SetOfParticles}

    # Form constants
    METHOD_FOURIER = 0
    METHOD_REAL_SPACE = 1
    METHOD_SHEARS = 2
    TYPE_N_SAMPLES = 0
    TYPE_STEP = 1

    # --------------------------- Class constructor --------------------------------------------
    def __init__(self, **args):
        # Calling parent class constructor
        super().__init__(**args)

        # Defining execution mode. Steps will take place in parallel now
        # Full tutorial on how to parallelize protocols can be read here:
        # https://scipion-em.github.io/docs/release-3.0.0/docs/developer/parallelization.html
        self.stepsExecutionMode = params.STEPS_PARALLEL

    # --------------------------- DEFINE param functions ------------------------
    def _defineParams(self, form):
        # Defining parallel arguments
        form.addParallelSection(threads=4)

        # Generating form
        form.addSection(label='Input subtomograms')
        form.addParam('inputVolumes', params.PointerParam, pointerClass="SetOfVolumes",
                      label='Set of volumes', help="Set of volumes whose projections will be generated.")
        form.addParam('cleanTmps', params.BooleanParam, default='True', label='Clean temporary files: ', expertLevel=params.LEVEL_ADVANCED,
                        help='Clean temporary files after finishing the execution.\nThis is useful to reduce unnecessary disk usage.')
        form.addParam('transformMethod', params.EnumParam, display=params.EnumParam.DISPLAY_COMBO, default=self.METHOD_FOURIER,
                      choices=['Fourier', 'Real space', 'Shears'], label="Transform method: ", expertLevel=params.LEVEL_ADVANCED,
                      help='Select the algorithm that will be used to obtain the projections.')
        projGroup = form.addGroup('Projection parameters')
        projGroup.addParam('maskMaps', params.BooleanParam, default=True, label="Mask maps?",
                           help="Automatically mask the input volumes to remove noise. Set to false "
                                "if projections do not look sensible enough.")
        projGroup.addParam('typeGeneration', params.EnumParam, display=params.EnumParam.DISPLAY_COMBO, default=self.TYPE_N_SAMPLES,
                           choices=['NSamples', 'Step'], label="Type of sample generation: ",
                           help='Select either the number of samples to be taken or the separation in degrees between each sample.')
        projGroup.addParam('rangeNSamples', params.IntParam, condition='typeGeneration==%d' % self.TYPE_N_SAMPLES, 
                           label='Number of samples:', default=1000,
                           help='Number of samples to be produced.\nIt has to be 1 or greater.')
        projGroup.addParam('rangeStep', params.IntParam, condition='typeGeneration==%d' % self.TYPE_STEP, 
                           label='Step:', default=5,
                           help='Number of degrees each sample will be separated from the next.\nIt has to be greater than 0.')

    # --------------------------- INSERT steps functions --------------------------------------------
    def _insertAllSteps(self):
        # Writing config file
        generateConfigFile = self._insertFunctionStep(self.generateConfigFile)

        # Defining list of function ids to be waited by the createOutput function
        deps = [generateConfigFile]
        # Generating projections for each volume
        volId = 1
        for volume in self.inputVolumes.get():
            deps.append(self._insertFunctionStep(self.generateProjections,
                                                 volume.getFileName(),
                                                 prerequisites=[generateConfigFile]))
            volId += 1

        # Conditionally removing temporary files
        if self.cleanTmps.get():
            deps.append(self._insertFunctionStep(self.removeTempFiles, prerequisites=deps))
        
        # Create output
        self._insertFunctionStep(self.createOutputStep, prerequisites=deps)

    # --------------------------- STEPS functions --------------------------------------------
    def generateConfigFile(self):
        """
        This function writes the config file for Xmipp Phantom.
        """
        confFile = open(self.getXmippParamPath(), "w")

        # Generating file content
        content = '# XMIPP_STAR_1 *\n'
        content += '# Projection Parameters\n'
        content += 'data_block1\n'
        content += '# X and Y projection dimensions [Xdim Ydim]\n'
        content += '_dimensions2D   \'{}\'\n'.format(self.getVolumeDimensions())
        content += '# Rotation range and number of samples [Start Finish NSamples]\n'
        content += '_projRotRange    \'0 360 {}\'\n'.format(self.getStepValue())
        content += '# Rotation angle added noise  [noise (bias)]\n'
        content += '_projRotNoise   \'0\'\n'
        content += '# Tilt range and number of samples for Tilt [Start Finish NSamples]\n'
        content += '_projTiltRange    \'0 180 {}\'\n'.format(self.getStepValue())
        content += '# Tilt angle added noise\n'
        content += '_projTiltNoise   \'0\'\n'
        content += '# Psi range and number of samples\n'
        content += '_projPsiRange    \'0 0 0\'\n'
        content += '# Psi added noise\n_projPsiNoise   \'0\'\n'
        content += '# Noise\n'

        # Writing content to file and closing
        confFile.write(content)
        confFile.close()

    def generateProjections(self, volume):
        """
        This function generates the projection for a given input volume.
        """
        masked_path = self.getVolumeAbsolutePath(volume)
        if self.maskMaps.get():
            vol_path = self.getVolumeAbsolutePath(volume)
            masked_path = self.getVolumeAbsolutePath(self._getExtraPath(pwutils.removeBaseExt(vol_path) + "_masked.mrc"))
            ih = ImageHandler(vol_path)
            mask = ih.generateMask(iterations=50, boxsize=64, smoothStairEdges=False)
            ImageHandler().write(ih.getData() * mask, masked_path, overwrite=True)

        params = '-i {}'.format(masked_path + ":mrc")   # Path to volume
        params += ' -o {}'.format(self.getProjectionAbsolutePath(volume))  # Path to output projection
        params += ' --method {}'.format(self.getMethodValue())             # Projection algorithm
        params += ' --params {}'.format(self.getXmippParamPath())          # Path to Xmipp phantom param file
        self.runJob("xmipp_phantom_project", params, cwd='/home', env=xmipp3.Plugin.getEnviron())

    def removeTempFiles(self):
        """
        This function removes the temporary files of this protocol.
        """
        # Removing Xmipp Phantom config file
        self.runJob('rm', self.getXmippParamPath())

    def createOutputStep(self):
        """
        This function generates the outputs of the protocol.
        """
        # Extracting input
        inputVolumes = self.inputVolumes.get()

        # Creating empty set of particles and setting sampling rate, alignment, and dimensions
        outputSetOfParticles = self._createSetOfParticles()
        outputSetOfParticles.setSamplingRate(inputVolumes.getSamplingRate())
        outputSetOfParticles.setAlignmentProj()
        dimensions = self.getVolumeDimensions().split(' ')
        outputSetOfParticles.setDim((int(dimensions[0]), int(dimensions[1]), 1))

        # Adding projections of each volume as a particle each
        label = 1
        for volume in inputVolumes.iterItems():
            md = XmippMetaData(self.getProjectionMetadataAbsolutePath(volume))
            subtomo_labels = label * np.ones(len(md))
            md[:, "subtomo_labels"] = subtomo_labels
            md.write(self.getProjectionMetadataAbsolutePath(volume), overwrite=True)
            readSetOfImages(self.getProjectionMetadataAbsolutePath(volume), outputSetOfParticles,
                            rowToParticleWithLabel)
            label += 1

        # Defining the ouput with summary and source relation
        outputSetOfParticles.setObjComment(self.getSummary(outputSetOfParticles))
        self._defineOutputs(outputSetOfParticles=outputSetOfParticles)
        self._defineSourceRelation(self.inputVolumes, outputSetOfParticles)

    # --------------------------- INFO functions --------------------------------------------
    def _validate(self):
        """
        This method validates the received params and checks that they all fullfill the requirements needed to run the protocol.
        """
        errors = []

        # Checking if number of samples is greater or equal than 1
        if self.typeGeneration.get() == self.TYPE_N_SAMPLES and (not self.rangeNSamples.get() == None) and self.rangeNSamples.get() < 1:
            errors.append('The number of samples cannot be less than 1.')
        
        # Checking if the step is greater than 0
        if self.typeGeneration.get() == self.TYPE_STEP and (not self.rangeStep.get() == None) and self.rangeStep.get() <= 0:
            errors.append('The step must be greater than 0.')
        
        # Checking if MPI is selected (only threads are allowed)
        if self.numberOfMpi > 1:
            errors.append('MPI cannot be selected, because Scipion is going to drop support for it. Select threads instead.')

        return errors

    def _summary(self):
        """
        This method usually returns a summary of the text provided by '_methods'.
        """
        return []

    def _methods(self):
        """
        This method returns a text intended to be copied and pasted in the paper section 'materials & methods'.
        """
        return []

    # --------------------------- UTILS functions --------------------------------------------
    def scapePath(self, path):
        """
        This function returns the given path with all the spaces in folder names scaped to avoid errors.
        """
        # os.path.baspath adds '\\' when finding a foldername with '\ ', so '\\\' needs to be replaced with ''
        # Then, '\' is inserted before every space again, to include now possible folders with spaces in the absolute path
        return path.replace('\\\ ', ' ').replace(' ', '\ ')
    
    def getVolumeRelativePath(self, volume):
        """
        This method returns a the volume path relative to current directory.
        Path is scaped to support spaces.
        Example:
            if a file is in /home/username/documents/test/import_file.mrc
            and current directory is /home/username/documents
            this will return '/test/import_file.mrc'
        """
        return self.scapePath(volume.getFileName() if isinstance(volume, Volume) else volume)

    def getVolumeAbsolutePath(self, volume):
        """
        This method returns a the absolute path for the volume.
        Path is scaped to support spaces.
        Example: '/home/username/documents/test/import_file.mrc'
        """
        return self.scapePath(os.path.abspath(self.getVolumeRelativePath(volume)))

    def getVolumeName(self, filename):
        """
        This method returns the full name of the given volume files.
        Example: import_file.mrc
        """
        return os.path.basename(filename)
    
    def getCleanVolumeName(self, filename):
        """
        This method returns the full name of the given volume file without the 'import_' prefix.
        Example:
            if filename is 'import_file.mrc'
            this will return 'file.mrc'
        """
        return self.getVolumeName(filename).replace('import_', '')

    def getProjectionName(self, volume):
        """
        This function returns the name of the projection file for a given input volume.
        """
        name = os.path.splitext(self.getCleanVolumeName(self.getVolumeAbsolutePath(volume)))[0]
        return '{}_image.mrc'.format(name)
    
    def getProjectionMetadataName(self, volume):
        """
        This function returns the filename of the metadata file for a given input volume.
        """
        return self.getProjectionName(volume).replace('.mrc', '.xmd')

    def getProjectionAbsolutePath(self, volume):
        """
        This function returns the full path of a given volume.
        """
        return self.scapePath(os.path.abspath(self._getExtraPath(self.getProjectionName(volume))))
    
    def getProjectionMetadataAbsolutePath(self, volume):
        """
        This function returns the full path of a given volume's metadata file.
        """
        return self.getProjectionAbsolutePath(volume).replace('.mrc', '.xmd')

    def getStepValue(self):
        """
        This function translates the provided sample generation input to number of samples for Xmipp phantom.
        """
        if self.typeGeneration.get() == self.TYPE_N_SAMPLES:
            return int(np.sqrt(self.rangeNSamples.get()))
        else:
            # Converting step to number of samples
            return int(np.sqrt(360.0 / self.rangeStep.get()))
    
    def getMethodValue(self):
        """
        This function returns the string value associated to the form value provided by the user regarding transform method.
        """
        if self.transformMethod.get() == self.METHOD_FOURIER:
            return 'fourier'
        elif self.transformMethod.get() == self.METHOD_REAL_SPACE:
            return 'real_space'
        else:
            return 'shears'

    def getVolumeDimensions(self):
        """
        This function retuns the first two dimensions of the subtomograms.
        """
        try:
            dimensions = self.inputVolumes.get().getFirstItem().getDimensions()
            return '{} {}'.format(dimensions[0], dimensions[1])
        except TypeError:
            errorMessage = " ".join(["No subtomograms were received. Check the output of the previous protocol.",
                                     "If you are using the integrated test, run the extract subtomos's test first."])
            raise Exception(errorMessage)

    def getXmippParamPath(self):
        """
        This function returns the path for the config file for Xmipp Phantom.
        """
        return self.scapePath(os.path.abspath(os.path.join(self._getExtraPath(''), 'xmippPhantom.param')))
    
    def getSummary(self, setOfParticles):
        """
        Returns the summary of a given set of particles.
        The summary consists of a text including the number of particles in the set.
        """
        return "Number of projections generated: {}".format(setOfParticles.getSize())
