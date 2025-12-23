# ***************************************************************************
# * Authors:    Marta Martinez (mmmtnez@cnb.csic.es)
# *             Roberto Marabini (roberto@cnb.csic.es)
# *
# * This program is free software; you can redistribute it and/or modify
# * it under the terms of the GNU General Public License as published by
# * the Free Software Foundation; either version 3 of the License, or
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
# ***************************************************************************/

import os
import glob
from tempfile import NamedTemporaryFile
import pwem.convert as emconv

from pwem.protocols import (ProtImportParticles,
                            ProtImportVolumes)
from pyworkflow.tests import BaseTest, setupTestProject
import numpy as np
from flexutils.protocols import (
    TensorflowProtAngularAlignmentHetSiren,
    ProtFlexDimRedSpace,
    ProtFlexAnnotateSpace
)
import tempfile
from xmipp3.protocols.protocol_reconstruct_fourier import (XmippProtReconstructFourier)

class TestFlexHetSiren(BaseTest):
    @classmethod
    def setUpClass(cls):
        setupTestProject(cls)
        cls.xmippAvailable = True
        cls.cryoSparcAvailable = True
        cls.sampling = 1.35
        cls.size = 128
        cls.gpuList = '1'

        from pwem import Domain # import here to avoid ImportError
        try:
            cls.xmipp3 = \
                Domain.importFromPlugin('xmipp3', doRaise=True)
        except Exception as e:
            print("xmipp3 not available, cancel test", e)
            cls.xmippAvailable = False
        cls.absTmpPath = os.path.join(
            cls.getOutputPath(), cls.proj.getTmpPath())

    def __runXmippProgram(self, program, args):
        """ Internal shortcut function to launch a Xmipp program.
        If xmipp not available o fails return False, else True"""
        try:
            from pwem import Domain
            xmipp3 = Domain.importFromPlugin('xmipp3', doRaise=True)
            xmipp3.Plugin.runXmippProgram(program, args)
        except ImportError:
            return False
        return True

    @classmethod
    def runCreateMask(cls, pattern, thr):
        """ Create a volume mask. 
        must be binary
        """
        from xmipp3.protocols import XmippProtCreateMask3D

        cls.msk = cls.newProtocol(XmippProtCreateMask3D,
                                  inputVolume=pattern,
                                  volumeOperation=0,  # OPERATION_THRESHOLD,
                                  threshold=thr,
                                  doSmall=False,
                                  smallSize=False,
                                  doBig=False,
                                  doSymmetrize=False,
                                  doMorphological=True,
                                  morphologicalOperation=0,  # dialtion
                                  elementSize=1,
                                  doInvert=False,
                                  doSmooth=False
                                  )
        cls.launchProtocol(cls.msk)
        return cls.msk

    @classmethod
    def runApplyMask(cls, volume, mask):
        """ Apply a mask to a volume. """
        from xmipp3.protocols import XmippProtMaskVolumes

        protApplyMask = cls.newProtocol(XmippProtMaskVolumes,
                                        inputVolumes=volume,
                                        source=1,  # SOURCE_VOLUME
                                        inputMask=mask
                                        )
        cls.launchProtocol(protApplyMask)
        return protApplyMask

    @classmethod
    def runImportVolume(cls, pattern, samplingRate,
                        importFrom=ProtImportParticles.IMPORT_FROM_FILES):
        """ Run an Import volumes protocol. """
        protImportVol = cls.newProtocol(ProtImportVolumes,
                                        filesPath=pattern,
                                        samplingRate=samplingRate,
                                        copyFiles=True
                                        )
        cls.launchProtocol(protImportVol)
        return protImportVol

    @classmethod
    def runImportParticles(cls, pattern, samplingRate, checkStack=False,
                           importFrom=ProtImportParticles.IMPORT_FROM_FILES):
        """ Import particles protocol. """
        if importFrom == ProtImportParticles.IMPORT_FROM_SCIPION:
            objLabel = 'from scipion (particles)'
        elif importFrom == ProtImportParticles.IMPORT_FROM_FILES:
            objLabel = 'from file (particles)'
        elif importFrom == ProtImportParticles.IMPORT_FROM_XMIPP3:
            objLabel = 'from xmipp3 (particles)'

        protImportPart = cls.newProtocol(ProtImportParticles,
                                         objLabel=objLabel,
                                         filesPath=pattern,  # input files
                                         sqliteFile=pattern,
                                         mdFile=pattern,
                                         samplingRate=samplingRate,
                                         checkStack=checkStack,
                                         importFrom=importFrom,
                                         voltage=300,
                                         sphericalAberration=2,
                                         amplitudeContrast=.1,
                                         copyFiles=True
                                         )
        cls.launchProtocol(protImportPart)
        # Check that input images have been imported (a better way to do this?)
        if protImportPart.outputParticles is None:
            raise Exception('Import of images: %s, failed. '
                            'outputParticles is None.' % pattern)
        return protImportPart

    def getAtomStructFile(self, atomStructID):
        "download pdb file from the database"
        aSH = emconv.AtomicStructHandler()
        atomStructPath = aSH.readFromPDBDatabase(
            atomStructID, dir=self.absTmpPath, type='pdb')
        # filter out HETATM
        tempFile  = os.path.join(self.absTmpPath, "kk")
        os.system(
            f"cat {atomStructPath} |"
            f" grep -v HETATM > {tempfile}; mv {tempfile} {atomStructPath}")
        return atomStructPath

    def createVolume(self, atomStructPath, volMapName):
        # create volume
        volumeName = f"{volMapName}"
        self.__runXmippProgram('xmipp_volume_from_pdb', f'-i {atomStructPath}'
                               f' -o {volumeName} --size {self.size}'
                               f' --centerPDB --sampling {self.sampling}')
        return volumeName + '.vol'

    def shiftFirstChain(
            self,
            atomStructPath,
            translation_vector=[0.0, 0.0, 0.0],
            nTimes=1):
        # import PDB needs new version of Biopython
        # or recent version of pwem
        # leave the import here so it is not executed
        # during the test detection
        from Bio.PDB import PDBParser, PDBIO

        def translate_structure(structure, translation_vector):
            """
            Translate the structure by a given vector.

            Args:
                structure: Biopython structure object.
                translation_vector: List or tuple of (x, y, z)
                                    translation values.
            """
            translation_vector = translation_vector * nTimes
            for model in structure:
                for chain in model:
                    for residue in chain:
                        for atom in residue:
                            # Translate the atom's coordinates
                            new_coord = atom.coord + translation_vector
                            atom.coord = new_coord
                    break  # Break after the first chain
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure("example_structure", atomStructPath)
        translate_structure(structure, translation_vector)
        io = PDBIO()
        io.set_structure(structure)
        output_path = atomStructPath.replace(".ent", f"_shifted_{nTimes}.pdb")
        io.save(output_path)
        return output_path

    def createProjections(self, volumeNames, angular_sampling_rate=15):
        projStackNames = []
        for vol in volumeNames:
            print("Creating projections for volume:",  vol)
            args = " -i %s" % vol
            args += " -o %s" % vol.replace(".vol", ".mrcs")
            args += f" --sampling_rate {angular_sampling_rate}"
            args += " --sym c1"
            args += " --method real_space"
            # args += " --method real_space&"
            progname = "xmipp_angular_project_library"
            self.xmipp3.Plugin.runXmippProgram(progname, args)
            projStackNames.append(vol.replace(".vol", ".doc"))
        # For production remove '&' and sleep
        # import time
        # time.sleep(25)
        return projStackNames

    def createCTFdata(self):
        ctfFileName = os.path.join(self.absTmpPath, "ctf.ctfdata")
        ctfFile = open(ctfFileName, "wb")
        command = f"""# XMIPP_STAR_1 *
#  SamplingRate should be the same that the one used in the micrographs
data_fullMicrograph
 _ctfSamplingRate {self.sampling}
 _ctfVoltage 300
 _ctfDefocusU 6000
 _ctfDefocusV 6000
 _ctfDefocusAngle -140.258
 _ctfSphericalAberration 2
 _ctfQ0 0.1
"""
        ctfFile.write(command.encode('utf8'))
        ctfFile.close()
        return ctfFile.name

    def unionSets(self, xmdProjNames):

        for i in range(1, len(xmdProjNames)):
            args = " -i %s" % xmdProjNames[0]
            args += " --set union %s" % xmdProjNames[i]
            args += " -o %s" % xmdProjNames[0]
            progname = "xmipp_metadata_utilities"
            self.xmipp3.Plugin.runXmippProgram(progname, args)
        return xmdProjNames[0]

    def addNoiseAndCTF(self, projDocNames):
        ctfFile = self.createCTFdata()
        print("ctfFile", ctfFile)
        xmdProjNames = []
        for projDocName in projDocNames:
            args = " -i %s" % projDocName
            args += " -o %s" % projDocName.replace(".doc", "ctf.mrcs")
            args += f" --ctf {ctfFile}"
            args += " --noise 10 --after_ctf_noise"
            progname = "xmipp_phantom_simulate_microscope"
            self.xmipp3.Plugin.runXmippProgram(progname, args)
            xmdProjNames.append(projDocName.replace(".doc", "ctf.xmd"))
        xmdProjName = self.unionSets(xmdProjNames)
        return xmdProjName

    def importData(self, xmdProjName, volName):
        # import particles
        protImportProj = self.runImportParticles(
            xmdProjName,
            self.sampling,
            importFrom=ProtImportParticles.IMPORT_FROM_XMIPP3)
        # import volume
        protImportVolume = self.runImportVolume(
            volName,
            self.sampling)
        return protImportProj, protImportVolume

    def protXmippReconstruct(self, particles, mask):
        recProt1 = self.newProtocol(
            XmippProtReconstructFourier,
            inputParticles=particles)

        protRelionReconstruct = self.launchProtocol(recProt1)
        return protRelionReconstruct

    def runDataPrepare(
            self, particles,volume, trainigBoxSize=128):
        protDataPrepare = self.newProtocol(
            ProtCryoSparc3DFlexDataPrepare,
            refVolume=volume,
            inputParticles=particles,
            bin_size_pix=trainigBoxSize)
        self.launchProtocol(protDataPrepare)
        return protDataPrepare

    def runFlexibleAlign(self, particles, volume, mask):
        # do not use volume as input
        # the program does not provide good results
        protFlexibleAlign = self.newProtocol(
            TensorflowProtAngularAlignmentHetSiren,
            inputParticles=particles,
            inputVolumeMask=mask,
            gpuList=self.gpuList,
            epochs=10,
            )
        self.launchProtocol(protFlexibleAlign)
        return protFlexibleAlign

    def runFlexDimRedSpace(self, particles, dimensions=3):
        protDimRedSpace = self.newProtocol(
            ProtFlexDimRedSpace,
            particles=particles,
            dimensions=dimensions,
            gpuList=self.gpuList,
            useGpu=True,
            )
        self.launchProtocol(protDimRedSpace)
        return protDimRedSpace

    def runFlexAnnotateSpace(self, particles):
        protFlexAnnotateSpace = self.newProtocol(
            ProtFlexAnnotateSpace,
            particles=particles,
            useGpu=True,
            gpuList=self.gpuList
            )
        self.saveProtocol(protFlexAnnotateSpace)
        return protFlexAnnotateSpace

    def testFlexSystem(self):
        if not self.xmippAvailable:
            # if xmipp is not available, just
            # skip this test
            return
        atomStructID = '3wtg'
        nVolumes = 2  # realistic value 5
        self.angular_sampling_rate = 30  # realistic value 3
        volMapName = os.path.join(
            self.absTmpPath, f'{atomStructID}_shifted_%d')
        atomStructPath = self.getAtomStructFile(atomStructID)
        translation_vector = np.array([0.5, 0.0, -0.25])
        volumeNames = []

        for i in range(nVolumes):
            shiftChain = self.shiftFirstChain(
                atomStructPath=atomStructPath,
                translation_vector=translation_vector,
                nTimes=i)
            volumeName = self.createVolume(
                atomStructPath=shiftChain,
                volMapName=volMapName % i)
            volumeNames.append(volumeName)
        projDocNames = self.createProjections(
            volumeNames, self.angular_sampling_rate)
        xmdProjFile = self.addNoiseAndCTF(projDocNames)
        volName = volumeNames[0]
        print("import data")
        protImportProj, protImportVolume =\
            self.importData(xmdProjName=xmdProjFile, volName=volName)
        print("create mask")
        protCreateMask = self.runCreateMask(
            pattern=protImportVolume.outputVolume,
            thr=0.1)
        print("reconstruction")
        protCryoSparcReconstruct = \
            self.protXmippReconstruct(
                protImportProj.outputParticles,
                protCreateMask.outputMask)
        print("apply mask")
        protApplyMask = self.runApplyMask(
            protCryoSparcReconstruct.outputVolume,
            protCreateMask.outputMask)
        print("run flexible align - hetsiren")
        protFlexibleAlign = self.runFlexibleAlign(
            protImportProj.outputParticles,
            protApplyMask.outputVol,
            protCreateMask.outputMask,
            )
        print("reduce DimRedSpace")
        protFlexDimRedSpace = self.runFlexDimRedSpace(
            protFlexibleAlign.outputParticles,
            dimensions=3
        )
        print("anotate space")
        protFlexAnnotateSpace = self.runFlexAnnotateSpace(
            protFlexDimRedSpace.outputParticles,
        )
