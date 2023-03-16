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


import numpy as np

from pwem import EMObject
from pwem.objects import SetOfParticles, Particle, SetOfClasses, Volume, Transform

from pyworkflow.object import String, CsvList


class ParticleFlex(Particle):
    """Particle with flexibility information stored"""

    def __init__(self, progName="", **kwargs):
        Particle.__init__(self, **kwargs)
        self._flexInfo = FlexInfo(progName)
        self._zFlex = CsvList()
        self._zRed = CsvList()
        self._transform = Transform()

    def getFlexInfo(self):
        return self._flexInfo

    def setFlexInfo(self, flexInfo):
        self._flexInfo = flexInfo

    def getZFlex(self):
        return np.fromstring(self._zFlex.get(), sep=",")

    def setZFlex(self, zFlex):
        csvZFlex = CsvList()
        for c in zFlex:
            csvZFlex.append(c)
        self._zFlex = csvZFlex

    def getZRed(self):
        return np.fromstring(self._zRed.get(), sep=",")

    def setZRed(self, zRed):
        csvZRed = CsvList()
        for c in zRed:
            csvZRed.append(c)
        self._zRed = csvZRed

    def copyInfo(self, other):
        self.copy(other, copyId=False)


class SetOfParticlesFlex(SetOfParticles):
    """SetOfParticles with flexibility information stored"""
    ITEM_TYPE = ParticleFlex

    def __init__(self, progName="", **kwargs):
        SetOfParticles.__init__(self, **kwargs)
        self._flexInfo = FlexInfo(progName)

    def getFlexInfo(self):
        return self._flexInfo

    def setFlexInfo(self, flexInfo):
        self._flexInfo = flexInfo

    def copyInfo(self, other):
        super(SetOfParticles, self).copyInfo(other)
        if hasattr(other, "_flexInfo"):
            self._flexInfo.copyInfo(other._flexInfo)


class FlexInfo(EMObject):
    """ Object storing any information needed by other protocols/plugins to work properly such us
    neural network paths, basis degrees..."""

    def __init__(self, progName="", **kwargs):
        EMObject.__init__(self, **kwargs)
        self._progName = String(progName)

    def copyInfo(self, other):
        self.copy(other, copyId=False)

    def getProgName(self):
        return self._progName.get()

    def setProgName(self, progName):
        self._progName = String(progName)


class VolumeFlex(Volume):
    """Volume with flexibility information stored"""

    def __init__(self, **kwargs):
        Volume.__init__(self, **kwargs)
        self._flexInfo = FlexInfo()
        self._zFlex = CsvList()
        self._zRed = CsvList()

    def getFlexInfo(self):
        return self._flexInfo

    def setFlexInfo(self, flexInfo):
        self._flexInfo = flexInfo

    def getZFlex(self):
        return np.fromstring(self._zFlex.get(), sep=",")

    def setZFlex(self, zFlex):
        csvZFlex = CsvList()
        for c in zFlex:
            csvZFlex.append(c)
        self._zFlex = csvZFlex

    def getZRed(self):
        return np.fromstring(self._zRed.get(), sep=",")

    def setZRed(self, zRed):
        csvZRed = CsvList()
        for c in zRed:
            csvZRed.append(c)
        self._zRed = csvZRed

    def copyInfo(self, other):
        self.copy(other, copyId=False)


class ClassFlex(SetOfParticlesFlex):
    """Class3D with flexibility information stored"""
    REP_TYPE = VolumeFlex
    def copyInfo(self, other):
        """ Copy basic information (id and other properties) but not
        _mapperPath or _size from other set of micrographs to current one.
        """
        self.copy(other, copyId=False, ignoreAttrs=['_mapperPath', '_size'])

    def clone(self):
        clone = self.getClass()()
        clone.copy(self, ignoreAttrs=['_mapperPath', '_size'])
        return clone

    def close(self):
        # Do nothing on close, since the db will be closed by SetOfClasses
        pass


class SetOfClassesFlex(SetOfClasses):
    """ SetOfClasses3D with flexibility information stored"""
    ITEM_TYPE = ClassFlex
    REP_TYPE = VolumeFlex

    pass

