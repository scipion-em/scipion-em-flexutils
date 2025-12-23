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


import pyworkflow.viewer as pwviewer

import pwem.viewers.views as vi
from pwem.viewers import DataViewer, MODE, MODE_MD

import pwem.objects.data_flexhub as pwobj


class FlexDataViewer(pwviewer.Viewer):
    """
    Wrapper to visualize different type of objects
    with the Xmipp program xmipp_showj
    """
    _environments = [pwviewer.DESKTOP_TKINTER]
    _targets = [
        pwobj.SetOfClassesFlex,
        pwobj.SetOfClassesStructFlex
    ]

    def __init__(self, **kwargs):
        pwviewer.Viewer.__init__(self, **kwargs)
        self._views = []

    def _getObjView(self, obj, fn, viewParams={}):
        return vi.ObjectView(
            self._project, obj.strId(), fn, viewParams=viewParams)

    def _visualize(self, obj, **kwargs):
        views = []
        cls = type(obj)

        if issubclass(cls, pwobj.SetOfClassesFlex):
            views.append(vi.Classes3DView(self._project, obj.strId(),
                                          obj.getFileName()))

        elif issubclass(cls, pwobj.SetOfClassesStructFlex):
            views.append(vi.Classes3DView(self._project, obj.strId(),
                                          obj.getFileName()))

        return views


# Register specific sets in pwem dataviewer.
if hasattr(DataViewer, "registerConfig"):
    DataViewer.registerConfig(pwobj.SetOfParticlesFlex,
                              config={MODE: MODE_MD})
    DataViewer.registerConfig(pwobj.SetOfVolumesFlex,
                              config={MODE: MODE_MD})
    DataViewer.registerConfig(pwobj.SetOfAtomStructFlex,
                              config={MODE: MODE_MD})
