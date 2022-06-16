# -*- coding: utf-8 -*-
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


import os

import pyworkflow.plugin as pwplugin

import flexutils
from flexutils.constants import CONDA_REQ


__version__ = "3.0.1"
_logo = "icon.png"
_references = []


class Plugin(pwplugin.Plugin):

    @classmethod
    def getEnvActivation(cls):
        return "conda activate flexutils"

    @classmethod
    def getProgram(cls, program, python=True):
        """ Return the program binary that will be used. """
        cmd = '%s %s && ' % (cls.getCondaActivationCmd(), cls.getEnvActivation())
        if python:
            cmd += 'python '
        return cmd + '%(program)s ' % locals()

    @classmethod
    def getCommand(cls, program, args, python=True):
        return cls.getProgram(program, python) + args

    @classmethod
    def defineBinaries(cls, env):
        def getCondaInstallation():
            installationCmd = cls.getCondaActivationCmd()
            installationCmd += 'conda create -y -n flexutils python==3.8.5 pip && '
            installationCmd += "conda activate flexutils && pip install -r " + CONDA_REQ + " && "
            installationCmd += "pip install -e %s" % (os.path.join(flexutils.__path__[0], ".."))
            return installationCmd

        commands = []
        installationEnv = getCondaInstallation()
        commands.append((installationEnv, []))

        env.addPackage('flexutils', version=__version__,
                       commands=commands,
                       tar="void.tgz",
                       default=True)
