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

import site

import glob

import importlib

import pyworkflow.plugin as pwplugin
import pyworkflow.utils as pwutils

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
        scipion_packages = site.getsitepackages()[0]
        flexutils_packages = scipion_packages.replace("scipion3/", "flexutils/")
        scipion_packages = glob.glob(os.path.join(scipion_packages, "scipion-em-*"))
        flexutils_packages = glob.glob(os.path.join(flexutils_packages, "scipion-em-*"))
        flexutils_packages = [package.replace("flexutils/", "scipion3/") for package in flexutils_packages]
        set_dif = set(scipion_packages).symmetric_difference(set(flexutils_packages))
        scipion_packages = list(set_dif)
        env_variables = ""
        for idx in range(len(scipion_packages)):
            if "egg-link" in scipion_packages[idx]:
                with open(scipion_packages[idx], "r") as file:
                    lines = file.readlines()
                scipion_packages[idx] = lines[0].strip("\n")
                package = os.path.basename(scipion_packages[idx])
                if "scipion-em-" in package:
                    package_name = package.replace("scipion-em-", "")
                    package = importlib.import_module(package_name)
                    package_env_vars = package.Plugin.getVars()
                    for item, value in package_env_vars.items():
                        if package_name.lower() in item.lower():
                            env_variables += " {}='{}'".format(item, value)
        scipion_packages = ":".join(scipion_packages)
        cmd = '%s %s && ' % (cls.getCondaActivationCmd(), cls.getEnvActivation())

        if python:
            with pwutils.weakImport("chimera"):
                from chimera import Plugin as chimeraPlugin
                cmd += "CHIMERA_HOME=%s " % chimeraPlugin.getHome()
            cmd += "PYTHONPATH=%s %s python " % (scipion_packages, env_variables)
        return cmd + '%(program)s ' % locals()

    @classmethod
    def getCommand(cls, program, args, python=True):
        return cls.getProgram(program, python) + args

    @classmethod
    def defineBinaries(cls, env):
        def getCondaInstallation():
            installationCmd = cls.getCondaActivationCmd()
            if 'CONDA_DEFAULT_ENV' in os.environ:
                installationCmd += 'conda create -y -n flexutils --clone %s && ' % os.environ['CONDA_DEFAULT_ENV']
            elif 'VIRTUAL_ENV' in os.environ:
                installationCmd += 'conda create -y -n flexutils --clone %s && ' % os.environ['VIRTUAL_ENV']
            installationCmd += "conda activate flexutils && conda install -c anaconda cudatoolkit -y && " \
                               "conda install -c conda-forge cudatoolkit-dev -y && pip install -r " + CONDA_REQ + " && "
            installationCmd += "pip install -e %s" % (os.path.join(flexutils.__path__[0], ".."))
            return installationCmd

        commands = []
        installationEnv = getCondaInstallation()
        commands.append((installationEnv, []))

        env.addPackage('flexutils', version=__version__,
                       commands=commands,
                       tar="void.tgz",
                       default=True)
