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

import importlib

import pyworkflow.plugin as pwplugin
import pyworkflow.utils as pwutils

from pwem import Config as emConfig

import flexutils
from flexutils.constants import CONDA_YML


__version__ = "3.1.5"
_logo = "icon.png"
_references = []


class Plugin(pwplugin.Plugin):

    @classmethod
    def getEnvActivation(cls):
        return "conda activate flexutils"

    @classmethod
    def getTensorflowActivation(cls):
        return "conda activate flexutils-tensorflow"

    @classmethod
    def getProgram(cls, program, python=False, cuda=False, chimera=False, needsPackages=None):
        """ Return the program binary that will be used. """
        scipion_packages = []
        env_variables = ""
        if needsPackages is not None:
            for package_name in needsPackages:
                package_name = package_name.lower()
                try:
                    package = importlib.import_module(package_name)
                    package_loc = importlib.util.find_spec(package_name).submodule_search_locations[0]
                    scipion_packages.append(os.path.dirname(package_loc))
                except ImportError:
                    raise ImportError(f"Package {package_name} is not installed in Scipion")
                package_env_vars = package.Plugin.getVars()
                for item, value in package_env_vars.items():
                    if package_name.lower() in item.lower():
                        env_variables += " {}='{}'".format(item, value)
        scipion_packages = ":".join(scipion_packages)
        cmd = '%s %s && ' % (cls.getCondaActivationCmd(), cls.getEnvActivation())

        if cuda:
            cmd += 'LD_LIBRARY_PATH=$CONDA_PREFIX/lib/:$LD_LIBRARY_PATH '

        if chimera:
            with pwutils.weakImport("chimera"):
                from chimera import Plugin as chimeraPlugin
                cmd += "CHIMERA_HOME=%s " % chimeraPlugin.getHome()

        if python:
            if needsPackages is not None:
                cmd += "PYTHONPATH=$PYTHONPATH:%s %s python " % (scipion_packages, env_variables)
            else:
                cmd += "python "
        return cmd + '%(program)s ' % locals()

    @classmethod
    def getTensorflowProgram(cls, program, python=True, log_level=2):
        cmd = '%s %s && ' % (cls.getCondaActivationCmd(), cls.getTensorflowActivation())
        if python:
            # import pyworkflow, pwem, xmipp3
            # pyworkflow_path = os.path.join(pyworkflow.__path__[0], "..")
            # pywem_path = os.path.join(pwem.__path__[0], "..")
            # xmipp3_path = os.path.join(xmipp3.__path__[0], "..")
            # paths = [os.path.join(flexutils.__path__[0], ".."), pyworkflow_path, pywem_path, xmipp3_path]
            return cmd + ' TF_CPP_MIN_LOG_LEVEL=%(log_level)d python %(program)s ' % locals()
        else:
            return cmd + ' TF_CPP_MIN_LOG_LEVEL=%(log_level)d %(program)s ' % locals()

    @classmethod
    def getCommand(cls, program, args, python=True):
        return cls.getProgram(program, python) + args

    def defineBinaries(cls, env):
        def getCondaInstallationFlexutils():
            installationCmd = f'if [ $(basename "$PWD") = flexutils-{__version__} ]; then cd ..; fi && '
            installationCmd += "git clone -b devel https://github.com/I2PC/Flexutils-Scripts.git && "
            installationCmd += "cd Flexutils-Scripts && "
            installationCmd += "bash install.sh && touch flexutils_installed && cd .."
            return installationCmd

        def getCondaInstallationTensorflow():
            conda_init = cls.getCondaActivationCmd()
            branch = "devel" if cls.inDevelMode() else "master"
            installationCmd = f'if [ $(basename "$PWD") = flexutils-{__version__} ]; then cd ..; fi && '
            installationCmd += f"{conda_init} conda activate flexutils && " \
                               f' if [ ! -d "Flexutils-Toolkit" ]; then git clone -b {branch} https://github.com/I2PC/Flexutils-Toolkit.git; fi && ' \
                               f"cd Flexutils-Toolkit && " \
                               f"bash install.sh && touch flexutils_tensorflow_installed && cd .."
            return installationCmd

        def getUpdateCommands():
            conda_init = cls.getCondaActivationCmd()
            updateCmd = f'if [ $(basename "$PWD") = flexutils-{__version__} ]; then cd ..; fi && '
            updateCmd += f"{conda_init} conda activate flexutils && "
            updateCmd += "echo '###### Updating scripts.... ######' && "
            updateCmd += "cd Flexutils-Scripts && "
            updateCmd += "git pull && "
            updateCmd += "pip install -e . && "
            updateCmd += "cd .. && "
            updateCmd += "echo '###### Script updated succesfully! ######' && "

            updateCmd += "echo '###### Updating NN binaries.... ######' && "
            updateCmd += "cd Flexutils-Toolkit && "
            updateCmd += "git pull && "
            updateCmd += "bash install.sh && "
            updateCmd += "cd .. && "
            updateCmd += "echo '###### Binaries updated succesfully! ######' && "
            updateCmd += f"cd flexutils-{__version__} && "
            updateCmd += "touch flexutils_updated"
            return updateCmd

        binary_path = os.path.join(emConfig.EM_ROOT, f'flexutils-{__version__}')
        commands = []

        if not os.path.isfile(os.path.join(binary_path, os.path.join("..", "Flexutils-Scripts", "flexutils_installed"))):
            installationEnv = getCondaInstallationFlexutils()
            commands.append((installationEnv, [os.path.join("..", "Flexutils-Scripts", "flexutils_installed")]))

        if not os.path.isfile(os.path.join(binary_path, os.path.join("..", "Flexutils-Toolkit", "flexutils_tensorflow_installed"))):
            installationTensorflow = getCondaInstallationTensorflow()
            commands.append((installationTensorflow, [os.path.join("..", "Flexutils-Toolkit", "flexutils_tensorflow_installed")]))

        if os.path.isfile(os.path.join(binary_path, "flexutils_tensorflow_updated")):
            os.remove(os.path.join(binary_path, "flexutils_tensorflow_updated"))
        commands.append((getUpdateCommands(), ["flexutils_updated"]))

        env.addPackage('flexutils', version=__version__,
                       commands=commands,
                       tar="void.tgz",
                       default=True)
