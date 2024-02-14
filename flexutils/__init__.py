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


__version__ = "3.1.1"
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
    def getProgram(cls, program, python=True, cuda=False, needsPackages=None):
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

        if python:
            with pwutils.weakImport("chimera"):
                from chimera import Plugin as chimeraPlugin
                cmd += "CHIMERA_HOME=%s " % chimeraPlugin.getHome()

            with pwutils.weakImport("xmipp3"):
                import xmipp3
                cmd += "XMIPP_HOME=%s " % xmipp3.Plugin.getHome()

            cmd += "SCIPION_CANCEL_XMIPP_BINDING_WARNING=1 "

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

    @classmethod
    def defineBinaries(cls, env):
        def getCondaInstallationFlexutils():
            installationCmd = cls.getCondaActivationCmd()
            installationCmd += 'conda env remove -n flexutils && conda env create -f ' + CONDA_YML + " && "
            installationCmd += "conda activate flexutils && "
            installationCmd += "pip install -e %s --no-dependencies && " % (os.path.join(flexutils.__path__[0], ".."))
            installationCmd += "mkdir -p $CONDA_PREFIX/etc/conda/activate.d && "
            installationCmd += ("echo \'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/\\n\' "
                                ">> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh && ")
            installationCmd += ("echo \'export XLA_FLAGS=--xla_gpu_cuda_data_dir=$CONDA_PREFIX/lib/\\n\' >> " 
                               "$CONDA_PREFIX/etc/conda/activate.d/env_vars.sh && ")
            installationCmd += "mkdir -p $CONDA_PREFIX/lib/nvvm/libdevice && "
            installationCmd += "cp $CONDA_PREFIX/lib/libdevice.10.bc $CONDA_PREFIX/lib/nvvm/libdevice/ && "
            installationCmd += "touch flexutils_installed"
            return installationCmd

        def getCondaInstallationTensorflow():
            conda_init = cls.getCondaActivationCmd()
            installationCmd = f"{conda_init} conda activate flexutils && " \
                              f"git clone -b master https://github.com/I2PC/Flexutils-Toolkit.git && " \
                              f"cd Flexutils-Toolkit && " \
                              f"bash install.sh && cd .. && "
            installationCmd += "touch flexutils_tensorflow_installed"
            return installationCmd

        def getUpdateCommands():
            conda_init = cls.getCondaActivationCmd()
            updateCmd = f"{conda_init} conda activate flexutils && "
            updateCmd += "echo '###### Updating NN binaries.... ######' && "
            updateCmd += "cd Flexutils-Toolkit && "
            updateCmd += "git pull && "
            updateCmd += "bash install.sh && "
            updateCmd += "cd .. && "
            updateCmd += "echo '###### Binaries updated succesfully! ######' && "
            updateCmd += "touch flexutils_tensorflow_updated"
            return updateCmd

        binary_path = os.path.join(emConfig.EM_ROOT, f'flexutils-{__version__}')
        commands = []

        if not os.path.isfile(os.path.join(binary_path, "flexutils_installed")):
            installationEnv = getCondaInstallationFlexutils()
            commands.append((installationEnv, ["flexutils_installed"]))

        if not os.path.isfile(os.path.join(binary_path, "flexutils_tensorflow_installed")):
            installationTensorflow = getCondaInstallationTensorflow()
            commands.append((installationTensorflow, ["flexutils_tensorflow_installed"]))

        if os.path.isfile(os.path.join(binary_path, "flexutils_tensorflow_updated")):
            os.remove(os.path.join(binary_path, "flexutils_tensorflow_updated"))
        commands.append((getUpdateCommands(), ["flexutils_tensorflow_updated"]))

        env.addPackage('flexutils', version=__version__,
                       commands=commands,
                       tar="void.tgz",
                       default=True)
