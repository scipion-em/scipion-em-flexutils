import os
from setuptools import setup, find_packages


scripts = os.listdir(os.path.join("tensorflow_toolkit", "scripts"))
scripts.remove("__init__.py")
scripts = [os.path.join("tensorflow_toolkit", "scripts", script)
           for script in scripts if ".py" in script]

setup(name='tensorflow_toolkit',
      version="1.0.0",  # Required
      description='Xmipp tensorflow utilities for flexibility',
      author='David Herreros',
      author_email='dherreros@cnb.csic.es',
      keywords='scipion continuous-heterogeneity imageprocessing xmipp',
      packages=find_packages(),
      scripts=scripts)
