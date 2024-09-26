<h1 align='center'>Scipion flexutils plugin</h1>

<p align="center">
        
<img alt="Supported Python versions" src="https://img.shields.io/badge/Supported_Python_Versions-3.8_%7C_3.9_%7C_3.10_%7C_3.11_%7C_3.12-blue">
<img alt="GitHub Downloads (all assets, all releases)" src="https://img.shields.io/github/downloads/scipion-em/scipion-em-flexutils/total">
<img alt="GitHub License" src="https://img.shields.io/github/license/scipion-em/scipion-em-flexutils">

</p>

<p align="center">
        
<img alt="Flexutils" width="300" src="https://github.com/scipion-em/scipion-em-flexutils/blob/devel/flexutils/icon.png">

</p>

This plugin contains a series of methods for the visualization and manipulation of flexibility data (specially designed for continuous heterogeneity).


# Plugin Installation


The plugin can be installed in production mode with the following command:

```bash

   scipion installp -p scipion-em-flexutils

```
 
Or in devel mode by executing the commands:

```bash

   git clone https://github.com/scipion-em/scipion-em-flexutils.git
   cd scipion-em-flexutils
   scipion installp -p . --devel

```

The plugin needs PyQt5 to be installed in the device. The following commands can be used to install PyQt5 in different distributions:

```bash
    
   Ubuntu:
   sudo apt-get install python3-pyqt5

   CentOS:
   yum install qt5-qtbase-devel

```

Thee following libraries are also needed by PyQt.

```bash

   Ubuntu:
   sudo apt-get install libxcb-xinerama0
   sudo apt install libxcb-image0

```

The viewers associated to this plugin require OpenGL 3.2 or later and Mesa 11.2 or later.

The installation of the Plugin also requires Conda to be installed in the system. In addition, Scipion needs to know where Conda has been installed. This can be done by adding to your ``scipion.conf`` file the following variable:

```bash

   CONDA_ACTIVATION_CMD = eval "$(/path/to/your/conda/installation/condabin/conda shell.bash hook)"

```

Additionally, the optional component *Open3D* can be installed to add extra functionalities during the network training phase. In order to install this package, the following requirements must be satisfied:

- CUDA must be installed in your system and properly added to the ``PATH`` and ``LD_LIBRARY_PATH`` variables
- You should check the following dependencies are installed in your system:

```bash

   sudo apt install xorg-dev libxcb-shm0 libglu1-mesa-dev python3-dev clang libc++-dev libc++abi-dev libsdl2-dev ninja-build libxi-dev libtbb-dev libosmesa6-dev libudev-dev autoconf libtool

```

If the previous requirements are not met, *Open3D* installation will be just skipped.

# Tutorials

Flexibility Hub tutorials are available in the [Scipion FlexHub Wiki](<https://scipion-em.github.io/docs/release-3.0.0/docs/user/tutorials/flexibilityHub/main_page.html>) and in [Youtube](<https://www.youtube.com/playlist?list=PLuu0votIJpSxTmPLvKRHV3ijadqlxxHfb>).
