=======================
Scipion flexutils plugin
=======================

This plugin contains a series of methods for the visualization and manipulation of flexibility data (specially desgined for continuous heterogeneity).

==========================
Plugin Installation
==========================

This plugin is currently under initial development and it is not ready for production yet. 

In the meantime, it can be used for development, base on Scipion v3.x with plugins. 
 
This tomography plugin can be enabled by cloning this repository and execute the command: 

.. code-block::

    git clone https://github.com/scipion-em/scipion-em-flexutils.git
    scipion installp -p ~/scipion-em-flexutils --devel

The plugin needs PyQt5 to be installed in the device. The following commands can be used to install PyQt5 in different distributions:

.. code-block::
    
    Ubuntu:
    sudo apt-get install python3-pyqt5

    CentOS:
    yum install qt5-qtbase-devel

Thee following libraries are also needed by PyQt.

.. code-block::

    Ubuntu:
    sudo apt-get install libxcb-xinerama0
    sudo apt install libxcb-image0

The viewers associated to this plugin require OpenGL 3.2 or later and Mesa 11.2 or later.
