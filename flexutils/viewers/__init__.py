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
from .viewers_data import FlexDataViewer

# from .viewer_morph_salesman import FlexMorphSalesmanViewer
from .viewer_show_structures import FlexShowStructuresViewer
from flexutils.viewers.map_model_viewers.viewer_show_maps import FlexShowMapsViewer
from flexutils.viewers.clustering_viewers.viewer_optimal_clusters import FlexShowOptimalClustersViewer
from flexutils.viewers.tensorboard.viewer_tensorboard import TensorboardViewer

# Flexibility Xmipp viewers (offered here due to strong package dependency)
from .xmipp import *
