# **************************************************************************
# *
# * Authors:     David Herreros (dherreros@cnb.csic.es)
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
from pyworkflow.utils import weakImport

from .protocol_reconstruct_zart import XmippProtReconstructZART
from .protocol_match_and_deform_structure_zernike3d import XmippMatchDeformSructZernike3D
from .protocol_match_and_deform_map_zernike3d import XmippMatchDeformMapZernike3D
from .protocol_assign_heterogeneity_priors_zernike3d import XmippProtHeterogeneityPriorsZernike3D
from .protocol_angular_alignment_zernike3d import XmippProtAngularAlignmentZernike3D
from .protocol_focus_zernike3d import XmippProtFocusZernike3D
from .protocol_reassign_reference_zernike3d import XmippProtReassignReferenceZernike3D
# from .protocol_compute_priors_zernike3d import XmippProtComputeHeterogeneityPriorsZernike3D
from .protocol_statistics_zernike3d import XmippProtStatisticsZernike3D
from .protocol_structure_landscape import XmippProtStructureLanscapes
from .protocol_cluster_structures_zernike3d import XmippProtClusterStructuresZernike3D
from .protocol_apply_field_zernike3d import XmippApplyFieldZernike3D
# from .protocol_apply_field_nma import XmippApplyFieldNMA
from .protocol_prepare_volumes_zernike3deep import XmippProtPrepareZernikeVolumes

# Weak imports to avoid library incompatibilities in viewers due to Xmipp environment (this avoids loading
# the Xmipp env and the Scipion libraries that interfere with flexutils ones)
with weakImport("xmipp3.base"):
    from .protocol_resize_zernike_data import XmippProtCropResizeZernikeParticles, XmippProtCropResizeZernikeVolumes

from .protocol_deform_map_zernike3d import ProtFlexVolumeDeformZernike3D
# from .protocol_structure_map_zernike3d import XmippProtStructureMapZernike3D

from .protocol_angular_align_zernike3deep import TensorflowProtAngularAlignmentZernike3Deep
from .protocol_predict_zernike3deep import TensorflowProtPredictZernike3Deep
from .protocol_angular_align_deep_pose import TensorflowProtAngularAlignmentDeepPose
from .protocol_predict_deep_pose import TensorflowProtPredictDeepPose
# from .protocol_angular_align_deep_nma import TensorflowProtAngularAlignmentDeepNMA
# from .protocol_predict_deep_nma import TensorflowProtPredictDeepNMA
from .protocol_angular_align_homo_siren import TensorflowProtAngularAlignmentHomoSiren
from .protocol_predict_homo_siren import TensorflowProtPredictHomoSiren
from .protocol_angular_align_het_siren import TensorflowProtAngularAlignmentHetSiren
from .protocol_predict_het_siren import TensorflowProtPredictHetSiren
from .protocol_denoise_particles_het_siren import TensorflowProtDenoiseParticlesHetSiren
from .protocol_train_flex_consensus import TensorflowProtTrainFlexConsensus
from .protocol_interactive_flex_consensus import TensorflowProtInteractiveFlexConsensus
