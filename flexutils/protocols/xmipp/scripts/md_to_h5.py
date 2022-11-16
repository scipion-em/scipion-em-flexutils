# **************************************************************************
# *
# * Authors:  David Herreros Calero (dherreros@cnb.csic.es)
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


import numpy as np
import h5py
import os
from joblib import Parallel, delayed
from tqdm import tqdm

import pwem.emlib.metadata as md

from pyworkflow.utils import runJob

import xmipp3


def ConvertMDToH5(metadata_file, outPath, sr, applyCTF, unStack, volume, mask, thr=4):
    # Read metadata file
    metadata = md.MetaData(metadata_file)
    metadata.sort()

    img_files = metadata.getColumnValues(md.MDL_IMAGE)
    rot = metadata.getColumnValues(md.MDL_ANGLE_ROT)
    tilt = metadata.getColumnValues(md.MDL_ANGLE_TILT)
    psi = metadata.getColumnValues(md.MDL_ANGLE_PSI)
    shift_x = metadata.getColumnValues(md.MDL_SHIFT_X)
    shift_y = metadata.getColumnValues(md.MDL_SHIFT_Y)
    defocusU = metadata.getColumnValues(md.MDL_CTF_DEFOCUSU)
    defocusV = metadata.getColumnValues(md.MDL_CTF_DEFOCUSV)
    defocusAngle = metadata.getColumnValues(md.MDL_CTF_DEFOCUS_ANGLE)
    cs = metadata.getColumnValues(md.MDL_CTF_CS)
    kv = metadata.getColumnValues(md.MDL_CTF_VOLTAGE)[0]

    # Unstack of convert to mrc
    if unStack:
        def unStackImages(iid):
            theo_file = os.path.join(outPath, "theo_%d.mrc" % iid)
            runJob(None, "xmipp_image_convert",
                   "-i %s -o %s " % (img_files[iid], theo_file),
                   numberOfMpi=1, env=xmipp3.Plugin.getEnviron())

        with Parallel(n_jobs=thr) as parallel:
            _ = parallel(delayed(unStackImages)(iid) for iid in tqdm(range(len(img_files))))
    else:
        stack = img_files[0].split("@")[1]
        runJob(None, "xmipp_image_convert",
               "-i %s -o %s " % (stack, os.path.join(outPath, "stack.mrc")),
               numberOfMpi=1, env=xmipp3.Plugin.getEnviron())

    # Copy mask and volume to output folder
    runJob(None, "xmipp_image_convert",
           "-i %s -o %s " % (volume, os.path.join(outPath, "volume.mrc")),
           numberOfMpi=1, env=xmipp3.Plugin.getEnviron())
    runJob(None, "xmipp_image_convert",
           "-i %s -o %s " % (mask, os.path.join(outPath, "mask.mrc")),
           numberOfMpi=1, env=xmipp3.Plugin.getEnviron())

    # Save to H5 file
    hf = h5py.File(os.path.join(outPath, "metadata.h5"), 'w')
    hf.create_dataset('images_path', data=np.array([outPath], dtype='S'))
    hf.create_dataset('mask', data=np.array([os.path.join(outPath, "mask.mrc")], dtype='S'))
    hf.create_dataset('volume', data=np.array([os.path.join(outPath, "volume.mrc")], dtype='S'))
    hf.create_dataset('angle_rot', data=rot)
    hf.create_dataset('angle_tilt', data=tilt)
    hf.create_dataset('angle_psi', data=psi)
    hf.create_dataset('shift_x', data=shift_x)
    hf.create_dataset('shift_y', data=shift_y)
    hf.create_dataset('defocusU', data=defocusU)
    hf.create_dataset('defocusV', data=defocusV)
    hf.create_dataset('defocusAngle', data=defocusAngle)
    hf.create_dataset('cs', data=cs)
    hf.create_dataset('kv', data=[kv])
    hf.create_dataset('sr', data=[sr])

    if applyCTF:
        hf.create_dataset('applyCTF', data=[1.0])
    else:
        hf.create_dataset('applyCTF', data=[0.0])
    hf.close()


if __name__ == '__main__':
    import argparse

    # Input parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--md_file', type=str, required=True)
    parser.add_argument('--out_path', type=str, required=True)
    parser.add_argument('--sr', type=float, required=True)
    parser.add_argument('--applyCTF', action='store_true')
    parser.add_argument('--unStack', action='store_true')
    parser.add_argument('--volume', type=str, required=True)
    parser.add_argument('--mask', type=str, required=True)
    parser.add_argument('--thr', type=int)

    args = parser.parse_args()

    inputs = {"metadata_file": args.md_file, "outPath": args.out_path, "sr": args.sr,
              "applyCTF": args.applyCTF, "unStack": args.unStack, "volume": args.volume,
              "mask": args.mask}

    if args.thr:
        inputs["thr"] = args.thr

    # Initialize volume slicer
    ConvertMDToH5(**inputs)
