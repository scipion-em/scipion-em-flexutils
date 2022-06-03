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
from scipy.ndimage import binary_dilation, binary_fill_holes
from skimage.morphology import ball

from pwem.emlib.image import ImageHandler


def floodFillMask(infile, outfile):
    data = ImageHandler().read(infile).getData()
    ball_kernel = ball(2)
    for _ in range(10):
        data = binary_dilation(data, ball_kernel)
    data = binary_fill_holes(data, ball_kernel)
    filled_vol = ImageHandler().createImage()
    filled_vol.setData(data.astype(np.float32))
    filled_vol.write(outfile)


if __name__ == '__main__':
    import argparse

    # Input parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)

    args = parser.parse_args()

    # Initialize volume slicer
    floodFillMask(args.input, args.output)
