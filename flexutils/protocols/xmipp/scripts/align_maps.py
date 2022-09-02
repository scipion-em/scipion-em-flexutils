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


import flexutils.protocols.xmipp.utils.utils as utl


def alignMaps(file_input, file_target, file_output, global_search=None):
    # Align maps in ChimeraX using fitmap
    _ = utl.alignMapsChimeraX(file_input, file_target, global_search=global_search,
                              output_map=file_output)


if __name__ == '__main__':
    import argparse

    # Input parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--i', type=str, required=True)
    parser.add_argument('--r', type=str, required=True)
    parser.add_argument('--o', type=str, required=True)
    parser.add_argument('--gs', type=int)

    args = parser.parse_args()

    if args.gs is None:
        gs = None
    else:
        gs = args.gs

    # Initialize volume slicer
    alignMaps(args.i, args.r, args.o, gs)