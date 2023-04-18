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


############## Imports ##############
import os
from subprocess import call
import numpy as np
from numpy import sin, cos, sqrt
from numpy import arctan2 as atan2
from scipy.ndimage import binary_erosion, distance_transform_edt, binary_closing, gaussian_filter
from scipy.spatial import KDTree
from skimage import filters
from skimage.measure import label
from skimage.morphology import ball, convex_hull_image, skeletonize
from subprocess import call
from xmipp_metadata.image_handler import ImageHandler

import pyworkflow.utils as pwutils

from pwem.convert.transformations import superimposition_matrix
from pwem.viewers import Chimera

import xmipp3
#####################################


############## Zernike related functions ##############
def computeZernikes3D(l1, n, l2, m, pos, r_max):

    # General variables
    pos_r = pos / r_max
    xr, yr, zr = pos_r[:, 0], pos_r[:, 1], pos_r[:, 2]
    xr2, yr2, zr2 = xr * xr, yr * yr, zr * zr
    r = np.linalg.norm(pos, axis=1) / r_max
    r2 = r * r

    # Variables needed for l2 >= 5
    tht = atan2(yr, xr)
    phi = atan2(zr, sqrt(xr2 + yr2))
    sinth = sin(abs(m)*phi)
    costh = cos(tht)
    cosph = cos(abs(m)*phi)
    sinth2 = sinth*sinth
    costh2 = costh*costh

    # Zernike Polynomials
    if l1 == 0:
        R = np.asarray([np.sqrt(3)] * len(xr))
    elif l1 == 1:
        R = np.sqrt(5) * r
    elif l1 == 2:
        if n == 0:
            R = -0.5 * np.sqrt(7) * (2.5 * (1 - 2 * r2) + 0.5)
        elif n == 2:
            R = np.sqrt(7) * r2
    elif l1 == 3:
        if n == 1:
            R = -1.5 * r * (3.5 * (1 - 2 * r2) + 1.5)
        elif n == 3:
            R = 3 * r2 * r
    elif l1 == 4:
        if n == 0:
            R = np.sqrt(11) * ((63 * r2 * r2 / 8) - (35 * r2 / 4) + (15 / 8))
        elif n == 2:
            R = -0.5 * np.sqrt(11) * r2 * (4.5 * (1 - 2 * r2) + 2.5)
        elif n == 4:
            R = np.sqrt(11) * r2 * r2
    elif l1 == 5:
        if n == 1:
            R = np.sqrt(13) * r * ((99 * r2 * r2 / 8) - (63 * r2 / 4) + (35 / 8))
        elif n == 3:
            R = -0.5 * np.sqrt(13) * r2 * r * (5.5 * (1 - 2 * r2) + 3.5)
        elif n == 5:
            R = np.sqrt(13) * r2 * r2 * r
    elif l1 == 6:
        if n == 0:
            R = 103.8 * r2 * r2 * r2 - 167.7 * r2 * r2 + 76.25 * r2 - 8.472
        elif n == 2:
            R = 69.23 * r2 * r2 * r2 - 95.86 * r2 * r2 + 30.5 * r2
        elif n == 4:
            R = 25.17 * r2 * r2 * r2 - 21.3 * r2 * r2
        elif n == 6:
            R = 3.873 * r2 * r2 * r2
    elif l1 == 7:
        if n == 1:
            R = 184.3 * r2 * r2 * r2 * r - 331.7 * r2 * r2 * r + 178.6 * r2 * r - 27.06 * r
        elif n == 3:
            R = 100.5 * r2 * r2 * r2 * r - 147.4 * r2 * r2 * r + 51.02 * r2 * r
        elif n == 5:
            R = 30.92 * r2 * r2 * r2 * r - 26.8 * r2 * r2 * r
        elif n == 7:
            R = 4.123 * r2 * r2 * r2 * r
    elif l1 == 8:
        if n == 0:
            R = 413.9*r2*r2*r2*r2 - 876.5*r2*r2*r2 + 613.6*r2*r2 - 157.3*r2 + 10.73
        if n == 2:
            R = 301.0*r2*r2*r2*r2 - 584.4*r2*r2*r2 + 350.6*r2*r2 - 62.93*r2
        if n == 4:
            R = 138.9*r2*r2*r2*r2 - 212.5*r2*r2*r2 + 77.92*r2*r2
        if n == 6:
            R = 37.05*r2*r2*r2*r2 - 32.69*r2*r2*r2
        if n == 8:
            R = 4.359*r2*r2*r2*r2
    elif l1 == 9:
        if n == 1:
            R = 751.6*r2*r2*r2*r2*r - 1741.0*r2*r2*r2*r + 1382.0*r2*r2*r - 430.0*r2*r + 41.35*r
        if n == 3:
            R = 462.6*r2*r2*r2*r2*r - 949.5*r2*r2*r2*r + 614.4*r2*r2*r - 122.9*r2*r
        if n == 5:
            R = 185.0*r2*r2*r2*r2*r - 292.1*r2*r2*r2*r + 111.7*r2*r2*r
        if n == 7:
            R = 43.53*r2*r2*r2*r2*r - 38.95*r2*r2*r2*r
        if n == 9:
            R = 4.583*r2*r2*r2*r2*r
    elif l1 == 10:
        if n == 0:
            R = 1652.0*r2*r2*r2*r2*r2 - 4326.0*r2*r2*r2*r2 + 4099.0*r2*r2*r2 - 1688.0*r2*r2 + 281.3*r2 - 12.98
        if n == 2:
            R = 1271.0*r2*r2*r2*r2*r2 - 3147.0*r2*r2*r2*r2 + 2732.0*r2*r2*r2 - 964.4*r2*r2 + 112.5*r2
        if n == 4:
            R = 677.7*r2*r2*r2*r2*r2 - 1452.0*r2*r2*r2*r2 + 993.6*r2*r2*r2 - 214.3*r2*r2
        if n == 6:
            R = 239.2*r2*r2*r2*r2*r2 - 387.3*r2*r2*r2*r2 + 152.9*r2*r2*r2
        if n == 8:
            R = 50.36*r2*r2*r2*r2*r2 - 45.56*r2*r2*r2*r2
        if n == 10:
            R = 4.796*r2*r2*r2*r2*r2
    elif l1 == 11:
        if n == 1:
            R = r*-5.865234375E+1+(r*r*r)*8.7978515625E+2-(r*r*r*r*r)*4.2732421875E+3+(r*r*r*r*r*r*r)*9.0212890625E+3-(r*r*r*r*r*r*r*r*r)*8.61123046875E+3+np.power(r,1.1E+1)*3.04705078125E+3
        if n == 3:
            R = (r*r*r)*2.513671875E+2-(r*r*r*r*r)*1.89921875E+3+(r*r*r*r*r*r*r)*4.920703125E+3-(r*r*r*r*r*r*r*r*r)*5.29921875E+3+np.power(r,1.1E+1)*2.0313671875E+3
        if n == 5:
            R = (r*r*r*r*r)*-3.453125E+2+(r*r*r*r*r*r*r)*1.5140625E+3-(r*r*r*r*r*r*r*r*r)*2.1196875E+3+np.power(r,1.1E+1)*9.559375E+2
        if n == 7:
            R = (r*r*r*r*r*r*r)*2.01875E+2-(r*r*r*r*r*r*r*r*r)*4.9875E+2+np.power(r,1.1E+1)*3.01875E+2
        if n == 9:
            R = (r*r*r*r*r*r*r*r*r)*-5.25E+1+np.power(r,1.1E+1)*5.75E+1
        if n == 11:
            R = np.power(r,1.1E+1)*5.0
    elif l1 == 12:
        if n == 0:
            R = (r*r)*-4.57149777110666E+2+(r*r*r*r)*3.885773105442524E+3-(r*r*r*r*r*r)*1.40627979054153E+4+(r*r*r*r*r*r*r*r)*2.460989633446932E+4-np.power(r,1.0E+1)*2.05828223888278E+4+np.power(r,1.2E+1)*6.597058457955718E+3+1.523832590368693E+1
        if n == 2:
            R = (r*r)*-1.828599108443595E+2+(r*r*r*r)*2.220441774539649E+3-(r*r*r*r*r*r)*9.375198603600264E+3+(r*r*r*r*r*r*r*r)*1.789810642504692E+4-np.power(r,1.0E+1)*1.583294029909372E+4+np.power(r,1.2E+1)*5.277646766364574E+3
        if n == 4:
            R = (r*r*r*r)*4.934315054528415E+2-(r*r*r*r*r*r)*3.409163128584623E+3+(r*r*r*r*r*r*r*r)*8.260664503872395E+3-np.power(r,1.0E+1)*8.444234826177359E+3+np.power(r,1.2E+1)*3.104498097866774E+3
        if n == 6:
            R = (r*r*r*r*r*r)*-5.244866351671517E+2+(r*r*r*r*r*r*r*r)*2.202843867704272E+3-np.power(r,1.0E+1)*2.98031817394495E+3+np.power(r,1.2E+1)*1.307157093837857E+3
        if n == 8:
            R = (r*r*r*r*r*r*r*r)*2.591581020820886E+2-np.power(r,1.0E+1)*6.274354050420225E+2+np.power(r,1.2E+1)*3.734734553815797E+2
        if n == 10:
            R = np.power(r,1.0E+1)*-5.975575286115054E+1+np.power(r,1.2E+1)*6.49519052838441E+1
        if n == 12:
            R = np.power(r,1.2E+1)*5.19615242270811
    elif l1 == 13:
        if n == 1:
            R = r*7.896313435467891E+1-(r*r*r)*1.610847940832376E+3+(r*r*r*r*r)*1.093075388422608E+4-(r*r*r*r*r*r*r)*3.400678986203671E+4+(r*r*r*r*r*r*r*r*r)*5.332882955634594E+4-np.power(r,1.1E+1)*4.102217658185959E+4+np.power(r,1.3E+1)*1.230665297454596E+4
        if n == 3:
            R = (r*r*r)*-4.602422688100487E+2+(r*r*r*r*r)*4.858112837433815E+3-(r*r*r*r*r*r*r)*1.854915810656548E+4+(r*r*r*r*r*r*r*r*r)*3.281774126553535E+4-np.power(r,1.1E+1)*2.734811772125959E+4+np.power(r,1.3E+1)*8.687049158513546E+3
        if n == 5:
            R = (r*r*r*r*r)*8.832932431697845E+2-(r*r*r*r*r*r*r)*5.707433263555169E+3+(r*r*r*r*r*r*r*r*r)*1.312709650617838E+4-np.power(r,1.1E+1)*1.286970245704055E+4+np.power(r,1.3E+1)*4.572131136059761E+3
        if n == 7:
            R = (r*r*r*r*r*r*r)*-7.60991101808846E+2+(r*r*r*r*r*r*r*r*r)*3.088728589691222E+3-np.power(r,1.1E+1)*4.064116565383971E+3+np.power(r,1.3E+1)*1.741764242306352E+3
        if n == 9:
            R = (r*r*r*r*r*r*r*r*r)*3.251293252306059E+2-np.power(r,1.1E+1)*7.741174410264939E+2+np.power(r,1.3E+1)*4.54373280601576E+2
        if n == 11:
            R = np.power(r,1.1E+1)*-6.731456008902751E+1+np.power(r,1.3E+1)*7.269972489634529E+1
        if n == 13:
            R = np.power(r,1.3E+1)*5.385164807128604
    elif l1 == 14:
        if n == 0:
            R = (r*r)*6.939451623205096E+2-(r*r*r*r)*7.910974850460887E+3+(r*r*r*r*r*r)*3.955487425231934E+4-(r*r*r*r*r*r*r*r)*1.010846786448956E+5+np.power(r,1.0E+1)*1.378427436065674E+5-np.power(r,1.2E+1)*9.542959172773361E+4+np.power(r,1.4E+1)*2.63567443819046E+4-1.749441585683962E+1
        if n == 2:
            R = (r*r)*2.775780649287626E+2-(r*r*r*r)*4.520557057410479E+3+(r*r*r*r*r*r)*2.636991616821289E+4-(r*r*r*r*r*r*r*r)*7.351612992358208E+4+np.power(r,1.0E+1)*1.060328796973228E+5-np.power(r,1.2E+1)*7.634367338204384E+4+np.power(r,1.4E+1)*2.170555419689417E+4
        if n == 4:
            R = (r*r*r*r)*-1.004568234980106E+3+(r*r*r*r*r*r)*9.589060424804688E+3-(r*r*r*r*r*r*r*r)*3.393052150321007E+4+np.power(r,1.0E+1)*5.655086917197704E+4-np.power(r,1.2E+1)*4.490804316592216E+4+np.power(r,1.4E+1)*1.370877107170224E+4
        if n == 6:
            R = (r*r*r*r*r*r)*1.475240065354854E+3-(r*r*r*r*r*r*r*r)*9.04813906750083E+3+np.power(r,1.0E+1)*1.99591302959919E+4-np.power(r,1.2E+1)*1.8908649754107E+4+np.power(r,1.4E+1)*6.527986224621534E+3
        if n == 8:
            R = (r*r*r*r*r*r*r*r)*-1.064486949119717E+3+np.power(r,1.0E+1)*4.201922167569399E+3-np.power(r,1.2E+1)*5.402471358314157E+3+np.power(r,1.4E+1)*2.270603904217482E+3
        if n == 10:
            R = np.power(r,1.0E+1)*4.001830635787919E+2-np.power(r,1.2E+1)*9.395602362267673E+2+np.power(r,1.4E+1)*5.44944937011227E+2
        if n == 12:
            R = np.power(r,1.2E+1)*-7.516481889830902E+1+np.power(r,1.4E+1)*8.073258326109499E+1
        if n == 14:
            R = np.power(r,1.4E+1)*5.567764362829621
    elif l1 == 15:
        if n == 1:
            R = r*-1.022829477079213E+2+(r*r*r)*2.720726409032941E+3-(r*r*r*r*r)*2.448653768128157E+4+(r*r*r*r*r*r*r)*1.042945123462677E+5-(r*r*r*r*r*r*r*r*r)*2.370329826049805E+5+np.power(r,1.1E+1)*2.953795629386902E+5-np.power(r,1.3E+1)*1.903557183384895E+5+np.power(r,1.5E+1)*4.958846444106102E+4
        if n == 3:
            R = (r*r*r)*7.773504025805742E+2-(r*r*r*r*r)*1.088290563613176E+4+(r*r*r*r*r*r*r)*5.688791582524776E+4-(r*r*r*r*r*r*r*r*r)*1.458664508337975E+5+np.power(r,1.1E+1)*1.969197086257935E+5-np.power(r,1.3E+1)*1.343687423563004E+5+np.power(r,1.5E+1)*3.653886853551865E+4
        if n == 5:
            R = (r*r*r*r*r)*-1.978710115659982E+3+(r*r*r*r*r*r*r)*1.750397410005331E+4-(r*r*r*r*r*r*r*r*r)*5.834658033359051E+4+np.power(r,1.1E+1)*9.266809817695618E+4-np.power(r,1.3E+1)*7.072039071393013E+4+np.power(r,1.5E+1)*2.08793534488678E+4
        if n == 7:
            R = (r*r*r*r*r*r*r)*2.333863213345408E+3-(r*r*r*r*r*r*r*r*r)*1.372860713732243E+4+np.power(r,1.1E+1)*2.926360995060205E+4-np.power(r,1.3E+1)*2.694110122436285E+4+np.power(r,1.5E+1)*9.077979760378599E+3
        if n == 9:
            R = (r*r*r*r*r*r*r*r*r)*-1.445116540770978E+3+np.power(r,1.1E+1)*5.57402094297111E+3-np.power(r,1.3E+1)*7.028113362878561E+3+np.power(r,1.5E+1)*2.90495352332294E+3
        if n == 11:
            R = np.power(r,1.1E+1)*4.846974733015522E+2-np.power(r,1.3E+1)*1.124498138058931E+3+np.power(r,1.5E+1)*6.455452274046838E+2
        if n == 13:
            R = np.power(r,1.3E+1)*-8.329615837475285E+1+np.power(r,1.5E+1)*8.904072102135979E+1
        if n == 15:
            R = np.power(r,1.5E+1)*5.744562646534177

    # Spherical Harmonics

    if l2 == 0:
        Y = np.asarray([(1.0 / 2.0) * np.sqrt(1.0 / np.pi)] * len(xr))
    elif l2 == 1:
        if m == -1:
            Y = np.sqrt(3.0 / (4.0 * np.pi)) * yr
        elif m == 0:
            Y = np.sqrt(3.0 / (4.0 * np.pi)) * zr
        elif m == 1:
            Y = np.sqrt(3.0 / (4.0 * np.pi)) * xr
    elif l2 == 2:
        if m == -2:
            Y = np.sqrt(15.0 / (4.0 * np.pi)) * xr * yr
        elif m == -1:
            Y = np.sqrt(15.0 / (4.0 * np.pi)) * zr * yr
        elif m == 0:
            Y = np.sqrt(5.0 / (16.0 * np.pi)) * (-xr2 - yr2 + 2.0 * zr2)
        elif m == 1:
            Y = np.sqrt(15.0 / (4.0 * np.pi)) * xr * zr
        elif m == 2:
            Y = np.sqrt(15.0 / (16.0 * np.pi)) * (xr2 - yr2)
    elif l2 == 3:
        if m == -3:
            Y = np.sqrt(35.0 / (16.0 * 2.0 * np.pi)) * yr * (3.0 * xr2 - yr2)
        elif m == -2:
            Y = np.sqrt(105.0 / (4.0 * np.pi)) * zr * yr * xr
        elif m == -1:
            Y = np.sqrt(21.0 / (16.0 * 2.0 * np.pi)) * yr * (4.0 * zr2 - xr2 - yr2)
        elif m == 0:
            Y = np.sqrt(7.0 / (16.0 * np.pi)) * zr * (2.0 * zr2 - 3.0 * xr2 - 3.0 * yr2)
        elif m == 1:
            Y = np.sqrt(21.0 / (16.0 * 2.0 * np.pi)) * xr * (4.0 * zr2 - xr2 - yr2)
        elif m == 2:
            Y = np.sqrt(105.0 / (16.0 * np.pi)) * zr * (xr2 - yr2)
        elif m == 3:
            Y = np.sqrt(35.0 / (16.0 * 2.0 * np.pi)) * xr * (xr2 - 3.0 * yr2)
    elif l2 == 4:
        if m == -4:
            Y = np.sqrt((35.0 * 9.0) / (16.0 * np.pi)) * yr * xr * (xr2 - yr2)
        elif m == -3:
            Y = np.sqrt((9.0 * 35.0) / (16.0 * 2.0 * np.pi)) * yr * zr * (3.0 * xr2 - yr2)
        elif m == -2:
            Y = np.sqrt((9.0 * 5.0) / (16.0 * np.pi)) * yr * xr * (7.0 * zr2 - (xr2 + yr2 + zr2))
        elif m == -1:
            Y = np.sqrt((9.0 * 5.0) / (16.0 * 2.0 * np.pi)) * yr * zr * (7.0 * zr2 - 3.0 * (xr2 + yr2 + zr2))
        elif m == 0:
            Y = np.sqrt(9.0 / (16.0 * 16.0 * np.pi)) * (35.0 * zr2 * zr2 - 30.0 * zr2 + 3.0)
        elif m == 1:
            Y = np.sqrt((9.0 * 5.0) / (16.0 * 2.0 * np.pi)) * xr * zr * (7.0 * zr2 - 3.0 * (xr2 + yr2 + zr2))
        elif m == 2:
            Y = np.sqrt((9.0 * 5.0) / (8.0 * 8.0 * np.pi)) * (xr2 - yr2) * (7.0 * zr2 - (xr2 + yr2 + zr2))
        elif m == 3:
            Y = np.sqrt((9.0 * 35.0) / (16.0 * 2.0 * np.pi)) * xr * zr * (xr2 - 3.0 * yr2)
        elif m == 4:
            Y = np.sqrt((9.0 * 35.0) / (16.0 * 16.0 * np.pi)) * (xr2 * (xr2 - 3.0 * yr2) - yr2 * (3.0 * xr2 - yr2))
    elif l2 == 5:
        if m == -5:
            Y = (3.0 / 16.0) * np.sqrt(77.0 / (2.0 * np.pi)) * sinth2 * sinth2 * sinth * np.sin(5.0 * phi)
        elif m == -4:
            Y = (3.0 / 8.0) * np.sqrt(385.0 / (2.0 * np.pi)) * sinth2 * sinth2 * np.sin(4.0 * phi)
        elif m == -3:
            Y = (1.0 / 16.0) * np.sqrt(385.0 / (2.0 * np.pi)) * sinth2 * sinth * (9.0 * costh2 - 1.0) * np.sin(3.0 * phi)
        elif m == -2:
            Y = (1.0 / 4.0) * np.sqrt(1155.0 / (4.0 * np.pi)) * sinth2 * (3.0 * costh2 * costh - costh) * np.sin(2.0 * phi)
        elif m == -1:
            Y = (1.0 / 8.0) * np.sqrt(165.0 / (4.0 * np.pi)) * sinth * (21.0 * costh2 * costh2 - 14.0 * costh2 + 1) * np.sin(phi)
        elif m == 0:
            Y = (1.0 / 16.0) * np.sqrt(11.0 / np.pi) * (63.0 * costh2 * costh2 * costh - 70.0 * costh2 * costh + 15.0 * costh)
        elif m == 1:
            Y = (1.0 / 8.0) * np.sqrt(165.0 / (4.0 * np.pi)) * sinth * (21.0 * costh2 * costh2 - 14.0 * costh2 + 1) * np.cos(phi)
        elif m == 2:
            Y = (1.0 / 4.0) * np.sqrt(1155.0 / (4.0 * np.pi)) * sinth2 * (3.0 * costh2 * costh - costh) * np.cos(2.0 * phi)
        elif m == 3:
            Y = (1.0 / 16.0) * np.sqrt(385.0 / (2.0 * np.pi)) * sinth2 * sinth * (9.0 * costh2 - 1.0) * np.cos(3.0 * phi)
        elif m == 4:
            Y = (3.0 / 8.0) * np.sqrt(385.0 / (2.0 * np.pi)) * sinth2 * sinth2 * np.cos(4.0 * phi)
        elif m == 5:
            Y = (3.0 / 16.0) * np.sqrt(77.0 / (2.0 * np.pi)) * sinth2 * sinth2 * sinth * np.cos(5.0 * phi)
    elif l2 == 6:
        if m == -6:
            Y = -0.6832*sinth*np.power(costh2 - 1.0, 3)
        elif m == -5:
            Y = 2.367*costh*sinth*np.power(1.0 - 1.0*costh2, 2.5)
        elif m == -4:
            Y = 0.001068*sinth*(5198.0*costh2 - 472.5)*np.power(costh2 - 1.0, 2)
        elif m == -3:
            Y = -0.005849*sinth*np.power(1.0 - 1.0*costh2, 1.5)*(- 1732.0*costh2*costh + 472.5*costh)
        elif m == -2:
            Y = -0.03509*sinth*(costh2 - 1.0)*(433.1*costh2*costh2 - 236.2*costh2 + 13.12)
        elif m == -1:
            Y = 0.222*sinth*np.power(1.0 - 1.0*costh2, 0.5)*(86.62*costh2*costh2*costh - 78.75*costh2*costh + 13.12*costh)
        elif m == 0:
            Y = 14.68*costh2*costh2*costh2 - 20.02*costh2*costh2 + 6.675*costh2 - 0.3178
        elif m == 1:
            Y = 0.222*cosph*np.power(1.0 - 1.0*costh2, 0.5)*(86.62*costh2*costh2*costh - 78.75*costh2*costh + 13.12*costh2)
        elif m == 2:
            Y = -0.03509*cosph*(costh2 - 1.0)*(433.1*costh2*costh2 - 236.2*costh2 + 13.12)
        elif m == 3:
            Y = -0.005849*cosph*np.power(1.0 - 1.0*costh2, 1.5)*(-1732.0*costh2*costh + 472.5*costh)
        elif m == 4:
            Y = 0.001068*cosph*(5198.0*costh2 - 472.5)*np.power(costh2 - 1.0, 2)
        elif m == 5:
            Y = 2.367*costh*cosph*np.power(1.0 - 1.0*costh2, 2.5)
        elif m == 6:
            Y = -0.6832*cosph*np.power(costh2 - 1.0, 3)
    elif l2 == 7:
        if m == -7:
            Y = 0.7072*sinth*np.power(1.0 - 1.0*costh2, 3.5)
        elif m == -6:
            Y = -2.646*costh*sinth*np.power(costh2 - 1.0, 3)
        elif m == -5:
            Y = 9.984e-5*sinth*np.power(1.0 - 1.0*costh2, 2.5)*(67570.0*costh2 - 5198.0)
        elif m == -4:
            Y = -0.000599*sinth*np.power(costh2 - 1.0, 2)*(-22520.0*costh2*costh + 5198.0*costh)
        elif m == -3:
            Y = 0.003974*sinth*np.power(1.0 - 1.0*costh2, 1.5)*(5631.0*costh2*costh2 - 2599.0*costh2 + 118.1)
        elif m == -2:
            Y = -0.0281*sinth*(costh2 - 1.0)*(1126.0*costh2*costh2*costh - 866.2*costh2*costh + 118.1*costh)
        elif m == -1:
            Y = 0.2065*sinth*np.power(1.0 - 1.0*costh2, 0.5)*(187.7*costh2*costh2*costh2 - 216.6*costh2*costh2 + 59.06*costh2 - 2.188)
        elif m == 0:
            Y = 29.29*costh2*costh2*costh2*costh - 47.32*costh2*costh2*costh + 21.51*costh2*costh - 2.39*costh
        elif m == 1:
            Y = 0.2065*cosph*np.power(1.0 - 1.0*costh2, 0.5)*(187.7*costh2*costh2*costh2 - 216.6*costh2*costh2 + 59.06*costh2 - 2.188)
        elif m == 2:
            Y = -0.0281*cosph*(costh2 - 1.0)*(1126.0*costh2*costh2*costh - 866.2*costh2*costh + 118.1*costh)
        elif m == 3:
            Y = 0.003974*cosph*np.power(1.0 - 1.0*costh2, 1.5)*(5631.0*costh2*costh2 - 2599.0*costh2 + 118.1)
        elif m == 4:
            Y = -0.000599*cosph*np.power(costh2 - 1.0, 2)*(- 22520.0*costh2*costh + 5198.0*costh)
        elif m == 5:
            Y = 9.984e-5*cosph*np.power(1.0 - 1.0*costh2, 2.5)*(67570.0*costh2 - 5198.0)
        elif m == 6:
            Y = -2.646*cosph*costh*np.power(costh2 - 1.0, 3)
        elif m == 7:
            Y = 0.7072*cosph*np.power(1.0 - 1.0*costh2, 3.5)
    elif l2 == 8:
        if m == -8:
            Y = sinth*np.power(costh2-1.0,4.0)*7.289266601746931E-1
        elif m == -7:
            Y = costh*sinth*np.power((costh2)*-1.0+1.0,7.0/2.0)*2.915706640698772
        elif m == -6:
            Y = sinth*((costh2)*1.0135125E+6-6.75675E+4)*np.power(costh2-1.0,3.0)*-7.878532816224526E-6
        elif m == -5:
            Y = sinth*np.power((costh2)*-1.0+1.0,5.0/2.0)*(costh*6.75675E+4-(costh2*costh)*3.378375E+5)*-5.105872826582925E-56
        elif m == -4:
            Y = sinth*np.power(costh2-1.0,2.0)*((costh2)*-3.378375E+4+(costh2*costh2)*8.4459375E+4+1.299375E+3)*3.681897256448963E-4
        elif m == -3:
            Y = sinth*np.power((costh2)*-1.0+1.0,3.0/2.0)*(costh*1.299375E+3-(costh*costh2)*1.126125E+4+(costh*costh2*costh2)*1.6891875E+4)*2.851985351334463E-3
        elif m == -2:
            Y = sinth*(costh2-1.0)*((costh2)*6.496875E+2-(costh2*costh2)*2.8153125E+3+(costh2*costh2*costh2)*2.8153125E+3-1.96875E+1)*-2.316963852365461E-2
        elif m == -1:
            Y = sinth*sqrt((costh2)*-1.0+1.0)*(costh*1.96875E+1-(costh*costh2)*2.165625E+2+(costh*costh2*costh2)*5.630625E+2-(costh*costh2*costh2*costh2)*4.021875E+2)*-1.938511038201796E-1;
        elif m == 0:
            Y = (costh2)*-1.144933081936324E+1+(costh2*costh2)*6.297131950652692E+1-(costh2*costh2*costh2)*1.091502871445846E+2+(costh2*costh2*costh2*costh2)*5.847336811327841E+1+3.180369672045344E-1
        elif m == 1:
            Y = cosph*sqrt((costh2)*-1.0+1.0)*(costh*1.96875E+1-(costh*costh2)*2.165625E+2+(costh*costh2*costh2)*5.630625E+2-(costh*costh2*costh2*costh2)*4.021875E+2)*-1.938511038201796E-1;
        elif m == 2:
            Y = cosph*(costh2-1.0)*((costh2)*6.496875E+2-(costh2*costh2)*2.8153125E+3+(costh2*costh2*costh2)*2.8153125E+3-1.96875E+1)*-2.316963852365461E-2
        elif m == 3:
            Y = cosph*np.power((costh2)*-1.0+1.0,3.0/2.0)*(costh*1.299375E+3-(costh*costh2)*1.126125E+4+(costh*costh2*costh2)*1.6891875E+4)*2.851985351334463E-3
        elif m == 4:
            Y = cosph*np.power(costh2-1.0,2.0)*((costh2)*-3.378375E+4+(costh2*costh2)*8.4459375E+4+1.299375E+3)*3.681897256448963E-4
        elif m == 5:
            Y = cosph*np.power((costh2)*-1.0+1.0,5.0/2.0)*(costh*6.75675E+4-(costh*costh2)*3.378375E+5)*-5.105872826582925E-5
        elif m == 6:
            Y = cosph*((costh2)*1.0135125E+6-6.75675E+4)*np.power(costh2-1.0,3.0)*-7.878532816224526E-6
        elif m == 7:
            Y = cosph*costh*np.power((costh2)*-1.0+1.0,7.0/2.0)*2.915706640698772
        elif m == 8:
            Y = cosph*np.power(costh2-1.0,4.0)*7.289266601746931E-1
    elif l2 == 9:
        if m == -9:
            Y = sinth*np.power((costh2)*-1.0+1.0,9.0/2.0)*7.489009518540115E-1
        elif m == -8:
            Y = costh*sinth*np.power(costh2-1.0,4.0)*3.17731764895143
        elif m == -7:
            Y = sinth*np.power((costh2)*-1.0+1.0,7.0/2.0)*((costh2)*1.72297125E+7-1.0135125E+6)*5.376406125665728E-7
        elif m == -6:
            Y = sinth*(costh*1.0135125E+6-(costh*costh2)*5.7432375E+6)*np.power(costh2-1.0,3.0)*3.724883428715686E-6
        elif m == -5:
            Y = sinth*np.power((costh2)*-1.0+1.0,5.0/2.0)*((costh2)*-5.0675625E+5+(costh2*costh2)*1.435809375E+6+1.6891875E+4)*2.885282297193648E-5
        elif m == -4:
            Y = sinth*np.power(costh2-1.0,2.0)*(costh*1.6891875E+4-(costh*costh2)*1.6891875E+5+(costh*costh2*costh2)*2.87161875E+5)*2.414000363328839E-4
        elif m == -3:
            Y = sinth*np.power((costh2)*-1.0+1.0,3.0/2.0)*((costh2)*8.4459375E+3-(costh2*costh2)*4.22296875E+4+(costh2*costh2*costh2)*4.78603125E+4-2.165625E+2)*2.131987394015766E-3
        elif m == -2:
            Y = sinth*(costh2-1.0)*(costh*2.165625E+2-(costh*costh2)*2.8153125E+3+(costh*costh2*costh2)*8.4459375E+3-(costh*costh2*costh2*costh2)*6.8371875E+3)*1.953998722751749E-2
        elif m == -1:
            Y = sinth*sqrt((costh2)*-1.0+1.0)*((costh2)*-1.0828125E+2+(costh2*costh2)*7.03828125E+2-(costh2*costh2*costh2)*1.40765625E+3+(costh2*costh2*costh2*costh2)*8.546484375E+2+2.4609375)*1.833013280775049E-1
        elif m == 0:
            Y = costh*3.026024588281871-(costh*costh2)*4.438169396144804E+1+(costh*costh2*costh2)*1.730886064497754E+2-(costh*costh2*costh2*costh2)*2.472694377852604E+2+(costh*costh2*costh2*costh2*costh2)*1.167661233986728E+2
        elif m == 1:
            Y = cosph*sqrt((costh2)*-1.0+1.0)*((costh2)*-1.0828125E+2+(costh2*costh2)*7.03828125E+2-(costh2*costh2*costh2)*1.40765625E+3+(costh2*costh2*costh2*costh2)*8.546484375E+2+2.4609375)*1.833013280775049E-1
        elif m == 2:
            Y = cosph*(costh2-1.0)*(costh*2.165625E+2-(costh*costh2)*2.8153125E+3+(costh*costh2*costh2)*8.4459375E+3-(costh*costh2*costh2*costh2)*6.8371875E+3)*1.953998722751749E-2
        elif m == 3:
            Y = cosph*np.power((costh2)*-1.0+1.0,3.0/2.0)*((costh2)*8.4459375E+3-(costh2*costh2)*4.22296875E+4+(costh2*costh2*costh2)*4.78603125E+4-2.165625E+2)*2.131987394015766E-3
        elif m == 4:
            Y = cosph*np.power(costh2-1.0,2.0)*(costh*1.6891875E+4-(costh*costh2)*1.6891875E+5+(costh*costh2*costh2)*2.87161875E+5)*2.414000363328839E-4
        elif m == 5:
            Y = cosph*np.power((costh2)*-1.0+1.0,5.0/2.0)*((costh2)*-5.0675625E+5+(costh2*costh2)*1.435809375E+6+1.6891875E+4)*2.885282297193648E-5
        elif m == 6:
            Y = cosph*(costh*1.0135125E+6-(costh*costh2)*5.7432375E+6)*np.power(costh2-1.0,3.0)*3.724883428715686E-6
        elif m == 7:
            Y = cosph*np.power((costh2)*-1.0+1.0,7.0/2.0)*((costh2)*1.72297125E+7-1.0135125E+6)*5.376406125665728E-7
        elif m == 8:
            Y = cosph*costh*np.power(costh2-1.0,4.0)*3.17731764895143
        elif m == 9:
            Y = cosph*np.power((costh2)*-1.0+1.0,9.0/2.0)*7.489009518540115E-1
    elif l2 == 10:
        if m == -10:
            Y = sinth*np.power(costh*costh-1.0,5.0)*-7.673951182223391E-1
        elif m == -9:
            Y = costh*sinth*np.power((costh*costh)*-1.0+1.0,9.0/2.0)*3.431895299894677
        elif m == -8:
            Y = sinth*((costh*costh)*3.273645375E+8-1.72297125E+7)*np.power(costh*costh-1.0,4.0)*3.231202683857352E-8
        elif m == -7:
            Y = sinth*np.power((costh*costh)*-1.0+1.0,7.0/2.0)*(costh*1.72297125E+7-(costh*costh*costh)*1.091215125E+8)*-2.374439349284684E-7
        elif m == -6:
            Y = sinth*np.power(costh*costh-1.0,3.0)*((costh*costh)*-8.61485625E+6+(costh*costh*costh*costh)*2.7280378125E+7+2.53378125E+5)*-1.958012847746993E-6
        elif m == -5:
            Y = sinth*np.power((costh*costh)*-1.0+1.0,5.0/2.0)*(costh*2.53378125E+5-(costh*costh*costh)*2.87161875E+6+(costh*costh*costh*costh*costh)*5.456075625E+6)*1.751299931351813E-5
        elif m == -4:
            Y = sinth*np.power(costh*costh-1.0,2.0)*((costh*costh)*1.266890625E+5-(costh*costh*costh*costh)*7.179046875E+5+(costh*costh*costh*costh*costh*costh)*9.093459375E+5-2.8153125E+3)*1.661428994750302E-4
        elif m == -3:
            Y = sinth*np.power((costh*costh)*-1.0+1.0,3.0/2.0)*(costh*2.8153125E+3-(costh*costh*costh)*4.22296875E+4+(costh*costh*costh*costh*costh)*1.435809375E+5-(costh*costh*costh*costh*costh*costh*costh)*1.299065625E+5)*-1.644730792108362E-3
        elif m == -2:
            Y = sinth*(costh*costh-1.0)*((costh*costh)*-1.40765625E+3+(costh*costh*costh*costh)*1.0557421875E+4-(costh*costh*costh*costh*costh*costh)*2.393015625E+4+(costh*costh*costh*costh*costh*costh*costh*costh)*1.62383203125E+4+2.70703125E+1)*-1.67730288071084E-2
        elif m == -1:
            Y = sinth*sqrt((costh*costh)*-1.0+1.0)*(costh*2.70703125E+1-(costh*costh*costh)*4.6921875E+2+(costh*costh*costh*costh*costh)*2.111484375E+3-(costh*costh*costh*costh*costh*costh*costh)*3.41859375E+3+(costh*costh*costh*costh*costh*costh*costh*costh*costh)*1.8042578125E+3)*1.743104285446861E-1
        elif m == 0:
            Y = (costh*costh)*1.749717715557199E+1-(costh*costh*costh*costh)*1.516422020150349E+2+(costh*costh*costh*costh*costh*costh)*4.549266060441732E+2-(costh*costh*costh*costh*costh*costh*costh*costh)*5.524108787681907E+2+np.power(costh,1.0E+1)*2.332401488134637E+2-3.181304937370442E-1
        elif m == 1:
            Y = cosph*sqrt((costh*costh)*-1.0+1.0)*(costh*2.70703125E+1-(costh*costh*costh)*4.6921875E+2+(costh*costh*costh*costh*costh)*2.111484375E+3-(costh*costh*costh*costh*costh*costh*costh)*3.41859375E+3+(costh*costh*costh*costh*costh*costh*costh*costh*costh)*1.8042578125E+3)*1.743104285446861E-1
        elif m == 2:
            Y = cosph*(costh*costh-1.0)*((costh*costh)*-1.40765625E+3+(costh*costh*costh*costh)*1.0557421875E+4-(costh*costh*costh*costh*costh*costh)*2.393015625E+4+(costh*costh*costh*costh*costh*costh*costh*costh)*1.62383203125E+4+2.70703125E+1)*-1.67730288071084E-2
        elif m == 3:
            Y = cosph*np.power((costh*costh)*-1.0+1.0,3.0/2.0)*(costh*2.8153125E+3-(costh*costh*costh)*4.22296875E+4+(costh*costh*costh*costh*costh)*1.435809375E+5-(costh*costh*costh*costh*costh*costh*costh)*1.299065625E+5)*-1.644730792108362E-3
        elif m == 4:
            Y = cosph*np.power(costh*costh-1.0,2.0)*((costh*costh)*1.266890625E+5-(costh*costh*costh*costh)*7.179046875E+5+(costh*costh*costh*costh*costh*costh)*9.093459375E+5-2.8153125E+3)*1.661428994750302E-4
        elif m == 5:
            Y = cosph*np.power((costh*costh)*-1.0+1.0,5.0/2.0)*(costh*2.53378125E+5-(costh*costh*costh)*2.87161875E+6+(costh*costh*costh*costh*costh)*5.456075625E+6)*1.751299931351813E-5
        elif m == 6:
            Y = cosph*np.power(costh*costh-1.0,3.0)*((costh*costh)*-8.61485625E+6+(costh*costh*costh*costh)*2.7280378125E+7+2.53378125E+5)*-1.958012847746993E-6
        elif m == 7:
            Y = cosph*np.power((costh*costh)*-1.0+1.0,7.0/2.0)*(costh*1.72297125E+7-(costh*costh*costh)*1.091215125E+8)*-2.374439349284684E-7
        elif m == 8:
            Y = cosph*((costh*costh)*3.273645375E+8-1.72297125E+7)*np.power(costh*costh-1.0,4.0)*3.231202683857352E-8
        elif m == 9:
            Y = cosph*costh*np.power((costh*costh)*-1.0+1.0,9.0/2.0)*3.431895299894677
        elif m == 10:
            Y = cosph*np.power(costh*costh-1.0,5.0)*-7.673951182223391E-1
    elif l2 == 11:
        if m == -11:
            Y = sinth*np.power((costh*costh)*-1.0+1.0,1.1E+1/2.0)*7.846421057874977E-1
        elif m == -10:
            Y = costh*sinth*np.power(costh*costh-1.0,5.0)*-3.68029769880377
        elif m == -9:
            Y = sinth*np.power((costh*costh)*-1.0+1.0,9.0/2.0)*((costh*costh)*6.8746552875E+9-3.273645375E+8)*1.734709165873547E-9
        elif m == -8:
            Y = sinth*np.power(costh*costh-1.0,4.0)*(costh*3.273645375E+8-(costh*costh*costh)*2.2915517625E+9)*-1.343699941990114E-8
        elif m == -7:
            Y = sinth*np.power((costh*costh)*-1.0+1.0,7.0/2.0)*((costh*costh)*-1.6368226875E+8+(costh*costh*costh*costh)*5.72887940625E+8+4.307428125E+6)*1.171410451514688E-7
        elif m == -6:
            Y = sinth*np.power(costh*costh-1.0,3.0)*(costh*4.307428125E+6-(costh*costh*costh)*5.456075625E+7+(costh*costh*costh*costh*costh)*1.14577588125E+8)*-1.111297530512201E-6
        elif m == -5:
            Y = sinth*np.power((costh*costh)*-1.0+1.0,5.0/2.0)*((costh*costh)*2.1537140625E+6-(costh*costh*costh*costh)*1.36401890625E+7+(costh*costh*costh*costh*costh*costh)*1.90962646875E+7-4.22296875E+4)*1.122355489741045E-5
        elif m == -4:
            Y = sinth*np.power(costh*costh-1.0,2.0)*(costh*4.22296875E+4-(costh*costh*costh)*7.179046875E+5+(costh*costh*costh*costh*costh)*2.7280378125E+6-(costh*costh*costh*costh*costh*costh*costh)*2.7280378125E+6)*-1.187789403385153E-4
        elif m == -3:
            Y = sinth*np.power((costh*costh)*-1.0+1.0,3.0/2.0)*((costh*costh)*-2.111484375E+4+(costh*costh*costh*costh)*1.79476171875E+5-(costh*costh*costh*costh*costh*costh)*4.5467296875E+5+(costh*costh*costh*costh*costh*costh*costh*costh)*3.410047265625E+5+3.519140625E+2)*1.301158099600741E-3
        elif m == -2:
            Y = sinth*(costh*costh-1.0)*(costh*3.519140625E+2-(costh*costh*costh)*7.03828125E+3+(costh*costh*costh*costh*costh)*3.5895234375E+4-(costh*costh*costh*costh*costh*costh*costh)*6.495328125E+4+(costh*costh*costh*costh*costh*costh*costh*costh*costh)*3.78894140625E+4)*-1.46054634441839E-2
        elif m == -1:
            Y = sinth*sqrt((costh*costh)*-1.0+1.0)*((costh*costh)*1.7595703125E+2-(costh*costh*costh*costh)*1.7595703125E+3+(costh*costh*costh*costh*costh*costh)*5.9825390625E+3-(costh*costh*costh*costh*costh*costh*costh*costh)*8.11916015625E+3+np.power(costh,1.0E+1)*3.78894140625E+3-2.70703125)*1.665279049125274E-1
        elif m == 0:
            Y = costh*-3.662285987506039+(costh*costh*costh)*7.934952972922474E+1-(costh*costh*costh*costh*costh)*4.760971783753484E+2+(costh*costh*costh*costh*costh*costh*costh)*1.156236004628241E+3-(costh*costh*costh*costh*costh*costh*costh*costh*costh)*1.220471338216215E+3+np.power(costh,1.1E+1)*4.65998147319071E+2
        elif m == 1:
            Y = cosph*sqrt((costh*costh)*-1.0+1.0)*((costh*costh)*1.7595703125E+2-(costh*costh*costh*costh)*1.7595703125E+3+(costh*costh*costh*costh*costh*costh)*5.9825390625E+3-(costh*costh*costh*costh*costh*costh*costh*costh)*8.11916015625E+3+np.power(costh,1.0E+1)*3.78894140625E+3-2.70703125)*1.665279049125274E-1
        elif m == 2:
            Y = cosph*(costh*costh-1.0)*(costh*3.519140625E+2-(costh*costh*costh)*7.03828125E+3+(costh*costh*costh*costh*costh)*3.5895234375E+4-(costh*costh*costh*costh*costh*costh*costh)*6.495328125E+4+(costh*costh*costh*costh*costh*costh*costh*costh*costh)*3.78894140625E+4)*-1.46054634441839E-2
        elif m == 3:
            Y = cosph*np.power((costh*costh)*-1.0+1.0,3.0/2.0)*((costh*costh)*-2.111484375E+4+(costh*costh*costh*costh)*1.79476171875E+5-(costh*costh*costh*costh*costh*costh)*4.5467296875E+5+(costh*costh*costh*costh*costh*costh*costh*costh)*3.410047265625E+5+3.519140625E+2)*1.301158099600741E-3
        elif m == 4:
            Y = cosph*np.power(costh*costh-1.0,2.0)*(costh*4.22296875E+4-(costh*costh*costh)*7.179046875E+5+(costh*costh*costh*costh*costh)*2.7280378125E+6-(costh*costh*costh*costh*costh*costh*costh)*2.7280378125E+6)*-1.187789403385153E-4
        elif m == 5:
            Y = cosph*np.power((costh*costh)*-1.0+1.0,5.0/2.0)*((costh*costh)*2.1537140625E+6-(costh*costh*costh*costh)*1.36401890625E+7+(costh*costh*costh*costh*costh*costh)*1.90962646875E+7-4.22296875E+4)*1.122355489741045E-5
        elif m == 6:
            Y = cosph*np.power(costh*costh-1.0,3.0)*(costh*4.307428125E+6-(costh*costh*costh)*5.456075625E+7+(costh*costh*costh*costh*costh)*1.14577588125E+8)*-1.111297530512201E-6
        elif m == 7:
            Y = cosph*np.power((costh*costh)*-1.0+1.0,7.0/2.0)*((costh*costh)*-1.6368226875E+8+(costh*costh*costh*costh)*5.72887940625E+8+4.307428125E+6)*1.171410451514688E-7
        elif m == 8:
            Y = cosph*np.power(costh*costh-1.0,4.0)*(costh*3.273645375E+8-(costh*costh*costh)*2.2915517625E+9)*-1.343699941990114E-8
        elif m == 9:
            Y = cosph*np.power((costh*costh)*-1.0+1.0,9.0/2.0)*((costh*costh)*6.8746552875E+9-3.273645375E+8)*1.734709165873547E-9
        elif m == 10:
            Y = cosph*costh*np.power(costh*costh-1.0,5.0)*-3.68029769880377
        elif m == 11:
            Y = cosph*np.power((costh*costh)*-1.0+1.0,1.1E+1/2.0)*7.846421057874977E-1
    elif l2 == 12:
        if m == -12:
            Y = sinth*np.power(costh*costh-1.0,6.0)*8.00821995784645E-1
        elif m == -11:
            Y = costh*sinth*np.power((costh*costh)*-1.0+1.0,1.1E+1/2.0)*3.923210528933851
        elif m == -10:
            Y = sinth*((costh*costh)*1.581170716125E+11-6.8746552875E+9)*np.power(costh*costh-1.0,5.0)*-8.414179483959553E-11
        elif m == -9:
            Y = sinth*np.power((costh*costh)*-1.0+1.0,9.0/2.0)*(costh*6.8746552875E+9-(costh*costh*costh)*5.27056905375E+10)*-6.83571172712202E-10
        elif m == -8:
            Y = sinth*np.power(costh*costh-1.0,4.0)*((costh*costh)*-3.43732764375E+9+(costh*costh*costh*costh)*1.3176422634375E+10+8.1841134375E+7)*6.265033283689913E-9
        elif m == -7:
            Y = sinth*np.power((costh*costh)*-1.0+1.0,7.0/2.0)*(costh*8.1841134375E+7-(costh*costh*costh)*1.14577588125E+9+(costh*costh*costh*costh*costh)*2.635284526875E+9)*6.26503328367365E-8
        elif m == -6:
            Y = sinth*np.power(costh*costh-1.0,3.0)*((costh*costh)*4.09205671875E+7-(costh*costh*costh*costh)*2.864439703125E+8+(costh*costh*costh*costh*costh*costh)*4.392140878125E+8-7.179046875E+5)*-6.689225062143228E-7
        elif m == -5:
            Y = sinth*np.power((costh*costh)*-1.0+1.0,5.0/2.0)*(costh*7.179046875E+5-(costh*costh*costh)*1.36401890625E+7+(costh*costh*costh*costh*costh)*5.72887940625E+7-(costh*costh*costh*costh*costh*costh*costh)*6.27448696875E+7)*-7.50863650966771E-6
        elif m == -4:
            Y = sinth*np.power(costh*costh-1.0,2.0)*((costh*costh)*-3.5895234375E+5+(costh*costh*costh*costh)*3.410047265625E+6-(costh*costh*costh*costh*costh*costh)*9.54813234375E+6+(costh*costh*costh*costh*costh*costh*costh*costh)*7.8431087109375E+6+5.2787109375E+3)*8.756499656747962E-5
        elif m == -3:
            Y = sinth*np.power((costh*costh)*-1.0+1.0,3.0/2.0)*(costh*5.2787109375E+3-(costh*costh*costh)*1.1965078125E+5+(costh*costh*costh*costh*costh)*6.82009453125E+5-(costh*costh*costh*costh*costh*costh*costh)*1.36401890625E+6+(costh*costh*costh*costh*costh*costh*costh*costh*costh)*8.714565234375E+5)*1.050779958809755E-3
        elif m == -2:
            Y = sinth*(costh*costh-1.0)*((costh*costh)*2.63935546875E+3-(costh*costh*costh*costh)*2.99126953125E+4+(costh*costh*costh*costh*costh*costh)*1.136682421875E+5-(costh*costh*costh*costh*costh*costh*costh*costh)*1.7050236328125E+5+np.power(costh,1.0E+1)*8.714565234375E+4-3.519140625E+1)*-1.286937365514973E-2
        elif m == -1:
            Y = sinth*sqrt((costh*costh)*-1.0+1.0)*(costh*3.519140625E+1-(costh*costh*costh)*8.7978515625E+2+(costh*costh*costh*costh*costh)*5.9825390625E+3-(costh*costh*costh*costh*costh*costh*costh)*1.62383203125E+4+(costh*costh*costh*costh*costh*costh*costh*costh*costh)*1.894470703125E+4-np.power(costh,1.1E+1)*7.92233203125E+3)*-1.597047270888652E-1
        elif m == 0:
            Y = (costh*costh)*-2.481828104582382E+1+(costh*costh*costh*costh)*3.102285130722448E+2-(costh*costh*costh*costh*costh*costh)*1.40636925926432E+3+(costh*costh*costh*costh*costh*costh*costh*costh)*2.862965992070735E+3-np.power(costh,1.0E+1)*2.672101592600346E+3+np.power(costh,1.2E+1)*9.311869186330587E+2+3.181830903313312E-1
        elif m == 1:
            Y = cosph*sqrt((costh*costh)*-1.0+1.0)*(costh*3.519140625E+1-(costh*costh*costh)*8.7978515625E+2+(costh*costh*costh*costh*costh)*5.9825390625E+3-(costh*costh*costh*costh*costh*costh*costh)*1.62383203125E+4+(costh*costh*costh*costh*costh*costh*costh*costh*costh)*1.894470703125E+4-np.power(costh,1.1E+1)*7.92233203125E+3)*-1.597047270888652E-1
        elif m == 2:
            Y = cosph*(costh*costh-1.0)*((costh*costh)*2.63935546875E+3-(costh*costh*costh*costh)*2.99126953125E+4+(costh*costh*costh*costh*costh*costh)*1.136682421875E+5-(costh*costh*costh*costh*costh*costh*costh*costh)*1.7050236328125E+5+np.power(costh,1.0E+1)*8.714565234375E+4-3.519140625E+1)*-1.286937365514973E-2
        elif m == 3:
            Y = cosph*np.power((costh*costh)*-1.0+1.0,3.0/2.0)*(costh*5.2787109375E+3-(costh*costh*costh)*1.1965078125E+5+(costh*costh*costh*costh*costh)*6.82009453125E+5-(costh*costh*costh*costh*costh*costh*costh)*1.36401890625E+6+(costh*costh*costh*costh*costh*costh*costh*costh*costh)*8.714565234375E+5)*1.050779958809755E-3
        elif m == 4:
            Y = cosph*np.power(costh*costh-1.0,2.0)*((costh*costh)*-3.5895234375E+5+(costh*costh*costh*costh)*3.410047265625E+6-(costh*costh*costh*costh*costh*costh)*9.54813234375E+6+(costh*costh*costh*costh*costh*costh*costh*costh)*7.8431087109375E+6+5.2787109375E+3)*8.756499656747962E-5
        elif m == 5:
            Y = cosph*np.power((costh*costh)*-1.0+1.0,5.0/2.0)*(costh*7.179046875E+5-(costh*costh*costh)*1.36401890625E+7+(costh*costh*costh*costh*costh)*5.72887940625E+7-(costh*costh*costh*costh*costh*costh*costh)*6.27448696875E+7)*-7.50863650966771E-6
        elif m == 6:
            Y = cosph*np.power(costh*costh-1.0,3.0)*((costh*costh)*4.09205671875E+7-(costh*costh*costh*costh)*2.864439703125E+8+(costh*costh*costh*costh*costh*costh)*4.392140878125E+8-7.179046875E+5)*-6.689225062143228E-7
        elif m == 7:
            Y = cosph*np.power((costh*costh)*-1.0+1.0,7.0/2.0)*(costh*8.1841134375E+7-(costh*costh*costh)*1.14577588125E+9+(costh*costh*costh*costh*costh)*2.635284526875E+9)*6.26503328367365E-8
        elif m == 8:
            Y = cosph*np.power(costh*costh-1.0,4.0)*((costh*costh)*-3.43732764375E+9+(costh*costh*costh*costh)*1.3176422634375E+10+8.1841134375E+7)*6.265033283689913E-9
        elif m == 9:
            Y = cosph*np.power((costh*costh)*-1.0+1.0,9.0/2.0)*(costh*6.8746552875E+9-(costh*costh*costh)*5.27056905375E+10)*-6.83571172712202E-10
        elif m == 10:
            Y = cosph*((costh*costh)*1.581170716125E+11-6.8746552875E+9)*np.power(costh*costh-1.0,5.0)*-8.414179483959553E-11
        elif m == 11:
            Y = cosph*costh*np.power((costh*costh)*-1.0+1.0,1.1E+1/2.0)*3.923210528933851
        elif m == 12:
            Y = cosph*np.power(costh*costh-1.0,6.0)*8.00821995784645E-1

    # Make zero those positions where d_pos_r > 1
    Z = R * Y
    d_pos_r = np.linalg.norm(pos_r, axis=1)
    idx = np.where(d_pos_r > 1)
    Z[idx] = 0

    return Z.reshape(-1, 1)

def computeBasis(pos, **kwargs):
    L1 = kwargs.pop('L1', None)
    L2 = kwargs.pop('L2', None)
    r = kwargs.pop('r', None)
    groups = kwargs.pop("groups", None)
    centers = kwargs.pop("centers", None)

    degrees = basisDegreeVectors(L1, L2)
    if centers is None:
        basis = [computeZernikes3D(degrees[idx, 0], degrees[idx, 1], degrees[idx, 2], degrees[idx, 3],
                                   pos, r) for idx in range(degrees.shape[0])]
        basis = np.hstack(basis)
    else:
        basis_centers = [computeZernikes3D(degrees[idx, 0], degrees[idx, 1], degrees[idx, 2], degrees[idx, 3],
                                           centers, r) for idx in range(degrees.shape[0])]
        basis_centers = np.hstack(basis_centers)
        basis = np.zeros((pos.shape[0], basis_centers.shape[1]))
        for group, basis_center in zip(np.unique(groups), basis_centers):
            basis[groups == group] = basis_center

    return basis

def computeZernikeCoefficients(Df, Z):
    return np.transpose(computeInverse(Z.T @ Z) @ Z.T @ Df)


def readZernikeParams(filename):
    with open(filename, 'r') as fid:
        lines = fid.readlines()
    basis_params = np.fromstring(lines[0].strip('\n'), sep=' ')
    return [int(basis_params[0]), int(basis_params[1]), basis_params[2]]

def readZernikeFile(filename):
    with open(filename, 'r') as fid:
        lines = fid.readlines()
    basis_params = np.fromstring(lines[0].strip('\n'), sep=' ')

    z_clnm = []
    for line in lines[1:]:
        z_clnm.append(np.fromstring(line.strip('\n'), sep=' '))
    z_clnm = np.asarray(z_clnm)

    return basis_params, z_clnm

def writeZernikeFile(file, z_clnm, L1, L2, Rmax):
    with open(file, 'w') as f:
        f.write(' '.join(map(str, [L1, L2, Rmax])) + "\n")
        f.write(' '.join(map(str, z_clnm.reshape(-1))) + "\n")

def resizeZernikeCoefficients(A):
    if A.ndim == 1:
        size = int(len(A) / 3)
        return np.vstack([A[:size], A[size:2 * size], A[2 * size:]])
    else:
        return np.concatenate([A[0], A[1], A[2]])

def maskDeformationField(A, Z, coords, mask, mode="soft", **kwargs):
    deformation_field = Z @ A.T
    logic = coordsInMask(coords, mask)

    # In case we masking requires a finer deformation field
    if "Z_new" in kwargs:
        Z = kwargs.pop("Z_new")

    if mode == "soft":
        deformation_field[np.logical_not(logic)] *= 0
    elif mode == "hard":
        deformation_field = deformation_field[logic]
        Z = Z[logic]
    return computeZernikeCoefficients(deformation_field, Z)

def basisDegreeVectors(L1, L2):
    degrees = []

    # Compute basis degrees for each component
    for h in range(0, L2 + 1):
        totalSPH = 2 * h + 1
        aux = np.floor(totalSPH / 2)
        for l in range(h, L1 + 1, 2):
            for m in range(totalSPH):
                degrees.append([l, h, h, m - aux])

    return np.asarray(degrees)

def reassociateCoefficients(Z, Zpp, Ap, A=None):
    # If A is None, invert the coefficients
    A = A if A is not None else np.zeros(Ap.shape)
    return np.transpose(computeInverse(Zpp.T @ Zpp) @ Zpp.T @ Z @ (A.T - Ap.T))

def applyDeformationField(map, mask, output, path, z_clnm, L1, L2, Rmax):
    writeZernikeFile(os.path.join(path, "z_clnm.txt"), z_clnm, L1, L2, Rmax)
    params = '-i %s --mask %s --step 1 --blobr 2 -o %s --clnm %s' % \
             (os.path.join(path, map), os.path.join(path, mask),
              os.path.join(path, output), os.path.join(path, "z_clnm.txt"))
    xmipp3.Plugin.runXmippProgram('xmipp_volume_apply_coefficient_zernike3d', params)


############## General functions ##############
def inscribedRadius(atoms):
    # Center Atoms with the Origin
    atoms_centered = atoms - np.mean(atoms, axis=0)

    # Maximum distance to the origin (inscribed radius)
    atoms_r = np.amax(np.linalg.norm(atoms_centered, axis=1))

    return 1.1 * atoms_r

def computeInverse(matrix):
    tol = np.amax(matrix) * np.amax(np.array(matrix.shape)) * 1e-10  # Probably -6 (for maps) -8 (for PDBs)
    u, s, vh = np.linalg.svd(matrix)

    for idx in range(len(s)):
        s_i = s[idx]
        if np.abs(s_i) > tol:
            s[idx] = 1.0 / s_i
        else:
            s[idx] = 0.0

    return np.dot(vh.T * s, u.T)

def centerOfMass(coords):
    return np.mean(coords, axis=0)

def computeRMSD(coords_1, coords_2):
    return np.sqrt(np.mean((coords_1 - coords_2) ** 2))


############## Functions to handle maps ##############
def readMap(file):
    if pwutils.getExt(file) == ".mrc":
        file += ":mrc"
    return ImageHandler().read(file).getData()

def saveMap(map, file):
    ImageHandler().write(map, file)

def getXmippOrigin(map):
    return np.asarray([int(0.5 * map.shape[2]),
                       int(0.5 * map.shape[1]),
                       int(0.5 * map.shape[0])])

def maskMapOtsu(map):
    thr = filters.threshold_otsu(map)
    mask = map > thr
    return mask.astype(int)

def maskMapLi(map):
    thr = filters.threshold_li(map)
    mask = map > thr
    return mask.astype(int)

def maskMapYen(map):
    thr = filters.threshold_yen(map)
    mask = map > thr
    return mask.astype(int)

def extractBorder(mask, erosion_steps=1):
    if erosion_steps == 1:
        return mask - binary_erosion(mask, iterations=1).astype(int)
    else:
        return binary_erosion(mask, iterations=erosion_steps)\
               - binary_erosion(mask, iterations=erosion_steps+1).astype(int)

def extractSkeleton(mask, filter=None):
    if filter is not None:
        mask = gaussian_filter(mask.astype(float), sigma=filter)
        mask = mask > 0.001
        mask = mask.astype(int)
    return skeletonize(mask)

def disconnectHetRegions(mask_1, mask_2, dist_thr):
    dist_mask_1 = mask_1 * distance_transform_edt(1 - mask_2)
    dist_mask_1_close = np.logical_and(dist_mask_1 <= dist_thr, dist_mask_1 > 0)
    dist_mask_1_far = dist_mask_1 > dist_thr
    return dist_mask_1_close, dist_mask_1_far

def associateBorderAndSkeleton(border_coords, skeleton_coords):
    tree = KDTree(skeleton_coords)
    _, indeces = tree.query(border_coords, k=1)
    return indeces.reshape(-1)

def improveCCMask(mask, iterations, ball_size=5):
    ball_elem = ball(ball_size)
    mask_closed = binary_closing(mask, structure=ball_elem, iterations=iterations)
    return mask_closed - binary_erosion(mask_closed, iterations=1).astype(int)

def reextractComponent(mask_1, mask_2):
    mask_1_hull = convex_hull_image(mask_1)
    mask_1 = mask_1_hull * mask_2
    return mask_1.astype(int)

def removeSmallCC(mask, area_thr):
    labels = label(mask)
    if (labels.max() == 0):  # if no components
        return []
    size_ccs = np.bincount(labels.flat)[1:]
    keep_ccs = np.asarray(np.where(size_ccs > area_thr)).reshape(-1) + 1
    largestsCC = np.zeros(mask.shape)
    for keep_cc in keep_ccs:
        CC = labels == keep_cc
        largestsCC += CC.astype(int) * keep_cc
    return largestsCC

def extractCC(mask, level):
    CC = mask == level
    return CC.astype(int)

def matchCC(mask_1, mask_2):
    levels_1 = np.unique(mask_1)[1:]
    levels_2 = np.unique(mask_2)[1:]
    associations = []
    for level_1 in levels_1:
        coords_1 = getCoordsAtLevel(mask_1, level_1)
        dist_cost = getXmippOrigin(mask_1)[0]
        for level_2 in levels_2:
            coords_2 = getCoordsAtLevel(mask_2, level_2)
            dist_cost_trial = computeRMSD(centerOfMass(coords_1), centerOfMass(coords_2))
            if dist_cost_trial < dist_cost:
                dist_cost = dist_cost_trial
                level_pair = [level_1, level_2]
        associations.append(level_pair)
    return associations

def getCoordsAtLevel(map, level):
    coords = np.asarray(np.where(map == level))
    # Coordinates should be in X,Y,Z format (Xmipp returns them in Z,Y,X)
    coords = np.transpose(np.asarray([coords[2, :], coords[1, :], coords[0, :]]))
    return coords

def coordsInMask(coords, mask):
    logic = mask[coords[:, 2], coords[:, 1], coords[:, 0]]
    return logic.astype(bool)

def alignMapsChimeraX(map_file_1, map_file_2, global_search=None, output_map=None):
    scriptFile = 'fitmap_transformation.cxc'
    OPEN_FILE = "open %s\n"
    VOXEL_SIZE = "volume #%d voxelSize %f\n"
    FITMAP = "fitmap #1 inMap #2\n"
    with open(scriptFile, 'w') as fhCmd:
        fhCmd.write(OPEN_FILE % map_file_1)
        fhCmd.write(OPEN_FILE % map_file_2)
        fhCmd.write(VOXEL_SIZE % (1, 1))
        fhCmd.write(VOXEL_SIZE % (2, 1))
        if global_search:
            FITMAP = FITMAP.rstrip("\n") + " search {}".format(global_search) + "\n"
        fhCmd.write(FITMAP)
        fhCmd.write("save transformation.positions #1\n")
        fhCmd.write("volume resample #1 onGrid #2\n")
        if output_map:
            fhCmd.write("save %s #3\n" % output_map)
        else:
            fhCmd.write("save start_aligned.mrc #3\n")
        fhCmd.write("exit\n")

    chimera_home = os.environ.get("CHIMERA_HOME")
    program = os.path.join(chimera_home, 'bin', os.path.basename("ChimeraX"))
    cmd = program + ' --nogui "%s"' % scriptFile
    call(cmd, shell=True, env=Chimera.getEnviron(), cwd=os.getcwd())

    with open('transformation.positions') as f:
        line = f.readline()
        line = line.split(",")[1:]
        Tr = np.array(line).reshape(3, 4)
        Tr = Tr.astype(np.float)
        Tr = np.vstack([Tr, np.array([0, 0, 0, 1])])

    if output_map:
        map_1_algn = readMap(output_map)
    else:
        map_1_algn = readMap("start_aligned.mrc")

    return map_1_algn, Tr


############## Functions to handle point clouds ##############
def icp(coords_1, coords_2):
    coords_1 = np.hstack([coords_1, np.ones([coords_1.shape[0], 1])])
    coords_2 = np.hstack([coords_2, np.ones([coords_2.shape[0], 1])])
    shift = np.mean(coords_2, axis=0) - np.mean(coords_1, axis=0)
    Tr = np.eye(4)
    Tr[0, 3] = shift[0]
    Tr[1, 3] = shift[1]
    Tr[2, 3] = shift[2]
    tree = KDTree(coords_2)
    for _ in range(100):
        coords_step = np.transpose(Tr @ coords_1.T)
        _, indeces = tree.query(coords_step, k=1)
        neighbours = coords_2[indeces.reshape(-1)]
        M = superimposition_matrix(coords_step.T, neighbours.T, usesvd=True)
        Tr = M @ Tr

    return Tr

def applyTransformation(coords, Tr, order=None):
    if order == "xmipp":
        coords = np.transpose(np.vstack([coords[:, 2], coords[:, 1], coords[:, 0], np.ones([coords.shape[0]])]))
        coords_tr = np.transpose(Tr @ coords.T)[:, :3]
        coords = np.transpose(np.vstack([coords[:, 2], coords[:, 1], coords[:, 0], coords[:, 3]]))
        coords_tr = np.transpose(np.vstack([coords_tr[:, 2], coords_tr[:, 1], coords_tr[:, 0]]))
    else:
        coords = np.hstack([coords, np.ones([coords.shape[0], 1])])
        coords_tr = np.transpose(Tr @ coords.T)[:, :3]
    return coords_tr, coords_tr - coords[:, :3]

def matchCoords(coords_1, coords_2, mutual=True):
    tree_1 = KDTree(coords_1)
    tree_2 = KDTree(coords_2)

    if mutual:
        dist_1, idx_1 = tree_1.query(coords_2, k=1)
        dist_2, idx_2 = tree_2.query(coords_1, k=1)

        idx_1 = idx_1.reshape(-1)
        dist_1 = dist_1.reshape(-1)
        idx_2 = idx_2.reshape(-1)

        dist_thr = 5

        logic_idx_1 = dist_1 <= dist_thr

        match_1 = []
        match_2 = []
        # p are coords_2 indexes
        for p in range(len(logic_idx_1)):
            if p == idx_2[idx_1[p]] and logic_idx_1[p] == 1:
                match_2.append(p)
                match_1.append(idx_1[p])

        return np.asarray(match_1), np.asarray(match_2)
    else:
        _, idx_2 = tree_2.query(coords_1, k=1)

        return np.asarray(range(coords_1.shape[0])), idx_2.reshape(-1)
