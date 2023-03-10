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


import math
import numpy as np

import tensorflow as tf
# from tensorflow.python.ops.numpy_ops import deg2rad


# axis sequences for Euler angles
_NEXT_AXIS = [1, 2, 0, 1]

# map axes strings to/from tuples of inner axis, parity, repetition, frame
_AXES2TUPLE = {
    'sxyz': (0, 0, 0, 0), 'sxyx': (0, 0, 1, 0), 'sxzy': (0, 1, 0, 0),
    'sxzx': (0, 1, 1, 0), 'syzx': (1, 0, 0, 0), 'syzy': (1, 0, 1, 0),
    'syxz': (1, 1, 0, 0), 'syxy': (1, 1, 1, 0), 'szxy': (2, 0, 0, 0),
    'szxz': (2, 0, 1, 0), 'szyx': (2, 1, 0, 0), 'szyz': (2, 1, 1, 0),
    'rzyx': (0, 0, 0, 1), 'rxyx': (0, 0, 1, 1), 'ryzx': (0, 1, 0, 1),
    'rxzx': (0, 1, 1, 1), 'rxzy': (1, 0, 0, 1), 'ryzy': (1, 0, 1, 1),
    'rzxy': (1, 1, 0, 1), 'ryxy': (1, 1, 1, 1), 'ryxz': (2, 0, 0, 1),
    'rzxz': (2, 0, 1, 1), 'rxyz': (2, 1, 0, 1), 'rzyz': (2, 1, 1, 1)}

_TUPLE2AXES = dict((v, k) for k, v in _AXES2TUPLE.items())

def euler_matrix(alpha, beta, gamma):
    A = []

    ca = tf.cos(deg2rad(alpha))
    sa = tf.sin(deg2rad(alpha))
    cb = tf.cos(deg2rad(beta))
    sb = tf.sin(deg2rad(beta))
    cg = tf.cos(deg2rad(gamma))
    sg = tf.sin(deg2rad(gamma))

    cc = cb * ca
    cs = cb * sa
    sc = sb * ca
    ss = sb * sa

    A.append([cg * cc - sg * sa, cg * cs + sg * ca, -cg * sb])
    A.append([-sg * cc - cg * sa, -sg * cs + cg * ca, sg * sb])
    A.append([sc, ss, cb])
    return tf.stack(A)

############## Zernike related functions ##############
def computeZernikes3D(l1, n, l2, m, pos, r_max):

    # General variables
    pos_r = pos / r_max
    xr, yr, zr = pos_r[:, 0], pos_r[:, 1], pos_r[:, 2]
    xr2, yr2, zr2 = xr * xr, yr * yr, zr * zr
    r = np.linalg.norm(pos, axis=1) / r_max
    r2 = r * r

    # Variables needed for l2 >= 5
    tht = np.arctan2(yr, xr)
    phi = np.arctan2(zr, np.sqrt(xr2 + yr2))
    sinth = np.sin(abs(m)*phi)
    costh = np.cos(tht)
    cosph = np.cos(abs(m)*phi)
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
            Y = 0.222*cosph*np.power(1.0 - 1.0*costh2, 0.5)*(86.62*costh2*costh2*costh - 78.75*costh2*costh + 13.12*costh)
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
            Y = sinth*np.sqrt((costh2)*-1.0+1.0)*(costh*1.96875E+1-(costh*costh2)*2.165625E+2+(costh*costh2*costh2)*5.630625E+2-(costh*costh2*costh2*costh2)*4.021875E+2)*-1.938511038201796E-1
        elif m == 0:
            Y = (costh2)*-1.144933081936324E+1+(costh2*costh2)*6.297131950652692E+1-(costh2*costh2*costh2)*1.091502871445846E+2+(costh2*costh2*costh2*costh2)*5.847336811327841E+1+3.180369672045344E-1
        elif m == 1:
            Y = cosph*np.sqrt((costh2)*-1.0+1.0)*(costh*1.96875E+1-(costh*costh2)*2.165625E+2+(costh*costh2*costh2)*5.630625E+2-(costh*costh2*costh2*costh2)*4.021875E+2)*-1.938511038201796E-1
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
            Y = sinth*np.sqrt((costh2)*-1.0+1.0)*((costh2)*-1.0828125E+2+(costh2*costh2)*7.03828125E+2-(costh2*costh2*costh2)*1.40765625E+3+(costh2*costh2*costh2*costh2)*8.546484375E+2+2.4609375)*1.833013280775049E-1
        elif m == 0:
            Y = costh*3.026024588281871-(costh*costh2)*4.438169396144804E+1+(costh*costh2*costh2)*1.730886064497754E+2-(costh*costh2*costh2*costh2)*2.472694377852604E+2+(costh*costh2*costh2*costh2*costh2)*1.167661233986728E+2
        elif m == 1:
            Y = cosph*np.sqrt((costh2)*-1.0+1.0)*((costh2)*-1.0828125E+2+(costh2*costh2)*7.03828125E+2-(costh2*costh2*costh2)*1.40765625E+3+(costh2*costh2*costh2*costh2)*8.546484375E+2+2.4609375)*1.833013280775049E-1
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
            Y = sinth*np.sqrt((costh*costh)*-1.0+1.0)*(costh*2.70703125E+1-(costh*costh*costh)*4.6921875E+2+(costh*costh*costh*costh*costh)*2.111484375E+3-(costh*costh*costh*costh*costh*costh*costh)*3.41859375E+3+(costh*costh*costh*costh*costh*costh*costh*costh*costh)*1.8042578125E+3)*1.743104285446861E-1
        elif m == 0:
            Y = (costh*costh)*1.749717715557199E+1-(costh*costh*costh*costh)*1.516422020150349E+2+(costh*costh*costh*costh*costh*costh)*4.549266060441732E+2-(costh*costh*costh*costh*costh*costh*costh*costh)*5.524108787681907E+2+np.power(costh,1.0E+1)*2.332401488134637E+2-3.181304937370442E-1
        elif m == 1:
            Y = cosph*np.sqrt((costh*costh)*-1.0+1.0)*(costh*2.70703125E+1-(costh*costh*costh)*4.6921875E+2+(costh*costh*costh*costh*costh)*2.111484375E+3-(costh*costh*costh*costh*costh*costh*costh)*3.41859375E+3+(costh*costh*costh*costh*costh*costh*costh*costh*costh)*1.8042578125E+3)*1.743104285446861E-1
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
            Y = sinth*np.sqrt((costh*costh)*-1.0+1.0)*((costh*costh)*1.7595703125E+2-(costh*costh*costh*costh)*1.7595703125E+3+(costh*costh*costh*costh*costh*costh)*5.9825390625E+3-(costh*costh*costh*costh*costh*costh*costh*costh)*8.11916015625E+3+np.power(costh,1.0E+1)*3.78894140625E+3-2.70703125)*1.665279049125274E-1
        elif m == 0:
            Y = costh*-3.662285987506039+(costh*costh*costh)*7.934952972922474E+1-(costh*costh*costh*costh*costh)*4.760971783753484E+2+(costh*costh*costh*costh*costh*costh*costh)*1.156236004628241E+3-(costh*costh*costh*costh*costh*costh*costh*costh*costh)*1.220471338216215E+3+np.power(costh,1.1E+1)*4.65998147319071E+2
        elif m == 1:
            Y = cosph*np.sqrt((costh*costh)*-1.0+1.0)*((costh*costh)*1.7595703125E+2-(costh*costh*costh*costh)*1.7595703125E+3+(costh*costh*costh*costh*costh*costh)*5.9825390625E+3-(costh*costh*costh*costh*costh*costh*costh*costh)*8.11916015625E+3+np.power(costh,1.0E+1)*3.78894140625E+3-2.70703125)*1.665279049125274E-1
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
            Y = sinth*np.sqrt((costh*costh)*-1.0+1.0)*(costh*3.519140625E+1-(costh*costh*costh)*8.7978515625E+2+(costh*costh*costh*costh*costh)*5.9825390625E+3-(costh*costh*costh*costh*costh*costh*costh)*1.62383203125E+4+(costh*costh*costh*costh*costh*costh*costh*costh*costh)*1.894470703125E+4-np.power(costh,1.1E+1)*7.92233203125E+3)*-1.597047270888652E-1
        elif m == 0:
            Y = (costh*costh)*-2.481828104582382E+1+(costh*costh*costh*costh)*3.102285130722448E+2-(costh*costh*costh*costh*costh*costh)*1.40636925926432E+3+(costh*costh*costh*costh*costh*costh*costh*costh)*2.862965992070735E+3-np.power(costh,1.0E+1)*2.672101592600346E+3+np.power(costh,1.2E+1)*9.311869186330587E+2+3.181830903313312E-1
        elif m == 1:
            Y = cosph*np.sqrt((costh*costh)*-1.0+1.0)*(costh*3.519140625E+1-(costh*costh*costh)*8.7978515625E+2+(costh*costh*costh*costh*costh)*5.9825390625E+3-(costh*costh*costh*costh*costh*costh*costh)*1.62383203125E+4+(costh*costh*costh*costh*costh*costh*costh*costh*costh)*1.894470703125E+4-np.power(costh,1.1E+1)*7.92233203125E+3)*-1.597047270888652E-1
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

    degrees = basisDegreeVectors(L1, L2)
    basis = [computeZernikes3D(degrees[idx, 0], degrees[idx, 1], degrees[idx, 2], degrees[idx, 3],
             pos, r) for idx in range(degrees.shape[0])]
    basis = np.hstack(basis)

    return basis

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

def getXmippOrigin(boxsize):
    return np.asarray([int(0.5 * boxsize),
                       int(0.5 * boxsize),
                       int(0.5 * boxsize)])

def euler_from_matrix(matrix):
    # Only valid for Xmipp axes szyz
    firstaxis, parity, repetition, frame = (2, 1, 1, 0)
    _EPS = np.finfo(float).eps * 4.0

    i = firstaxis
    j = _NEXT_AXIS[i + parity]
    k = _NEXT_AXIS[i - parity + 1]

    M = np.array(matrix, dtype=np.float64, copy=False)[:3, :3]
    if repetition:
        sy = math.sqrt(M[i, j] * M[i, j] + M[i, k] * M[i, k])
        if sy > _EPS:
            ax = math.atan2(M[i, j], M[i, k])
            ay = math.atan2(sy, M[i, i])
            az = math.atan2(M[j, i], -M[k, i])
        else:
            ax = math.atan2(-M[j, k], M[j, j])
            ay = math.atan2(sy, M[i, i])
            az = 0.0
    else:
        cy = math.sqrt(M[i, i] * M[i, i] + M[j, i] * M[j, i])
        if cy > _EPS:
            ax = math.atan2(M[k, j], M[k, k])
            ay = math.atan2(-M[k, i], cy)
            az = math.atan2(M[j, i], M[i, i])
        else:
            ax = math.atan2(-M[j, k], M[j, j])
            ay = math.atan2(-M[k, i], cy)
            az = 0.0

    if parity:
        ax, ay, az = -ax, -ay, -az
    if frame:
        ax, az = az, ax
    return ax, ay, az

def xmippEulerFromMatrix(matrix):
    return -np.rad2deg(euler_from_matrix(matrix))

def euler_matrix_row(alpha, beta, gamma, row, batch_size):
    A = []

    for idx in range(batch_size):
        ca = tf.cos(tf.gather(alpha, idx, axis=0) * (np.pi / 180.0))
        sa = tf.sin(tf.gather(alpha, idx, axis=0) * (np.pi / 180.0))
        cb = tf.cos(tf.gather(beta, idx, axis=0) * (np.pi / 180.0))
        sb = tf.sin(tf.gather(beta, idx, axis=0) * (np.pi / 180.0))
        cg = tf.cos(tf.gather(gamma, idx, axis=0) * (np.pi / 180.0))
        sg = tf.sin(tf.gather(gamma, idx, axis=0) * (np.pi / 180.0))

        cc = cb * ca
        cs = cb * sa
        sc = sb * ca
        ss = sb * sa

        if row == 1:
            A.append([cg * cc - sg * sa, cg * cs + sg * ca, -cg * sb])
            # A.append([cg * cc - sg * sa, -sg * cc - cg, sc])
        elif row == 2:
            A.append([-sg * cc - cg * sa, -sg * cs + cg * ca, sg * sb])
            # A.append([cg * cs + sg * ca, -sg * cs + cg * ca, sg * ss])
        elif row == 3:
            A.append([sc, ss, cb])
            # A.append([-cg * sb, sg * ss, cb])

    return tf.stack(A)

def euler_matrix_batch(alpha, beta, gamma):

    ca = tf.cos(alpha * (np.pi / 180.0))[:, None]
    sa = tf.sin(alpha * (np.pi / 180.0))[:, None]
    cb = tf.cos(beta * (np.pi / 180.0))[:, None]
    sb = tf.sin(beta * (np.pi / 180.0))[:, None]
    cg = tf.cos(gamma * (np.pi / 180.0))[:, None]
    sg = tf.sin(gamma * (np.pi / 180.0))[:, None]

    cc = cb * ca
    cs = cb * sa
    sc = sb * ca
    ss = sb * sa

    row_1 = tf.concat([cg * cc - sg * sa, cg * cs + sg * ca, -cg * sb], axis=1)
    # A.append([cg * cc - sg * sa, -sg * cc - cg, sc])

    row_2 = tf.concat([-sg * cc - cg * sa, -sg * cs + cg * ca, sg * sb], axis=1)
    # A.append([cg * cs + sg * ca, -sg * cs + cg * ca, sg * ss])

    row_3 = tf.concat([sc, ss, cb], axis=1)
    # A.append([-cg * sb, sg * ss, cb])

    return row_1, row_2, row_3

def ctf_freqs(shape, d=1.0, full=False):
    """
    :param shape: Shape tuple.
    :param d: Frequency spacing in inverse Å (1 / pixel size).
    :param full: When false, return only unique Fourier half-space for real data.
    """
    if full:
        xfrq = tf.constant(np.fft.fftfreq(shape[1]), dtype=tf.float32)
    else:
        xfrq = tf.constant(np.fft.rfftfreq(shape[1]), dtype=tf.float32)
    x, y = tf.meshgrid(xfrq, tf.constant(np.fft.fftfreq(shape[0]), dtype=tf.float32))
    rho = tf.sqrt(x ** 2. + y ** 2.)
    a = tf.atan2(y, x)
    s = rho * d
    return s, a


# @numba.jit(cache=True, nopython=True, nogil=True)
def eval_ctf(s, a, def1, def2, angast=0, phase=0, kv=300, ac=0.1, cs=2.0, bf=0, lp=0):
    """
    :param s: Precomputed frequency grid for CTF evaluation.
    :param a: Precomputed frequency grid angles.
    :param def1: 1st prinicipal underfocus distance (Å).
    :param def2: 2nd principal underfocus distance (Å).
    :param angast: Angle of astigmatism (deg) from x-axis to azimuth.
    :param phase: Phase shift (deg).
    :param kv:  Microscope acceleration potential (kV).
    :param ac:  Amplitude contrast in [0, 1.0].
    :param cs:  Spherical aberration (mm).
    :param bf:  B-factor, divided by 4 in exponential, lowpass positive.
    :param lp:  Hard low-pass filter (Å), should usually be Nyquist.
    """
    angast = angast * (np.pi / 180.0)
    kv = kv * 1e3
    cs = cs * 1e7
    lamb = 12.2643247 / tf.sqrt(kv * (1. + kv * 0.978466e-6))
    def_avg = -(def1 + def2) * 0.5
    def_dev = -(def1 - def2) * 0.5
    k1 = np.pi / 2. * 2. * lamb
    k2 = np.pi / 2. * cs * lamb ** 3.
    k3 = tf.sqrt(1. - ac ** 2.)
    k4 = bf / 4.  # B-factor, follows RELION convention.
    k5 = phase * (np.pi / 180.0)  # Phase shift.
    if lp != 0:  # Hard low- or high-pass.
        s *= s <= (1. / lp)
    s_2 = s ** 2.
    s_4 = s_2 ** 2.
    dZ = def_avg[:, None, None] + def_dev[:, None, None] * (tf.cos(2. * (a - angast[:, None, None])))
    gamma = (k1 * dZ * s_2) + (k2[:, None, None] * s_4) - k5
    # dZ = def_avg + def_dev * (tf.cos(2. * (a - angast)))
    # gamma = (k1 * dZ * s_2) + (k2 * s_4) - k5
    ctf = -(k3 * tf.sin(gamma) - ac * tf.cos(gamma))
    if bf != 0:  # Enforce envelope.
        ctf *= tf.exp(-k4 * s_2)
    return ctf

def computeCTF(defocusU, defocusV, defocusAngle, cs, kv, sr, pad_factor, img_shape, batch_size, applyCTF):
    if applyCTF == 1:
        # s, a = ctf_freqs([img_shape[0], img_shape[0]], 1 / sr)
        # ctf = []
        # for idx in range(batch_size):
        #     def1, def2, angast, cs_var = defocusU[idx], defocusV[idx], defocusAngle[idx], cs[idx]
        #     ctf_img = eval_ctf(s, a, def1, def2, angast=angast, cs=cs_var, kv=kv)
        #     ctf.append(tf.signal.fftshift(ctf_img[:, :img_shape[1]]))
        # return tf.stack(ctf)

        s, a = ctf_freqs([pad_factor * img_shape[0], pad_factor * img_shape[0]], 1 / sr)
        s, a = tf.tile(s[None, :, :], [batch_size, 1, 1]), tf.tile(a[None, :, :], [batch_size, 1, 1])
        ctf = eval_ctf(s, a, defocusU, defocusV, angast=defocusAngle, cs=cs, kv=kv)
        ctf = tf.signal.fftshift(ctf)
        return ctf

    else:
        # size_aux = int(0.5 * pad_factor * img_shape[0] + 1)
        return tf.ones([batch_size, pad_factor * img_shape[0], pad_factor * img_shape[1] - (pad_factor - 1)], dtype=tf.float32)
        # return tf.ones([batch_size, img_shape[0], img_shape[1]], dtype=tf.float32)

def fft_pad(imgs, size_x, size_y):
    padded_imgs = tf.image.resize_with_crop_or_pad(imgs, size_x, size_y)
    ft_images = tf.signal.fftshift(tf.signal.rfft2d(padded_imgs[:, :, :, 0]))
    return ft_images

def ifft_pad(ft_imgs, size_x, size_y):
    padded_imgs = tf.signal.irfft2d(tf.signal.ifftshift(ft_imgs))[..., None]
    imgs = tf.image.resize_with_crop_or_pad(padded_imgs, size_x, size_y)
    return imgs

def gramSchmidt(r):
    c1 = tf.nn.l2_normalize(r[:, :3], axis=-1)
    c2 = tf.nn.l2_normalize(r[:, 3:] - dot(c1, r[:, 3:]) * c1, axis=-1)
    c3 = tf.linalg.cross(c1, c2)
    return tf.stack([c1, c2, c3], axis=2)

def dot(a, b):
    return tf.reduce_sum(a * b, axis=-1, keepdims=True)
