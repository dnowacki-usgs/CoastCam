# %load_ext autoreload
# %autoreload 2
from pathlib import Path
import imageio
import numpy as np
import glob
import os
import matplotlib.pyplot as plt
# %matplotlib inline
import glob
import os
import shutil
import skimage.io
from skimage.exposure import match_histograms
from coastcam_funcs import json2dict
from calibration_crs import *
from rectifier_crs import *
import pandas as pd
from joblib import Parallel, delayed
import xarray as xr
# n9468333 = xr.load_dataset('/Volumes/Backstaff/field/unk/n9468333.nc')

# %%
camera = 'both'
product = 'timex'

extrinsic_cal_files = ['/Users/dnowacki/Projects/ak/py/field_extrinsic.json', ]
intrinsic_cal_files = ['/Users/dnowacki/Projects/ak/py/field_intrinsic.json',]

local_origin = {'x': 0,'y':0, 'angd':  0}

metadata= {'name': 'UNK',
           'serial_number': 1,
           'camera_number': camera,
           'calibration_date': 'xxxx-xx-xx',
           'coordinate_system': 'xyz'}

# read cal files and make lists of cal dicts
extrinsics_list = []
for f in extrinsic_cal_files:
    if camera in f or camera is 'both':
        extrinsics_list.append( json2dict(f) )
intrinsics_list = []
for f in intrinsic_cal_files:
    if camera in f or camera is 'both':
        intrinsics_list.append( json2dict(f) )
print(extrinsics_list)
print(intrinsics_list)

# check test for coordinate system
if metadata['coordinate_system'].lower() == 'xyz':
    print('Extrinsics are local coordinates')
elif metadata['coordinate_system'].lower() == 'geo':
    print('Extrinsics are in world coordinates')
else:
    print('Invalid value of coordinate_system: ',metadata['coordinate_system'])

print(extrinsics_list[0])
print(extrinsics_list[0]['y']-local_origin['y'])

calibration = CameraCalibration(metadata,intrinsics_list[0],extrinsics_list[0],local_origin)
print(calibration.local_origin)
print(calibration.world_extrinsics)
print(calibration.local_extrinsics)

def lazyrun(metadata, intrinsics_list, extrinsics_list, local_origin, t, z):
    print(t)
    """ coordinate system setup"""
    xmin = -35
    xmax = 45
    ymin = -110
    ymax = -20
    dx = .05
    dy = .05
    # z = 0 # z is now defined by water level input
    print(f'***Z: {z} m NAVD88')
    rectifier_grid = TargetGrid(
        [xmin, xmax],
        [ymin, ymax],
        dx,
        dy,
        z
    )

    rectifier = Rectifier(rectifier_grid)

    fildir = '/Users/dnowacki/projects/ak/field/'
    if camera is 'both':
        # image_files = [fildir + 'PICT0068_H_202105051930x4LsX.jpg']
        image_files = [fildir + 'PICT0049_H_202105051130d49kN.jpg']

        # print(image_files)
        c1ref = skimage.io.imread(image_files[0])
        # c2src = skimage.io.imread(image_files[1])
        # c2matched = match_histograms(c2src, c1ref, multichannel=True)
    else:
        image_files = [fildir + 'products/' + t + '.' + camera + '.' + product + '.jpg']

    rectified_image = rectifier.rectify_images(metadata, [c1ref], intrinsics_list, extrinsics_list, local_origin)
    plt.imshow(rectified_image)
    # ofile = fildir + 'rect.png'
    ofile = fildir + 'IRrect.png'
    imageio.imwrite(ofile, np.flip(rectified_image, 0), format='png', optimize=True)

    # rectified_image_matched = rectifier.rectify_images(metadata, [c1ref, c2matched], intrinsics_list, extrinsics_list, local_origin)
    # ofile = fildir + 'proc/rect/' + t + '.' + camera + '.' + product + '.rect.matched.png'
    # imageio.imwrite(ofile, np.flip(rectified_image_matched, 0), format='png', optimize=True)

# ts1 = [os.path.basename(x).split('.')[0] for x in glob.glob('/Volumes/Backstaff/field/unk/products/*c1.'+ product + '.jpg')]
# ts2 = [os.path.basename(x).split('.')[0] for x in glob.glob('/Volumes/Backstaff/field/unk/products/*c2.' + product + '.jpg')]
#
# if camera is 'c1':
#     ts = ts1
# elif camera is 'c2':
#     ts = ts2
# elif camera is 'both':
#     ts = list(set(ts1) & set(ts2))
#
# # with open('/Users/dnowacki/Downloads/source_times.txt', 'w') as f:
# #     for item in ts:
# #         f.write(f"{item}\n")
#
# with open('/Volumes/Backstaff/field/unk/proc/rect/' + product + '/done.txt') as f:
#     tsdone = [line.rstrip().split('.')[0] for line in f]
# # this will get what remains to be done
# print(len(ts))
# print(len(tsdone))
# print(set(ts) == set(tsdone))
# ts = set(ts) ^ set(tsdone)
# print('***', len(ts))
#
print(camera)
# t = ts[0]
# n9468333['v'][np.argmin(np.abs(pd.DatetimeIndex(n9468333.time.values) - pd.to_datetime(t, unit='s')))].values
lazyrun(metadata, intrinsics_list, extrinsics_list, local_origin, 5, 21.5)
# [lazyrun(metadata, intrinsics_list, extrinsics_list, local_origin, t, n9468333['v'][np.argmin(np.abs(pd.DatetimeIndex(n9468333.time.values) - pd.to_datetime(t, unit='s')))].values) for t in ts[0:2]]
# %%
fildir = '/Users/dnowacki/projects/ak/field/'
im = skimage.io.imread(fildir + 'rect.png')
plt.figure(figsize=(10,8))
plt.imshow(im, extent=(-35,45,-110,-20))
plt.grid()
plt.xlabel('x coordinate [m]')
plt.ylabel('y coordinate [m]')
