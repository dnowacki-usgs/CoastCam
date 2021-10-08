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


# %%
camera = 'both'
product = 'timex'

extrinsic_cal_files = ['/Users/dnowacki/Projects/ak/py/brw_extrinsic_c1.json',
                       '/Users/dnowacki/Projects/ak/py/brw_extrinsic_c2.json',]
intrinsic_cal_files = ['/Users/dnowacki/Projects/ak/py/brw_intrinsic_c1.json',
                       '/Users/dnowacki/Projects/ak/py/brw_intrinsic_c2.json',]

local_origin = {'x': 0,'y':0, 'angd': 0}

metadata= {'name': 'Utqiagvik',
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
    xmin = 0
    xmax = 300
    ymin = 0
    ymax = 250
    dx = 0.25
    dy = 0.25
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

    fildir = '/Volumes/Backstaff/field/brw/'
    if camera is 'both':
        image_files = [fildir + 'products/' + t + '.c1.' + product + '.jpg',
                       fildir + 'products/' + t + '.c2.' + product + '.jpg']
        # print(image_files)
        c1ref = skimage.io.imread(image_files[0])
        c2src = skimage.io.imread(image_files[1])
        # c2matched = match_histograms(c2src, c1ref, multichannel=True)
    else:
        image_files = [fildir + 'products/' + t + '.' + camera + '.' + product + '.jpg']

    rectified_image = rectifier.rectify_images(metadata, [c1ref, c2src], intrinsics_list, extrinsics_list, local_origin)
    ofile = fildir + 'proc/rect/' + product + '/' + t + '.' + camera + '.' + product + '.rect.png'
    imageio.imwrite(ofile, np.flip(rectified_image, 0), format='png', optimize=True)

    # rectified_image_matched = rectifier.rectify_images(metadata, [c1ref, c2matched], intrinsics_list, extrinsics_list, local_origin)
    # ofile = fildir + 'proc/rect/' + t + '.' + camera + '.' + product + '.rect.matched.png'
    # imageio.imwrite(ofile, np.flip(rectified_image_matched, 0), format='png', optimize=True)

# ts1 = [os.path.basename(x).split('.')[0] for x in glob.glob('/Volumes/Backstaff/field/unk/products/163[0-9][3-9][0-9][6-9]*c1.'+ product + '.jpg')]
# ts2 = [os.path.basename(x).split('.')[0] for x in glob.glob('/Volumes/Backstaff/field/unk/products/163[0-9][3-9][0-9][6-9]*c2.' + product + '.jpg')]

ts1 = [os.path.basename(x).split('.')[0] for x in glob.glob('/Volumes/Backstaff/field/brw/products/*c1.'+ product + '.jpg')]
ts2 = [os.path.basename(x).split('.')[0] for x in glob.glob('/Volumes/Backstaff/field/brw/products/*c2.' + product + '.jpg')]

if camera is 'c1':
    ts = ts1
elif camera is 'c2':
    ts = ts2
elif camera is 'both':
    ts = list(set(ts1) & set(ts2))

ts  = [x for x in ts if int(x) >= 1630294200 ] # only good images from when the camera was aimed correctly


# with open('/Users/dnowacki/Downloads/source_times.txt', 'w') as f:
#     for item in ts:
#         f.write(f"{item}\n")

# with open('/Volumes/Backstaff/field/brw/proc/rect/' + product + '/done.txt') as f:
#     tsdone = [line.rstrip().split('.')[0] for line in f]
# # this will get what remains to be done
# print('length of ts', len(ts))
# print('length of the done list', len(tsdone))
# print(set(ts) == set(tsdone))

# ts = set(ts) ^ set(tsdone)
set(ts)
# %%
# tsnew = []
# for n in ts:
#     if n not in tsdone:
#         tsnew.append(n)
# print('values in t not already in the done list', len(tsnew))
# ts = tsnew

print('***', len(ts))
print(product, camera)
# %%
print(product, camera)
# t = ts[0]
# n9468333['v'][np.argmin(np.abs(pd.DatetimeIndex(n9468333.time.values) - pd.to_datetime(t, unit='s')))].values
Parallel(n_jobs=4, backend='multiprocessing')(
    delayed(lazyrun)(
        metadata, intrinsics_list, extrinsics_list, local_origin, t,
        1) # set z to 1, which is the approx water elevation from the beach survey
        for t in ts
        #n9468333['v'][np.argmin(np.abs(pd.DatetimeIndex(n9468333.time.values) - pd.to_datetime(t, unit='s')))].values)
        )
# [lazyrun(metadata, intrinsics_list, extrinsics_list, local_origin, t, n9468333['v'][np.argmin(np.abs(pd.DatetimeIndex(n9468333.time.values) - pd.to_datetime(t, unit='s')))].values) for t in ts[0:2]]
