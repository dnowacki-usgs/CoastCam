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
n9468333 = xr.load_dataset('/Volumes/Backstaff/field/unk/n9468333.nc')

# %%
camera = 'both'
product = 'timex'

extrinsic_cal_files = ['/Users/dnowacki/Projects/ak/py/unk_extrinsic_c1.json',
                       '/Users/dnowacki/Projects/ak/py/unk_extrinsic_c2.json',]
intrinsic_cal_files = ['/Users/dnowacki/Projects/ak/py/unk_intrinsic_c1.json',
                       '/Users/dnowacki/Projects/ak/py/unk_intrinsic_c2.json',]

local_origin = {'x': 0,'y':0, 'angd': 0}

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
    xmin = 0
    xmax = 250
    ymin = -150
    ymax = 40
    dx = 0.1
    dy = 0.1
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

    fildir = '/Volumes/Backstaff/field/unk/'
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
    ofile = fildir + 'proc/rect/' + t + '.' + camera + '.' + product + '.rect.png'
    imageio.imwrite(ofile, np.flip(rectified_image, 0), format='png', optimize=True)

    # rectified_image_matched = rectifier.rectify_images(metadata, [c1ref, c2matched], intrinsics_list, extrinsics_list, local_origin)
    # ofile = fildir + 'proc/rect/' + t + '.' + camera + '.' + product + '.rect.matched.png'
    # imageio.imwrite(ofile, np.flip(rectified_image_matched, 0), format='png', optimize=True)

ts1 = [os.path.basename(x).split('.')[0] for x in glob.glob('/Volumes/Backstaff/field/unk/products/*c1.'+ product + '.jpg')]
ts2 = [os.path.basename(x).split('.')[0] for x in glob.glob('/Volumes/Backstaff/field/unk/products/*c2.' + product + '.jpg')]

if camera is 'c1':
    ts = ts1
elif camera is 'c2':
    ts = ts2
elif camera is 'both':
    ts = list(set(ts1) & set(ts2))

# with open('/Users/dnowacki/Downloads/source_times.txt', 'w') as f:
#     for item in ts:
#         f.write(f"{item}\n")

with open('/Users/dnowacki/Downloads/needs_processing.txt') as f:
    ts = [line.rstrip() for line in f]

# %%
print(camera)
# t = ts[0]
# n9468333['v'][np.argmin(np.abs(pd.DatetimeIndex(n9468333.time.values) - pd.to_datetime(t, unit='s')))].values
Parallel(n_jobs=4, backend='multiprocessing')(
    delayed(lazyrun)(
        metadata, intrinsics_list, extrinsics_list, local_origin, t, n9468333['v'][np.argmin(np.abs(pd.DatetimeIndex(n9468333.time.values) - pd.to_datetime(t, unit='s')))].values) for t in ts)
# [lazyrun(metadata, intrinsics_list, extrinsics_list, local_origin, t, n9468333['v'][np.argmin(np.abs(pd.DatetimeIndex(n9468333.time.values) - pd.to_datetime(t, unit='s')))].values) for t in ts[0:2]]
