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
from tqdm import tqdm
n9468333 = xr.load_dataset('/Users/dnowacki/OneDrive - DOI/Alaska/unk/noaa/n9468333.nc')
import multiprocess as mp


# %%
fildir = '/Volumes/Argus/unk/'
camera = 'cx'
product = 'snap'

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
    if camera in f or camera == 'cx':
        extrinsics_list.append( json2dict(f) )
intrinsics_list = []
for f in intrinsic_cal_files:
    if camera in f or camera == 'cx':
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
    # metadata, intrinsics_list, extrinsics_list, local_origin, t, z = inputs
    print(t)
    if np.isnan(z):
        print('*** NaN detected in Z; skipping ***')
        return
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

    if camera == 'cx':
        c1bad = False
        c2bad = False
        image_files = [fildir + 'products/' + t + '.c1.' + product + '.jpg',
                       fildir + 'products/' + t + '.c2.' + product + '.jpg']
        # print(image_files)
        try:
            c1ref = skimage.io.imread(image_files[0])
        except (AttributeError, ValueError):
            print('could not process', image_files[0], '; RETURNING')
            return
        except FileNotFoundError:
            print('could not process', image_files[0], '; replacing with zeros')
            c1ref = np.zeros((1536, 2048, 3), dtype=np.uint8)
            c1bad = True

        try:
            c2src = skimage.io.imread(image_files[1])
        except (AttributeError, ValueError):
            print('could not process', image_files[1], '; RETURNING')
            return
        except FileNotFoundError:
            print('could not process', image_files[1], '; replacing with zeros')
            c2src = np.zeros((1536, 2048, 3), dtype=np.uint8)
            c2bad = True

        # c2matched = match_histograms(c2src, c1ref, multichannel=True)
    else:
        image_files = [fildir + 'products/' + t + '.' + camera + '.' + product + '.jpg']

    print(f"{c1bad=}, {c2bad=}")

    if c1bad:
        inimg = [c2src]
        intrinsics_list = [intrinsics_list[1]]
        extrinsics_list = [extrinsics_list[1]]
    elif c2bad:
        inimg = [c1ref]
        intrinsics_list = [intrinsics_list[0]]
        extrinsics_list = [extrinsics_list[0]]
    elif not c1bad and not c2bad:
        inimg = [c1ref, c2src]
    else:
        raise ValueError('no images')

    rectified_image = rectifier.rectify_images(metadata, inimg, intrinsics_list, extrinsics_list, local_origin)
    ofile = fildir + 'proc/rect/' + product + '/' + t + '.' + camera + '.' + product + '.png'
    print(ofile)
    imageio.imwrite(ofile, np.flip(rectified_image, 0), format='png', optimize=True)

    # rectified_image_matched = rectifier.rectify_images(metadata, [c1ref, c2matched], intrinsics_list, extrinsics_list, local_origin)
    # ofile = fildir + 'proc/rect/' + t + '.' + camera + '.' + product + '.rect.matched.png'
    # imageio.imwrite(ofile, np.flip(rectified_image_matched, 0), format='png', optimize=True)

# ts1 = [os.path.basename(x).split('.')[0] for x in glob.glob('/Volumes/Backstaff/field/unk/products/163[0-9][3-9][0-9][6-9]*c1.'+ product + '.jpg')]
# ts2 = [os.path.basename(x).split('.')[0] for x in glob.glob('/Volumes/Backstaff/field/unk/products/163[0-9][3-9][0-9][6-9]*c2.' + product + '.jpg')]

ts1 = [os.path.basename(x).split('.')[0] for x in glob.glob(fildir + 'products/*c1.'+ product + '.jpg')]
ts2 = [os.path.basename(x).split('.')[0] for x in glob.glob(fildir + 'products/*c2.' + product + '.jpg')]

if camera == 'c1':
    ts = ts1
elif camera == 'c2':
    ts = ts2
elif camera == 'cx':
    ts = list(set(ts1) | set(ts2)) # use OR instead of AND so we can have rectified images when just one camera was active


with open(fildir + 'proc/rect/' + product + '/done.txt', 'w') as f:
    for g in glob.glob(fildir + 'proc/rect/' + product + '/*png'):
        f.write(f"{g.split('/')[-1].split('.')[0]}\n")
    for g in glob.glob(fildir + 'proc/rect/' + product + '/dark/*png'):
        f.write(f"{g.split('/')[-1].split('.')[0]}\n")

with open(fildir + 'proc/rect/' + product + '/done.txt') as f:
    tsdone = [line.rstrip() for line in f]

# this will get what remains to be done
print('length of ts', len(ts))
print('length of the done list', len(tsdone))
print(set(ts) == set(tsdone))
len(set(tsdone))
# ts = set(ts) ^ set(tsdone)
# print('length of the todo list', len(ts))

# ts = set(ts) ^ set(tsdone)
# set(ts)
# %
print(len(ts))
tsnew = []
for n in ts:
    if n not in tsdone:
        tsnew.append(n)
print('values in t not already in the done list', len(tsnew))
ts = tsnew

print('***', len(ts))
print(product, camera)
# %%
print(product, camera)
# t = ts[0]
# n9468333['v'][np.argmin(np.abs(pd.DatetimeIndex(n9468333.time.values) - pd.to_datetime(t, unit='s')))].values

ds = xr.Dataset()
ds['time'] = xr.DataArray(pd.to_datetime(ts, unit='s'), dims='time')
ds['timestamp'] = xr.DataArray(ts, dims='time')
ds['wl'] = n9468333['water_level'].reindex_like(ds['time'], method='nearest', tolerance='10min')
ds = ds.sortby('time')

# Parallel(n_jobs=-4)(
#     delayed(lazyrun)(
#         metadata, intrinsics_list, extrinsics_list, local_origin, t,
#         ds['wl'][ds.timestamp == t].values)
#         for t in ts
#         )
def multi_run_wrapper(args):
   return lazyrun(*args)

tasks = [(metadata, intrinsics_list, extrinsics_list, local_origin, t, ds['wl'][ds.timestamp == t].values) for t in ts]
with mp.Pool(mp.cpu_count()-4) as pool:
    for _ in tqdm(pool.imap_unordered(multi_run_wrapper, tasks), total=len(tasks)):
        pass

# with mp.Pool() as pool:
#     result = pool.map(lazyrun, [(
#         metadata, intrinsics_list, extrinsics_list, local_origin, t,
#         n9468333['v'][np.argmin(np.abs(pd.DatetimeIndex(n9468333.time.values) - pd.to_datetime(t, unit='s')))].values) for t in ts])
# %%
[lazyrun(metadata, intrinsics_list, extrinsics_list, local_origin, t, ds['wl'][ds.timestamp == t].values) for t in ts]

# [lazyrun(metadata, intrinsics_list, extrinsics_list, local_origin, t, n9468333['v'][np.argmin(np.abs(pd.DatetimeIndex(n9468333.time.values) - pd.to_datetime(t, unit='s')))].values) for t in ts[0:2]]
# %%
plt.plot(pd.to_datetime([x.split('/')[-1].split('.')[0] for x in glob.glob('/Volumes/Backstaff/field/unk/proc/rect/' + product + '/*png')], unit='s'),
         marker='*')
