# %load_ext autoreload
# %autoreload 2
import imageio
import numpy as np
import glob
import os
import skimage.io
from coastcam_funcs import json2dict
from calibration_crs import CameraCalibration
from rectifier_crs import Rectifier, TargetGrid
import pandas as pd
from joblib import Parallel, delayed
import xarray as xr

# water level for 2021 data
# rbr = xr.load_dataset('/Users/dnowacki/Library/CloudStorage/OneDrive-DOI/Alaska/brw2021/rbr/041230_20211007_2113/wave_interval_512/brwrbrs-a.nc')
# waterlevel during wave gauge deployment 1.25 m -- see brw_trimble.py
# rbr['wl'] = rbr['water_depth'] + (1.25 - rbr['water_depth'][0])

# n9497645 = xr.load_dataset('/Users/dnowacki/OneDrive - DOI/Alaska/noaa/gpsn9497645.nc')
# https://dggs.alaska.gov/hazards/coastal/ak-tidal-datum-portal.html
# https://dggs.alaska.gov/hazards/coastal/download/202312_reference_table.pdf
# n9497645['wl'] = n9497645['v'] - np.mean([-0.971, -0.925]) # this is the mean of two OPUS/datum analysis values


n9468333 = xr.load_dataset('/Users/dnowacki/OneDrive - DOI/Alaska/unk/noaa/n9468333.nc')
n9468333['wl'] = n9468333['water_level']

USE_GNSS = True
gnss = xr.load_dataset('/Users/dnowacki/OneDrive - DOI/Alaska/gnssr/glv0_model2.nc')
USE_SPLINE = True
gnss = xr.load_dataset('/Users/dnowacki/OneDrive - DOI/Alaska/gnssr/glv0_model2_spline.nc')

# %%
fildir = '/Volumes/Argus/glv/'
fildir = 'd:' + fildir
camera = 'cx'
product = 'snap'

extrinsic_cal_files = ['/Users/dnowacki/projects/ak/py/glv_extrinsic_c1.json',
                       '/Users/dnowacki/projects/ak/py/glv_extrinsic_c2.json',]
intrinsic_cal_files = ['/Users/dnowacki/projects/ak/py/glv_intrinsic_c1.json',
                       '/Users/dnowacki/projects/ak/py/glv_intrinsic_c2.json',]

local_origin = {'x': 594224, 'y':7158851, 'angd': 293}
# local_origin = {'x': 0, 'y':0, 'angd': 293}

metadata= {'name': 'Golovin',
           'serial_number': 1,
           'camera_number': camera,
           'calibration_date': 'xxxx-xx-xx',
           'coordinate_system': 'geo'}

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
    print(t)

    if np.isnan(z):
        print("No wl data for", pd.to_datetime(int(t), unit='s'))
        return
    """ coordinate system setup"""
    dx = 0.1
    dy = 0.1

    xmin = 0
    xmax = 300-dx
    ymin = 0
    ymax = 250-dy
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
        image_files = [fildir + 'products/' + t + pd.Timestamp(int(t), unit='s').strftime('.%a.%b.%d_%H_%M_%S.GMT.%Y.golovin') + '.c1.' + product + '.jpg',
                       fildir + 'products/' + t + pd.Timestamp(int(t), unit='s').strftime('.%a.%b.%d_%H_%M_%S.GMT.%Y.golovin') + '.c2.' + product + '.jpg']
        # print(image_files)
        try:
            c1ref = skimage.io.imread(image_files[0])
        except (AttributeError, ValueError, OSError):
            print('could not process', image_files[0], '; RETURNING')
            return
        except FileNotFoundError:
            print('could not process', image_files[0], '; replacing with zeros')
            # c1ref = np.zeros((1536, 2048, 3), dtype=np.uint8)
            c1bad = True

        try:
            c2src = skimage.io.imread(image_files[1])
        except (AttributeError, ValueError, OSError):
            print('could not process', image_files[1], '; RETURNING')
            return
        except FileNotFoundError:
            print('could not process', image_files[1], '; replacing with zeros')
            # c2src = np.zeros((1536, 2048, 3), dtype=np.uint8)
            c2bad = True

        # c2matched = match_histograms(c2src, c1ref, multichannel=True)
    else:
        image_files = [fildir + 'products/' + t + '.' + camera + '.' + product + '.jpg']

    print(f"{c1bad=}, {c2bad=}")

    if c1bad and c2bad:
        print('both bad, returning')
        return

    if c1bad:
        intrinsics_list = [intrinsics_list[1]]
        extrinsics_list = [extrinsics_list[1]]
        image_files = [image_files[1]]
    elif c2bad:
        intrinsics_list = [intrinsics_list[0]]
        extrinsics_list = [extrinsics_list[0]]
        image_files = [image_files[0]]

    rectified_image = rectifier.rectify_images(metadata, image_files, intrinsics_list, extrinsics_list, local_origin)
    if USE_GNSS and not USE_SPLINE:
        ofile = fildir + 'rect/gnssr/' + product + '/' + t + pd.Timestamp(int(t), unit='s').strftime('.%a.%b.%d_%H_%M_%S.GMT.%Y.golovin') + '.' + camera + '.' + product + '.png'
    elif USE_GNSS and USE_SPLINE:
        ofile = fildir + 'rect/gnssr_spline/' + product + '/' + t + pd.Timestamp(int(t), unit='s').strftime('.%a.%b.%d_%H_%M_%S.GMT.%Y.golovin') + '.' + camera + '.' + product + '.png'
    else:
        ofile = fildir + 'rect/' + product + '/' + t + pd.Timestamp(int(t), unit='s').strftime('.%a.%b.%d_%H_%M_%S.GMT.%Y.golovin') + '.' + camera + '.' + product + '.png'
    print(ofile)
    imageio.imwrite(ofile, np.flip(rectified_image, 0), format='png', optimize=True)

    # rectified_image_matched = rectifier.rectify_images(metadata, [c1ref, c2matched], intrinsics_list, extrinsics_list, local_origin)
    # ofile = fildir + 'proc/rect/' + t + '.' + camera + '.' + product + '.rect.matched.png'
    # imageio.imwrite(ofile, np.flip(rectified_image_matched, 0), format='png', optimize=True)

    return rectifier

ts1 = [os.path.basename(x).split('.')[0] for x in glob.glob(fildir + 'products/1*c1.'+ product + '.jpg')]
ts2 = [os.path.basename(x).split('.')[0] for x in glob.glob(fildir + 'products/1*c2.' + product + '.jpg')]

if camera == 'c1':
    ts = ts1
elif camera == 'c2':
    ts = ts2
elif camera == 'cx':
    ts = list(set(ts1) | set(ts2)) # use OR instead of AND so we can have rectified images when just one camera was active

if USE_GNSS and not USE_SPLINE:
    shim = 'gnssr/'
elif USE_GNSS and USE_SPLINE:
    shim = 'gnssr_spline/'
else:
    shim = ''
with open(fildir + 'rect/' + shim + product + '/done.txt', 'w') as f:
    for g in glob.glob(fildir + 'rect/' + shim + product + '/*png'):
        f.write(f"{g.split('/')[-1].split("\\")[-1].split('.')[0]}\n")
    for g in glob.glob(fildir + 'rect/' + shim + product + '/dark/*png'):
        f.write(f"{g.split('/')[-1].split("\\")[-1].split('.')[0]}\n")

with open(fildir + 'rect/' + shim + product + '/done.txt') as f:
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

ds = xr.Dataset()
ds['time'] = xr.DataArray(pd.to_datetime([int(x) for x in ts], unit='s'), dims='time')
ds['timestamp'] = xr.DataArray(ts, dims='time')
if USE_GNSS:
    ds['wl'] = gnss['wl'].reindex_like(ds['time'], method='nearest', tolerance='60min')
else:
    ds['wl'] = n9468333['water_level'].reindex_like(ds['time'], method='nearest', tolerance='10min')
ds = ds.sortby('time')

# randomize ts
import random
random.shuffle(ts)

        
import multiprocessing as mp
def split_into_chunks(lst, chunk_size):
    result = []
    for i in range(0, len(lst), chunk_size):
        result.append(lst[i:i + chunk_size])
    return result
    
print(len(ts))

if __name__ == '__main__':
    for tsshort in split_into_chunks(ts, 200):
        with mp.Pool(processes=10,maxtasksperchild=10) as pool:
            result = pool.starmap(lazyrun, [(
                metadata, intrinsics_list, extrinsics_list, local_origin, t,
                ds['wl'][ds.timestamp == t].values) for t in tsshort])