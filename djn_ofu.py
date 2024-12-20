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
import skimage.color
from skimage.exposure import match_histograms
from coastcam_funcs import json2dict
from calibration_crs import *
from rectifier_crs import *
import pandas as pd
from joblib import Parallel, delayed
import xarray as xr
from tqdm import tqdm
import sys
sys.path.append('/Users/dnowacki/Documents/python')
import noaa
# n9468333 = xr.load_dataset('/Users/dnowacki/OneDrive - DOI/Alaska/unk/noaa/n9468333.nc')
import multiprocess as mp


# %%

# site = 'ofu1'
# prefix = 'Record'

site = 'ofu2'
prefix = 'AmSamCam2'

fildir = f'/Volumes/Argus/{site}/'
ym = '2020-03'

camera = ''
product = ''


# https://tidesandcurrents.noaa.gov/stationhome.html?id=1770000
n1770000 = noaa.get_long_coops_data(1770000, '20200228', '20200402', product='predictions')
# elevation of lowest GCP in the registration image is 0.274 m ASVD
# tidal predictions at that time of the image capture 2020-02-09 00:15 UTC is -0.07 m MLLW
# assume that water level was 7 cm below the lowest GCP (this is a massive guess)
# so water level at that time was 0.20 m ASVD
# so add 0.27 m to MLLW predictions to get ASVD
n1770000['v'].values = n1770000.v + 0.27
n1770000.attrs['datum'] = 'estimated ASVD'
# now shift time since Tau is 25 minutes ahead of Pago Pago for HT/LT
# high tide of 12:31 at Pago Pago would be 12:06 at Tau
n1770000['time'] = n1770000['time'].values - pd.Timedelta('25min')
# %%
if site == 'ofu1':
    extrinsic_cal_files = ['/Users/dnowacki/projects/ofu/ofu1/ofu1_extrinsic.json']
    intrinsic_cal_files = ['/Users/dnowacki/projects/ofu/ofu1/ofu1_intrinsic.json']
    local_origin = {'x': 643900 ,'y': 8431456, 'angd': 290}
elif site == 'ofu2':
    extrinsic_cal_files = ['/Users/dnowacki/projects/ofu/ofu2/ofu2_extrinsic_infocus.json']
    intrinsic_cal_files = ['/Users/dnowacki/projects/ofu/ofu2/ofu2_intrinsic_infocus.json']
    local_origin = {'x': 643925 ,'y': 8431460, 'angd': 290}



metadata= {'name': site,
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
    # metadata, intrinsics_list, extrinsics_list, local_origin, t, z = inputs
    print(t)
    if np.isnan(z):
        print('*** NaN detected in Z; skipping ***')
        return
    """ coordinate system setup"""
    if site == 'ofu1':
        xmin = 10
        xmax = 149.9
        ymin = -60
        ymax = 84.9
        dx = .1
        dy = .1
    elif site == 'ofu2':
        xmin = 0
        xmax = 199.9
        ymin = -100
        ymax = 99.9
        dx = .1
        dy = .1

    # z = 0 # z is now defined by water level input
    print(f'***Z: {z} m NAVD88')
    rectifier_grid = TargetGrid(
        [xmin, xmax],
        [ymin, ymax],
        dx,
        dy,
        z
    )

    rectifier = Rectifier(rectifier_grid, ncolors=1)

    image_file = fildir + 'png0/' + ym + '/' + t + '.png'
    intrinsics_list = intrinsics_list
    extrinsics_list = extrinsics_list

    # inimg = [skimage.io.imread(image_file)]

    print(f"{image_file=}")
    rectified_image = rectifier.rectify_images(metadata, [image_file], intrinsics_list, extrinsics_list, local_origin,)

    # fdate = t.replace('(', '').replace(')','').split(' ')[1].replace("_", '-')
    # ftime = t.replace('(', '').replace(')','').split(' ')[2].replace("_", ":")
    fdate = t.split(" ")[0].split("_")[1]
    ftime = t.split(" ")[0].split("_")[2].replace("-", ":")
    ft = int((pd.Timestamp(fdate + ' ' + ftime + ' UTC') + pd.Timedelta('11h')).timestamp()) # need to do this because the utc time is decimated in the filename, so start with local and then add
    print(ft)
    # ofile = fildir + 'proc/rect/' + '/' + str(ft) + '.png'
    ofile = fildir + 'proc/rect/' + ym + '/' + t + '.png'
    print(ofile)
    # squeeze it if we are using greyscale, no change if not
    imageio.imwrite(ofile, np.squeeze(np.flip(rectified_image, 0)), format='png', optimize=True)

    # rectified_image_matched = rectifier.rectify_images(metadata, [c1ref, c2matched], intrinsics_list, extrinsics_list, local_origin)
    # ofile = fildir + 'proc/rect/' + t + '.' + camera + '.' + product + '.rect.matched.png'
    # imageio.imwrite(ofile, np.flip(rectified_image_matched, 0), format='png', optimize=True)



ts1 = [os.path.basename(x).split('.')[0] for x in glob.glob(fildir + 'png0/'+ ym + f'/{prefix}_' + ym + '-??_??-*-??*'+ product + '.png')]

ts = ts1
print('***')
print(ts)
print('***')

with open(fildir + 'proc/rect/' + ym + '/' + product + '/done.txt', 'w') as f:
    for g in glob.glob(fildir + 'proc/rect/' + ym + '/' + product + '/*png'):
        f.write(f"{g.split('/')[-1].split('.')[0]}\n")
    for g in glob.glob(fildir + 'proc/rect/' + ym + '/' + product + '/dark/*png'):
        f.write(f"{g.split('/')[-1].split('.')[0]}\n")

with open(fildir + 'proc/rect/' + ym + '/' + product + '/done.txt') as f:
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

fdate = [t.split(" ")[0].split("_")[1] for t in ts]
ftime = [t.split(" ")[0].split("_")[2].replace("-", ":") for t in ts]
tsepoch = [int((pd.Timestamp(fd + ' ' + ft + ' UTC') + pd.Timedelta('11h')).timestamp()) for fd, ft in zip(fdate, ftime)]

print('***', len(ts))
print(product, camera)
# %%
print(product, camera)
# t = ts[0]
# n9468333['v'][np.argmin(np.abs(pd.DatetimeIndex(n9468333.time.values) - pd.to_datetime(t, unit='s')))].values

# ds = xr.Dataset()
# ds['time'] = xr.DataArray(pd.to_datetime(ts, unit='s'), dims='time')
# ds['timestamp'] = xr.DataArray(ts, dims='time')
# ds['wl'] = n9468333['water_level'].reindex_like(ds['time'], method='nearest', tolerance='10min')
# ds = ds.sortby('time')

# Parallel(n_jobs=-4)(
#     delayed(lazyrun)(
#         metadata, intrinsics_list, extrinsics_list, local_origin, t,
#         ds['wl'][ds.timestamp == t].values)
#         for t in ts
#         )

# wl = np.full_like(tsepoch, np.nan, dtype=float)
# for n in range(len(tsepoch)):
#     try:
#         wl[n] = n1770000.v.sel(time=pd.to_datetime(tsepoch[n], unit='s'), method='nearest', tolerance='15min')
#     except KeyError:
#         print("No wl data for", pd.to_datetime(tsepoch[n], unit='s'))

run_mp = False
if run_mp:
    def multi_run_wrapper(args):
       return lazyrun(*args)

    # from mpire import WorkerPool

    tasks = [
             (metadata,
              intrinsics_list,
              extrinsics_list,
              local_origin,
              ts[n],
              n1770000.v.sel(time=pd.to_datetime(tsepoch[n], unit='s'), method='nearest', tolerance='15min').values)

              for n in range(len(ts))
              ]
    with mp.Pool(int(mp.cpu_count()/4)) as pool:
        for _ in tqdm(pool.imap_unordered(multi_run_wrapper, tasks), total=len(tasks)):
            pass

    # with WorkerPool(n_jobs=12) as pool:
    #     results = pool.map(multi_run_wrapper, tasks, progress_bar=True)


    # with mp.Pool() as pool:
    #     result = pool.map(lazyrun, [(
    #         metadata, intrinsics_list, extrinsics_list, local_origin, t,
    #         n9468333['v'][np.argmin(np.abs(pd.DatetimeIndex(n9468333.time.values) - pd.to_datetime(t, unit='s')))].values) for t in ts])
else:

    [lazyrun(metadata,
             intrinsics_list,
             extrinsics_list,
             local_origin,
             ts[n],
             n1770000.v.sel(time=pd.to_datetime(tsepoch[n], unit='s'), method='nearest', tolerance='15min').values)

             for n in range(len(ts))
    ]

# [lazyrun(metadata, intrinsics_list, extrinsics_list, local_origin, t, n9468333['v'][np.argmin(np.abs(pd.DatetimeIndex(n9468333.time.values) - pd.to_datetime(t, unit='s')))].values) for t in ts[0:2]]
# %%
# plt.plot(pd.to_datetime([x.split('/')[-1].split('.')[0] for x in glob.glob('/Volumes/Backstaff/field/unk/proc/rect/' + product + '/*png')], unit='s'),
#          marker='*')
