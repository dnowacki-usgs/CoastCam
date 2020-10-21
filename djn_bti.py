%load_ext autoreload
%autoreload 2
from pathlib import Path
import imageio
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import glob
import os
from coastcam_funcs import json2dict
from calibration_crs import *
from rectifier_crs import *

from joblib import Parallel, delayed

# %%
camera = 'c1'

extrinsic_cal_files = ['/Users/dnowacki/Projects/ak/py/extrinsic_c1.json',
                       '/Users/dnowacki/Projects/ak/py/extrinsic_c2.json',]
intrinsic_cal_files = ['/Users/dnowacki/Projects/ak/py/intrinsic_c1.json',
                       '/Users/dnowacki/Projects/ak/py/intrinsic_c2.json',]

local_origin = {'x': 0,'y':0, 'angd': 0}

metadata= {'name': 'BTI',
           'serial_number': 1,
           'camera_number': camera,
           'calibration_date': '2019-12-12',
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

""" coordinate system setup"""
xmin = 10
xmax = 200
ymin = -200
ymax = 5 # was 25
dx = 0.1
dy = 0.1
z = 1.5

rectifier_grid = TargetGrid(
    [xmin, xmax],
    [ymin, ymax],
    dx,
    dy,
    z
)

rectifier = Rectifier(rectifier_grid)

def lazyrun(metadata, intrinsics_list, extrinsics_list, local_origin, t):
    print(t)
    fildir = '/Volumes/Backstaff/field/bti/'
    if camera is 'both':
        image_files = [fildir + 'products/' + t + '.c1.snap.jpg',
                       fildir + 'products/' + t + '.c2.snap.jpg']
    else:
        image_files = [fildir + 'products/' + t + '.' + camera + '.snap.jpg']
    rectified_image = rectifier.rectify_images(metadata, image_files, intrinsics_list, extrinsics_list, local_origin)
    ofile = fildir + 'proc/rect/' + t + '.' + camera + '.snap.rect.png'
    imageio.imwrite(ofile,np.flip(rectified_image,0),format='png', optimize=True)


ts =[os.path.basename(x).split('.')[0] for x in glob.glob('/Volumes/Backstaff/field/bti/products/*c1.snap.jpg')]
ts[0:100]

# %%
print(camera)
Parallel(n_jobs=4, backend='multiprocessing')(delayed(lazyrun)(metadata, intrinsics_list, extrinsics_list, local_origin, t) for t in ts)
