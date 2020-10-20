%load_ext autoreload
%autoreload 2
from pathlib import Path
import imageio
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

from coastcam_funcs import json2dict
from calibration_crs import *
from rectifier_crs import *

from joblib import Parallel, delayed

# %%
camera = 'both'

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
ymax = 25
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
    fildir = '/Volumes/Backstaff/field/bti/untitled folder/'
    if 'both' in t:
        image_files = [fildir + 'c1/' + t[:-5] + '.c1.snap.jpg', fildir + 'c2/' + t[:-5] + '.c2.snap.jpg']
    elif 'c1' in t:
        image_files = [fildir + 'c1/' + t + '.snap.jpg']
    elif 'c2' in t:
        image_files = [fildir + 'c2/' + t + '.snap.jpg']
    rectified_image = rectifier.rectify_images(metadata, image_files, intrinsics_list, extrinsics_list, local_origin)
    ofile = t + '_rect.png'
    imageio.imwrite(ofile,np.flip(rectified_image,0),format='png', optimize=True)

ts = ['1531164600',
      '1531166400',
      '1531168200',
      '1531170000',
      '1531171800',
      '1531173600',
      '1531175400',
      '1531177200',
      '1531179000',
      '1531180800',
      '1531182600',
      '1531184400',
      '1531186200',
      '1531188000',
      '1531189800',
      '1531191600',
      '1531193400',
      '1531195200',
      '1531197000',
      '1531198800',
      '1531200600',]

# %%
print(camera)
Parallel(n_jobs=4, backend='multiprocessing')(delayed(lazyrun)(metadata, intrinsics_list, extrinsics_list, local_origin, t + '.' + camera) for t in ts)
