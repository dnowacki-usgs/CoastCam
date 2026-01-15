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
import xarray as xr
import multiprocessing as mp
import random

site = "golovin"
n9468333 = xr.open_mfdataset("/Users/dnowacki/OneDrive - DOI/Alaska/unk/noaa/n9468333_*.nc")

USE_GNSS = False
gnss = xr.load_dataset("/Users/dnowacki/OneDrive - DOI/Alaska/gnssr/glv0_model2.nc")
USE_SPLINE = False
gnss = xr.load_dataset(
    "/Users/dnowacki/OneDrive - DOI/Alaska/gnssr/glv0_model2_spline.nc"
)
USE_AWLW = True
awlw = xr.load_dataset(
    f"/Users/dnowacki/OneDrive - DOI/Alaska/noaa/awlw/{site}-gnssr.nc"
)
CONSTANT_WL = False
# %%
camera = "cx"
RAWIMAGES = False
if RAWIMAGES:
    fildir = "/Volumes/Argus/glv/RawImages/jpg1741548541513/"
    product = ""
else:
    fildir = "/Volumes/Argus/glv/"
    product = "snap"
fildir = "d:" + fildir

extrinsic_cal_files = [
    "/Users/dnowacki/projects/ak/py/glv_extrinsic_c1.json",
    "/Users/dnowacki/projects/ak/py/glv_extrinsic_c2.json",
]
intrinsic_cal_files = [
    "/Users/dnowacki/projects/ak/py/glv_intrinsic_c1.json",
    "/Users/dnowacki/projects/ak/py/glv_intrinsic_c2.json",
]

local_origin = {"x": 594224, "y": 7158851, "angd": 293}
# local_origin = {'x': 0, 'y':0, 'angd': 293}

metadata = {
    "name": "Golovin",
    "serial_number": 1,
    "camera_number": camera,
    "calibration_date": "xxxx-xx-xx",
    "coordinate_system": "geo",
}

# read cal files and make lists of cal dicts
extrinsics_list = []
for f in extrinsic_cal_files:
    if camera in f or camera == "cx":
        extrinsics_list.append(json2dict(f))
intrinsics_list = []
for f in intrinsic_cal_files:
    if camera in f or camera == "cx":
        intrinsics_list.append(json2dict(f))
print(extrinsics_list)
print(intrinsics_list)

# check test for coordinate system
if metadata["coordinate_system"].lower() == "xyz":
    print("Extrinsics are local coordinates")
elif metadata["coordinate_system"].lower() == "geo":
    print("Extrinsics are in world coordinates")
else:
    print("Invalid value of coordinate_system: ", metadata["coordinate_system"])

print(extrinsics_list[0])
print(extrinsics_list[0]["y"] - local_origin["y"])

calibration = CameraCalibration(
    metadata, intrinsics_list[0], extrinsics_list[0], local_origin
)
print(calibration.local_origin)
print(calibration.world_extrinsics)
print(calibration.local_extrinsics)


def lazyrun(metadata, intrinsics_list, extrinsics_list, local_origin, t, z):
    print(t)

    if np.all(np.isnan(z)):
        print("No wl data for", pd.to_datetime(int(t), unit="s"))
        return
    """ coordinate system setup"""
    dx = 0.1
    dy = 0.1

    xmin = 0
    xmax = 300 - dx
    ymin = 0
    ymax = 250 - dy

    # z = 0 # z is now defined by water level input
    print(f"***Z: {z} m NAVD88")
    rectifier_grid = TargetGrid([xmin, xmax], [ymin, ymax], dx, dy, z)

    rectifier = Rectifier(rectifier_grid)

    if camera == "cx":
        c1bad = False
        c2bad = False
        if RAWIMAGES:
            image_files = [fildir + "c1_" + t + ".jpg", fildir + "c2_" + t + ".jpg"]
            # sometimes we are off by +/- 1 ms between the camera captures
            if not os.path.isfile(image_files[1]):
                image_files[1] = fildir + "c2_" + str(int(t) - 1) + ".jpg"
            if not os.path.isfile(image_files[1]):
                image_files[1] = fildir + "c2_" + str(int(t) + 1) + ".jpg"
        else:
            image_files = [
                fildir
                + "products/"
                + t
                + pd.Timestamp(int(t), unit="s").strftime(".%a.%b.%d_%H_%M_%S.GMT.%Y.")
                + site
                + ".c1."
                + product
                + ".jpg",
                fildir
                + "products/"
                + t
                + pd.Timestamp(int(t), unit="s").strftime(".%a.%b.%d_%H_%M_%S.GMT.%Y.")
                + site
                + ".c2."
                + product
                + ".jpg",
            ]
        # print(image_files)
        try:
            skimage.io.imread(image_files[0])
        except (AttributeError, ValueError, OSError) as e:
            print("could not process", image_files[0], "; RETURNING", e)
            return
        except FileNotFoundError:
            print("could not process", image_files[0], "; replacing with zeros")
            # c1ref = np.zeros((1536, 2048, 3), dtype=np.uint8)
            c1bad = True

        try:
            skimage.io.imread(image_files[1])
        except (AttributeError, ValueError, OSError):
            print('could not process', image_files[1], '; RETURNING')
            return
        except FileNotFoundError:
            print("could not process", image_files[1], "; replacing with zeros")
            # c2src = np.zeros((1536, 2048, 3), dtype=np.uint8)
            c2bad = True

        # c2matched = match_histograms(c2src, c1ref, multichannel=True)
    else:
        image_files = [fildir + "products/" + t + "." + camera + "." + product + ".jpg"]

    print(f"{c1bad=}, {c2bad=}")

    if c1bad and c2bad:
        print("both bad, returning")
        return

    if c1bad:
        intrinsics_list = [intrinsics_list[1]]
        extrinsics_list = [extrinsics_list[1]]
        image_files = [image_files[1]]
    elif c2bad:
        intrinsics_list = [intrinsics_list[0]]
        extrinsics_list = [extrinsics_list[0]]
        image_files = [image_files[0]]

    rectified_image = rectifier.rectify_images(
        metadata, image_files, intrinsics_list, extrinsics_list, local_origin
    )

    if RAWIMAGES:
        if USE_GNSS and not USE_SPLINE:
            ofile = fildir + "rect/gnssr/" + product + "/" + t + "." + camera + ".png"
        elif USE_GNSS and USE_SPLINE:
            ofile = (
                fildir
                + "rect/gnssr_spline/"
                + product
                + "/"
                + t
                + "."
                + camera
                + ".png"
            )
        else:
            ofile = fildir + "rect/" + product + "/" + t + "." + camera + ".png"
    else:
        if USE_GNSS and not USE_SPLINE:
            folder = "rect/gnssr/"
        elif USE_GNSS and USE_SPLINE:
            folder = "rect/gnssr_spline/"
        elif USE_AWLW:
            folder = "rect/awlw/"
        else:
            folder = "rect/"
        ofile = (
            fildir
            + folder
            + product
            + "/"
            + t
            + pd.Timestamp(int(t), unit="s").strftime(".%a.%b.%d_%H_%M_%S.GMT.%Y.")
            + site
            + "."
            + camera
            + "."
            + product
            + ".png"
            )
    print(ofile)
    imageio.imwrite(ofile, np.flip(rectified_image, 0), format="png", optimize=True)

    # rectified_image_matched = rectifier.rectify_images(metadata, [c1ref, c2matched], intrinsics_list, extrinsics_list, local_origin)
    # ofile = fildir + 'proc/rect/' + t + '.' + camera + '.' + product + '.rect.matched.png'
    # imageio.imwrite(ofile, np.flip(rectified_image_matched, 0), format='png', optimize=True)

    return rectifier


if RAWIMAGES:
    ts1 = [
        os.path.basename(x).split("_")[1].split(".")[0]
        for x in glob.glob(fildir + "c1_*.jpg")
    ]
    ts2 = [
        os.path.basename(x).split("_")[1].split(".")[0]
        for x in glob.glob(fildir + "c2_*.jpg")
    ]
else:
    ts1 = [
        os.path.basename(x).split(".")[0]
        for x in glob.glob(fildir + "products/1*202[5,6]*c1." + product + ".jpg")
    ]
    ts2 = [
        os.path.basename(x).split(".")[0]
        for x in glob.glob(fildir + "products/1*202[5,6]*c2." + product + ".jpg")
    ]

if camera == "c1":
    ts = ts1
elif camera == "c2":
    ts = ts2
elif camera == "cx":
    ts = list(
        set(ts1) | set(ts2)
    )  # use OR instead of AND so we can have rectified images when just one camera was active

if RAWIMAGES:
    # need to ensure we don't double count since sometimes the timestamps aren't exactly equivalent.
    # We deal with these changes in the function.
    ts = ts1

if USE_GNSS and not USE_SPLINE:
    shim = "gnssr/"
elif USE_GNSS and USE_SPLINE:
    shim = "gnssr_spline/"
elif USE_AWLW:
    shim = "awlw/"   
elif CONSTANT_WL:
    shim = "constantwl/"
else:
    shim = ""

os.makedirs(fildir + "rect/" + shim + product, exist_ok=True)

with open(fildir + "rect/" + shim + product + "/done.txt", "w") as f:
    for g in glob.glob(fildir + "rect/" + shim + product + "/*png"):
        f.write(f"{g.split('/')[-1].split("\\")[-1].split('.')[0]}\n")
    for g in glob.glob(fildir + "rect/" + shim + product + "/dark/*png"):
        f.write(f"{g.split('/')[-1].split("\\")[-1].split('.')[0]}\n")

with open(fildir + "rect/" + shim + product + "/done.txt") as f:
    tsdone = [line.rstrip() for line in f]

# this will get what remains to be done
print("length of ts", len(ts))
print("length of the done list", len(tsdone))
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
print("values in t not already in the done list", len(tsnew))
ts = tsnew

print("***", len(ts))
print(product, camera)

ds = xr.Dataset()
if RAWIMAGES:
    ds["time"] = xr.DataArray(
        pd.to_datetime([int(x) for x in ts], unit="ms"), dims="time"
    )
else:
    ds["time"] = xr.DataArray(
        pd.to_datetime([int(x) for x in ts], unit="s"), dims="time"
    )
ds["timestamp"] = xr.DataArray(ts, dims="time")
if USE_GNSS:
    ds["wl"] = gnss["wl"].reindex_like(ds["time"], method="nearest", tolerance="60min")
elif USE_AWLW:
    ds["wl"] = awlw["water_surface_above_navd88"].reindex_like(ds["time"], method="nearest", tolerance="60min")
elif CONSTANT_WL:
    ds["wl"] = 1.37 * xr.ones_like(ds.time).astype(
        float
    )  # using mean of GNSS-R values (model2)
else:
    ds["wl"] = n9468333["water_level"].reindex_like(
        ds["time"], method="nearest", tolerance="10min"
    )
ds = ds.sortby("time")

import multiprocessing as mp


def split_into_chunks(lst, chunk_size):
    result = []
    for i in range(0, len(lst), chunk_size):
        result.append(lst[i : i + chunk_size])
    return result


print(len(ts))

# so we don't waste time checking for water levels we know don't exist
if USE_AWLW:
    for t in ts:
        if pd.Timestamp(int(t), unit="s") < awlw.time.min():
            ts.remove(t)
        if pd.Timestamp(int(t), unit="s") > awlw.time.max():
            ts.remove(t)

if __name__ == "__main__":
    for tsshort in split_into_chunks(ts, 200):
        with mp.Pool(processes=10, maxtasksperchild=10) as pool:
            result = pool.starmap(
                lazyrun,
                [
                    (
                        metadata,
                        intrinsics_list,
                        extrinsics_list,
                        local_origin,
                        t,
                        ds["wl"][ds.timestamp == t].values,
                    )
                    for t in tsshort
                ],
            )
