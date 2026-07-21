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
import yaml

site = "guam"
# n9468333 = xr.open_mfdataset("/Users/dnowacki/OneDrive - DOI/Alaska/unk/noaa/n9468756_*.nc")

USE_GNSS = False
gnss = xr.load_dataset("/Users/dnowacki/OneDrive - DOI/Alaska/gnssr/tla0_model2.nc")
USE_SPLINE = False
gnss = xr.load_dataset(
    "/Users/dnowacki/OneDrive - DOI/Alaska/gnssr/tla0_model2_spline.nc"
)
CONSTANT_WL = True
# %%
camera = "c1"
RAWIMAGES = False
if RAWIMAGES:
    fildir = "/Volumes/Argus/guam/RawImages/jpg1741548541513/"
    product = ""
else:
    fildir = "/Volumes/Argus/guam/"
    product = "snap"
fildir = "d:" + fildir

extrinsic_cal_files = [
    "/Users/dnowacki/OneDrive - DOI/Guam/configuration_files_for_runup/HagatnaGU_c1_20220207_EO_djn.yaml",
    "/Users/dnowacki/OneDrive - DOI/Guam/configuration_files_for_runup/HagatnaGU_c2_20220207_EO_djn.yaml",
]
intrinsic_cal_files = [
    "/Users/dnowacki/OneDrive - DOI/Guam/configuration_files_for_runup/HagatnaGU_c1_20220201_IO.yaml",
    "/Users/dnowacki/OneDrive - DOI/Guam/configuration_files_for_runup/HagatnaGU_c2_20220201_IO.yaml",
]

local_origin = {"x": 254148.334, "y": 1491435.828, "angd": 105}
# local_origin = {'x': 0, 'y':0, 'angd': 293}

metadata = {
    "name": "Guam",
    "serial_number": 1,
    "camera_number": camera,
    "calibration_date": "xxxx-xx-xx",
    "coordinate_system": "geo",
}

# read cal files and make lists of cal dicts
extrinsics_list = []
for f in extrinsic_cal_files:
    if camera in f or camera == "cx":
        with open(f) as ff:
            extrinsics_list.append(yaml.safe_load(ff))
intrinsics_list = []
for f in intrinsic_cal_files:
    if camera in f or camera == "cx":
        with open (f) as ff:
            intrinsics_list.append(yaml.safe_load(ff))
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

    xmin = -250
    xmax = -75 - dx
    ymin = 50
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
            c1ref = skimage.io.imread(image_files[0])
        except (AttributeError, ValueError, OSError) as e:
            print("could not process", image_files[0], "; RETURNING", e)
            return
        except FileNotFoundError:
            print("could not process", image_files[0], "; replacing with zeros")
            # c1ref = np.zeros((1536, 2048, 3), dtype=np.uint8)
            c1bad = True

        try:
            c2src = skimage.io.imread(image_files[1])
        except (AttributeError, ValueError, OSError):
            print("could not process", image_files[1], "; RETURNING")
            return
        except FileNotFoundError:
            print("could not process", image_files[1], "; replacing with zeros")
            # c2src = np.zeros((1536, 2048, 3), dtype=np.uint8)
            c2bad = True

        # c2matched = match_histograms(c2src, c1ref, multichannel=True)
    else:
        image_files = [fildir
                + "products/"
                + t
                + pd.Timestamp(int(t), unit="s").strftime(".%a.%b.%d_%H_%M_%S.GMT.%Y.")
                + site
                + "."
                + camera
                + "."
                + product
                + ".jpg",]
        c1bad = False
        c2bad = False

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
            ofile = (
                fildir
                + "rect/gnssr/"
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
        elif USE_GNSS and USE_SPLINE:
            ofile = (
                fildir
                + "rect/gnssr_spline/"
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
        elif CONSTANT_WL:
            ofile = (
                fildir
                + "rect/constantwl/"
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
        else:
            ofile = (
                fildir
                + "rect/"
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
        for x in glob.glob(fildir + "products/1*c1." + product + ".jpg")
    ]
    ts2 = [
        os.path.basename(x).split(".")[0]
        for x in glob.glob(fildir + "products/1*c2." + product + ".jpg")
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
