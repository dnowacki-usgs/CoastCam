import numpy as np
import datetime
import os
from dateutil import tz
from skimage import io
from PIL import Image
import json
import yaml
import matplotlib.pyplot as plt

def estimate_sharpness(filepath,fs=None):
    """
    Estimate image sharpness
    https://stackoverflow.com/questions/6646371/detect-which-image-is-sharper
    """
    if fs:
        # using fsspec for S3 files
        with fs.open(filepath,'rb') as f:
            im = Image.open(f).convert('L') # to grayscale
    else:
        # regular file system
        with open(filepath, 'rb') as f:
            im = Image.open(f).convert('L') # to grayscale

    array = np.asarray(im, dtype=np.int32)
    contrast = array.std()
    gy, gx = np.gradient(array)
    gnorm = np.sqrt(gx**2 + gy**2)
    sharpness = np.average(gnorm)
    return sharpness, contrast

def average_color(filepath,fs=None):
    """
    Calculate the average pixel intensity of an image
    Input:
        filepath - file path for an image
    Returned:
        av, avall - av (np.array of average r, g, b values), avall average of r,g,b
    """
    if fs:
        with fs.open(filepath,'rb') as f:
            img = io.imread(f)
    else:
        with open(filepath, 'rb') as f:
            img = io.imread(f)

    av = img.mean(axis=0).mean(axis=0)
    avall = av.mean(axis=0)
    return av, avall

def detect_blur_fft(filepath, size=60, vis=False, fs=None):
    """
    Use high-frequency content of image fft to determine blur
    From: https://www.pyimagesearch.com/2020/06/15/opencv-fast-fourier-transform-fft-for-blur-detection-in-images-and-video-streams/
    """
    if fs:
        with fs.open(filepath,'rb') as f:
            img = io.imread(f, as_gray=True)
    else:
        with open(filepath, 'rb') as f:
            img = io.imread(f, as_gray=True)

    # grab the dimensions of the image and use the dimensions to
    # derive the center (x, y)-coordinates
    (h, w) = img.shape
    (cX, cY) = (int(w / 2.0), int(h / 2.0))
    fft = np.fft.fft2(img)
    fftShift = np.fft.fftshift(fft)
    # zero-out the center of the FFT shift (i.e., remove low
    # frequencies), apply the inverse shift such that the DC
    # component once again becomes the top-left, and then apply
    # the inverse FFT
    fftShift[cY - size:cY + size, cX - size:cX + size] = 0
    # check to see if we are visualizing our output
    if vis:
        # compute the magnitude spectrum of the transform
        np.seterr(all='ignore') # suppress log of zero error
        magnitude = 20 * np.log(np.abs(fftShift))
        # display the original input image
        (fig, ax) = plt.subplots(1, 2, )
        ax[0].imshow(img, cmap="gray")
        ax[0].set_title("Input")
        ax[0].set_xticks([])
        ax[0].set_yticks([])
        # display the magnitude image
        ax[1].imshow(magnitude, cmap="gray")
        ax[1].set_title("Magnitude Spectrum")
        ax[1].set_xticks([])
        ax[1].set_yticks([])
        # show our plots
        plt.show()

    fftShift = np.fft.ifftshift(fftShift)
    recon = np.fft.ifft2(fftShift)
    # compute the magnitude spectrum of the reconstructed image,
    # then compute the mean of the magnitude values
    magnitude = 20 * np.log(np.abs(recon))
    mean = np.mean(magnitude)
    # the image will be considered "blurry" if the mean value of the
    # magnitudes is less than the threshold value
    return mean

def json2dict(jsonfile):
    """ Import contents of a JSON file as a dict
    Args:
        jsonfile (str): json2dict file to read
    Returns:
        dict interpreted from JSON file
    """
    with open(jsonfile, "r") as data:
        dictname = json.loads(data.read())
    return dictname

def yaml2dict(yamlfile):
    """ Import contents of a YAML file as a dict
    Args:
        yamlfile (str): YAML file to read
    Returns:
        dict interpreted from YAML file
    """
    dictname = None
    with open(yamlfile, "r") as infile:
        try:
            dictname = yaml.safe_load(infile)
        except yaml.YAMLerror as exc:
            print(exc)
    return dictname

def dts2unix(date_time_string, timezone='eastern'):
    """
    Return the unix epoch time and a datetime object with aware UTC time zone
    Input:
        date_time_string in format YY-MM-DD hh:mm
        time zone for date_time string
    Returns:
        epoch number, datetime_object
    """
    if timezone.lower() == 'eastern':
        tzone = tz.gettz('America/New_York')
    elif timezone.lower() == 'utc':
        tzone = tz.gettz('UTC')

    date_time_obj = datetime.datetime.strptime(date_time_string, '%Y-%m-%d %H:%M').replace(tzinfo=tzone)
    ts = date_time_obj.timestamp()
    return int(ts), date_time_obj

def unix2dts(unixnumber, timezone='eastern'):
    """
    Get local time from unix number
    Input:
        unixnumber - string containing unix time (aka epoch)
    Returns:
        date_time_string, date_time_object in utc
    TODO: not sure why this returns the correct value without specifying that input time zone is eastern
    """
    if timezone.lower() == 'eastern':
        tzone = tz.gettz('America/New_York')
    elif timezone.lower() == 'utc':
        tzone = tz.gettz('UTC')

    # images other than "snaps" end in 1, 2,...but these are not part of the time stamp.
    # replace with zero
    ts = int( unixnumber[:-1]+'0')
    date_time_obj =  datetime.datetime.utcfromtimestamp(ts)
    date_time_str = date_time_obj.strftime('%Y-%m-%d %H:%M:%S')
    return date_time_str, date_time_obj

def filetime2timestr(filepath, timezone='eastern'):
    """
    Return the local time and the Unix Epoch string from an image filename or path
    Does not work with backslashes (e.g., Windows paths)
    """
    if timezone.lower() == 'eastern':
        tzone = tz.gettz('America/New_York')
    elif timezone.lower() == 'utc':
        tzone = tz.gettz('UTC')

    # remove path
    filename = os.path.split(os.path.normpath(filepath))[-1]
    # split on '.', take first on
    s = filename.split('.')[0]

    #TODO - Could check camera type and correct last digit, but it does not affect seconds

    date_time_str, date_time_obj = unix2dts(s)
    return date_time_str, s

def timestr2filename(date_time_str, camera = 'c1', image_type = 'timex', timezone='eastern'):
    """
    Return a filename given a date_time_str and other info
    """
    # filenames have extra digit added to time stamps - here is a dict listing them
    last_number = {'snap': 0, 'timex': 1, 'var': 2, 'bright': 3, 'dark': 4, 'rundark': 5}
    if timezone.lower() == 'eastern':
        tzone = tz.gettz('America/New_York')
    elif timezone.lower() == 'utc':
        tzone = tz.gettz('UTC')

    date_time_obj = datetime.datetime.strptime(date_time_str, '%Y-%m-%d %H:%M').replace(tzinfo=tzone)
    ts = int(date_time_obj.timestamp())+int(last_number[image_type])

    fn = str(ts)+'.'+camera+'.'+image_type+'.jpg'
    return fn

def local_transform_points( xo, yo, ang, flag, xin, yin):
    """
    Transforms between local World Coordinates and Geographical
    World Coordinates. Local refers to the rotated coordinate system where x
    is positive offshore and y is oriented alongshore. This function can go
    from Local to Geographical or in reverse.
    Based on localTranformPoints.m by Brittany Bruder, but in radians
    Input:
        xo and yo - location of local origin (0,0) in Geographical coordinates.
              Typically xo is E and yo is N coordinate.
        ang - relative angle between the local X axis and the Geo X axis,
              positive counter-clockwise from the Geo X.  Units are radians.
              Note: Regardless of transformation direction, xo, yo, and ang
              should stay the same.
        flag = 1 or 0 to indicate transform direction
              Geo-->local (1) or
              local-->Geo (0)
        xin - Local (X) or Geo (E) coord depending on transformation direction
        yin = Local (Y) or Geo (N) coord depending on transformation direction
    Returns:
        xout - Local (X) or Geo (E) coord depending on transformation direction
        yout - Local (Y) or Geo (N) coord depending on transformation direction
    """

    if flag == 1:
        # transform from world -> local
        # translate from origin
        easp = xin-xo
        norp = yin-yo

        #rotate
        xout = easp*np.cos(ang)+norp*np.sin(ang)
        yout = norp*np.cos(ang)-easp*np.sin(ang)

    if flag == 0:
        # rotate
        xout = xin*np.cos(ang)-yin*np.sin(ang)
        yout = yin*np.cos(ang)+xin*np.sin(ang)
        # translate
        xout = xout+xo
        yout = yout+yo

    return xout, yout

def local_transform_extrinsics(local_xo,local_yo,local_angd,flag,extrinsics_in):
    """
    Tranforms between Local World Coordinates and Geographical
    World Coordinates for the extrinsics vector. Local refers to the rotated
    coordinate system where X is positive offshore and y is oriented
    alongshore. The function can go from Local to Geographical and in
    reverse. Note, this only performs horizontal rotations/transformations.
    Based on localTranformExtrinsics.m by Brittany Bruder
    Input:
    local_xo and local_yo - local origin = Location of Local (0,0) in
        Geographical Coordinates. Typically first entry is E and second is N coordinate.
    local_angd -The relative angle between the new (local) X axis and old (Geo)
        X axis, positive counter-clockwise from the old (Geo) X.  Units are degrees.
    Note: Regardless of transformation direction, local_ang, local_xo, and local_yo
        should stay the same. z, tilt, and roll do not change.
    extrinsics_in - dict with Local or Geo x, y, z, a(zimuth), t(ilt), r(oll)
        By CIRN convention, extrinsic angles are in radians.
    flag = 1 or 0 to indicate whether you are going from
        Geo-->Local (1) or
        Local-->Geo (0)
    Output:
    extrinsics_out - dict with Local or Geo x, y, z, a(zimuth), t(ilt), r(oll)
    """
    local_angr = np.deg2rad(local_angd)
    extrinsics_out = extrinsics_in.copy()
    if flag == 1:
        # World to local
        extrinsics_out['x'], extrinsics_out['y'] = local_transform_points(local_xo,local_yo,local_angr,1,extrinsics_in['x'],extrinsics_in['y'])
        extrinsics_out['a'] = extrinsics_in['a']+local_angr

    if flag == 0:
        # local to world
        extrinsics_out['x'], extrinsics_out['y'] = local_transform_points(local_xo,local_yo,local_ang,0,extrinsics_in['x'],extrinsics_in['y'])
        extrinsics_out['a'] = extrinsics_in['a']-local_angr

    return extrinsics_out

def assembleP(extrinsics, intrinsics):
    """Assembles and returns Projective (P) matrix from LCP and Beta values.
    Notes:
        - Derived from lcpBeta2P.m + CiRN notes
        - K converts angle away from the center of view into camera coordinates
        - R describes the 3D viewing direction of camera compared to world coordinates
        - beta[:3] camera location in world coordinates (x,y,z)
        - beta[3::] camera orientation (azimuth, tilt, roll)
    Returns:
        P (np.ndarray): Projective matrix
    Based on Axion object
    """
    # K: intrinsic matrix, puts image in pixel units of the specific camera
    K = np.array([
        [intrinsics['fx'], 0.,                intrinsics['c0U']],
        [0.,              -intrinsics['fy'],  intrinsics['c0V']],
        [0.,               0.,                1.]
    ])
    # R: rotation matrix, puts image in camera orientation
    R = angle2R(
        extrinsics['a'],
        extrinsics['t'],
        extrinsics['r']
    )
    # I: identity matrix augmented by camera center, puts image in camera coordinates
    IC = np.vstack((
        np.eye(3),
        -extrinsics['x'],-extrinsics['y'],-extrinsics['z']
        )).T
    KR = np.matmul(K, R)
    P = np.matmul(KR, IC)

    # Make the matrix homogenous, methods use homogenous coordinates for easier math
    # - normalize to make last element equal 1
    P = P/P[-1, -1]

    return P


def angle2R(azimuth, tilt, swing):
    """Assembles and returns a rotation matrix R from azimuth, tilt, and swing (roll)
    Notes:
        - derived from angles2R.m by Costal Imaging Research Network and Oregon State University
        - From p 612 of Wolf, 1983
    Arguments:
        azimuth (float): Azimuth
        tilt (float): Tilt
        swith (float): swing
    Returns:
        R (np.ndarray): Rotation matrix
    """
    a = azimuth
    t = tilt
    s = swing
    R = np.zeros((3, 3))

    R[0, 0] = np.cos(a) * np.cos(s) + np.sin(a) * np.cos(t) * np.sin(s)
    R[0, 1] = -np.cos(s) * np.sin(a) + np.sin(s) * np.cos(t) * np.cos(a)
    R[0, 2] = np.sin(s) * np.sin(t)

    R[1, 0] = -np.sin(s) * np.cos(a) + np.cos(s) * np.cos(t) * np.sin(a)
    R[1, 1] = np.sin(s) * np.sin(a) + np.cos(s) * np.cos(t) * np.cos(a)
    R[1, 2] = np.cos(s) * np.sin(t)

    R[2, 0] = np.sin(t) * np.sin(a)
    R[2, 1] = np.sin(t) * np.cos(a)
    R[2, 2] = -np.cos(t)

    return R
