import numpy as np
import scipy
import corefunctions
import pandas as pd
import sys
sys.path.append('/Users/dnowacki/Documents/python')
import matlabtools
# %%
# from extrinsicsSolver.m
# %  The following command actually uses two functions: CIRN xyz2DistUV and
# %  nlinfit. xyzToDistUV transforms a set of real world coordinates (XYZ) to
# %  distorted (UV) coordinates given an intrinsic matrix (I), extrinsics (EO), and XYZ points.
# %  Nlinfit solves the inverse of the xyzToDistUV function: it finds the
# %  optimum extrinsics (EO) that minimizes error given a set of UV, XYZ, an intrinsics matrix (IO,
# %  an initial guess for extrinsics (extrinsicsInitialGuess).
#
# %  Ultimately we are telling nlinfit- our UV results are a function of extrinsics
# %  and xyz [@(extrinsics,xyz)], all input left out of those brackets (intrinsics) should
# %  not be solved for and taken as constants in this solution.
# [extrinsics,R,J,CovB]   = nlinfit(xyz,[UV(:,1); UV(:,2)],@(extrinsics,xyz)xyz2DistUV(intrinsics,extrinsics,xyz), extrinsicsInitialGuess);
# beta = nlinfit(X,Y,modelfun,beta0)

# xyz = real world coordinates
# UV = distorted UV coordinates


# %%
dfxyz = pd.read_csv('/Users/dnowacki/OneDrive - DOI/Alaska/norton2024/golovin/argus/extrinsic/glvc1_beach_gcps.csv',
            header=None, names=['gcp', 'x', 'y', 'z'])

# %%

intrin = matlabtools.loadmat('/Users/dnowacki/OneDrive - DOI/Alaska/norton2024/golovin/argus/intrinsic/camb_23271994_101123025/glvcamb_IO.mat')

mat = matlabtools.loadmat('/Users/dnowacki/OneDrive - DOI/Alaska/norton2024/golovin/argus/extrinsic/glvc1_gcpUVdInitial.mat')

UVd = {}

for a in mat['gcp']:
    for d in a._fieldnames:
        if d not in UVd:
            UVd[d] = []
        UVd[d].append(getattr(a, d))

for d in UVd:
    UVd[d] = np.stack(UVd[d])
# %%
assert np.all(dfxyz['gcp'] == UVd['num'])

# %%

# chatgpt
from scipy.optimize import curve_fit

# # Define the function that maps extrinsics and xyz to UV
# def xyz2DistUV(intrinsics, extrinsics, xyz):
#     # Implement the transformation logic here
#     # This is a placeholder; replace with actual implementation
#     # For example, you might apply a transformation based on camera extrinsics
#     # and return the corresponding UV coordinates
#     return np.zeros((xyz.shape[0], 2))  # Replace with actual computation

# Define the residual function for curve fitting
def residuals(extrinsics, xyz, UV, intrinsics):
    # Calculate the predicted UV values
    UV_predicted = xyz2DistUV(intrinsics, extrinsics, xyz)
    # Flatten the predicted UV to match the shape of the observed UV
    return np.concatenate((UV_predicted[:, 0], UV_predicted[:, 1])) - UV.flatten()

# Your input data

xyz = np.array(dfxyz[['x', 'y', 'z']])  # Nx3 matrix of x,y,z values
UV = np.array( UVd['UVd'])   # Nx2 matrix of pixel coordinates
intrinsics = intrin['intrinsics']
extrinsics_initial_guess = np.array([0,0,0,0,0,0])  # 1x6 matrix

# Flatten UV for fitting
# UV_flat = UV.flatten()

# xyz_flat = xyz.flatten()

# Use curve_fit to optimize the extrinsics
extrinsics_opt, covariance = curve_fit(
    lambda extrinsics, *xyz_flat: residuals(extrinsics, xyz_flat, UV_flat, intrinsics),
    xyz_flat,
    UV_flat,
    p0=extrinsics_initial_guess
)

# R and J are not directly available from curve_fit, but you can compute them if needed
# R is the residuals at the optimized parameters
R = residuals(extrinsics_opt, xyz_flat, UV, intrinsics)

# Covariance is already obtained from curve_fit
CovB = covariance

# Output the results
print("Optimized Extrinsics:", extrinsics_opt)
print("Residuals:", R)
print("Covariance Matrix:", CovB)


# %%
# [extrinsics,R,J,CovB]   = nlinfit(xyz,[UV(:,1); UV(:,2)],@(extrinsics,xyz)xyz2DistUV(intrinsics,extrinsics,xyz), extrinsicsInitialGuess);

camdata = corefunctions.CameraData(intrin['intrinsics'], [0,0,0,0,0,0])




dx = 0.1
dy = 0.1
xmin = 0
xmax = 300-dx
ymin = 0
ymax = 250-dy
z = 1

xyzgrid = corefunctions.XYZGrid([xmin, xmax],[ymin, ymax],dx,dy,z)

# scipy.optimize.curve_fit(xyz2DistUV, dfxyz[['x', 'y', 'z']], UVd['UVd'])
# scipy.optimize.curve_fit(xyz2DistUV, xyzgrid, UVd['UVd'])
extrin = [0, 0, 0, 0, 0, 0]
scipy.optimize.curve_fit(prepxyz, dfxyz[['x', 'y', 'z']], UVd['UVd'])
# %%


# Assuming xyz, UV, intrinsics, and extrinsicsInitialGuess are defined
intrinsics = intrin['intrinsics']
# Define the fitting function
def fit_function(xyz, *extrinsics):
    return xyz2DistUV(intrinsics, extrinsics, xyz)

# Prepare the data for fitting
# Flatten the UV data to match the expected input
UV = UVd['UVd']
UV_data = np.concatenate((UV[:, 0], UV[:, 1]))

extrinsicsInitialGuess = [0, 0, 0, 0, 0, 0]
xyz = dfxyz[['x', 'y', 'z']].values
# Perform the curve fitting
# The 'xyz' needs to be reshaped or passed appropriately
popt, pcov = scipy.optimize.curve_fit(fit_function, xyz, UVd['UVd'], p0=np.array(extrinsicsInitialGuess))

# Extract the optimized extrinsics
extrinsics = popt

# The covariance of the parameters can be obtained from pcov
CovB = pcov

# Note: The Jacobian is not directly available from curve_fit,
# but you can calculate it if needed based on the fitted function.



def prepxyz(xyz, extrin):
    print(extrin)

    return xyz2DistUV(intrin['intrinsics'], extrin, xyz)

def xyz2DistUV(intrinsics, extrinsics, xyz):
    print(intrinsics, extrinsics)
    # Take Calibration Information, combine it into a singular P matrix
    # containing both intrinsics and extrinsic information.
    P, K, R, IC = intrinsics_extrinsics_to_P(intrinsics, extrinsics)

    # Find the Undistorted UV Coordinates attributed to each xyz point.
    UV = P @ np.vstack((xyz.T, np.ones((1, xyz.shape[0]))))
    UV /= UV[2, :]  # Make Homogeneous

    # Distort the UV coordinates to pull the correct pixel values from the distorted image.
    Ud, Vd, flag = distort_UV(UV[0, :], UV[1, :], intrinsics)

    # Find Negative Zc Camera Coordinates. Adds invalid point to flag (=0).
    xyzC = R @ IC @ np.vstack((xyz.T, np.ones((1, xyz.shape[0]))))
    bind = np.where(xyzC[2, :] <= 0)[0]
    flag[bind] = 0

    # Make into a singular matrix for use in the non-linear solver
    UVd = np.vstack((Ud, Vd))

    return UVd, flag


import numpy as np

def distort_UV(U, V, intrinsics):
    # Section 1: Assign Coefficients out of Intrinsic Matrix
    NU = intrinsics[0]
    NV = intrinsics[1]
    c0U = intrinsics[2]
    c0V = intrinsics[3]
    fx = intrinsics[4]
    fy = intrinsics[5]
    d1 = intrinsics[6]
    d2 = intrinsics[7]
    d3 = intrinsics[8]
    t1 = intrinsics[9]
    t2 = intrinsics[10]

    # Section 2: Calculate Distorted Coordinates
    # Normalize Distances
    x = (U - c0U) / fx
    y = (V - c0V) / fy

    # Radial Distortion
    r2 = x**2 + y**2  # distortion found based on Large format units
    fr = 1 + d1 * r2 + d2 * r2**2 + d3 * r2**3

    # Tangential Distortion
    dx = 2 * t1 * x * y + t2 * (r2 + 2 * x**2)
    dy = t1 * (r2 + 2 * y**2) + 2 * t2 * x * y

    # Apply Correction, answer in chip pixel units
    xd = x * fr + dx
    yd = y * fr + dy
    Ud = xd * fx + c0U
    Vd = yd * fy + c0V

    # Section 3: Determine if Points are within the Image
    # Initialize Flag that all are acceptable.
    flag = np.ones_like(Ud, dtype=int)

    # Find negative UV coordinates
    bind = np.where(np.round(Ud) <= 0)[0]
    flag[bind] = 0

    # Find UVd coordinates greater than the image size
    bind = np.where(np.round(Ud) >= NU)[0]
    flag[bind] = 0
    bind = np.where(np.round(Vd) >= NV)[0]
    flag[bind] = 0

    # Section 4: Determine if Tangential Distortion is within Range
    # Find Maximum possible tangential distortion at corners
    Um = np.array([0, 0, NU, NU])
    Vm = np.array([0, NV, NV, 0])

    # Normalization
    xm = (Um - c0U) / fx
    ym = (Vm - c0V) / fy
    r2m = xm**2 + ym**2

    # Tangential Distortion
    dxm = 2 * t1 * xm * ym + t2 * (r2m + 2 * xm**2)
    dym = t1 * (r2m + 2 * ym**2) + 2 * t2 * xm * ym

    # Find Values Larger than those at corners
    bind = np.where(np.abs(dy) > np.max(np.abs(dym)))[0]
    flag[bind] = 0

    bind = np.where(np.abs(dx) > np.max(np.abs(dxm)))[0]
    flag[bind] = 0

    return Ud, Vd, flag


def CIRN_angles_to_R(azimuth, tilt, swing):
    # Section 1: Define R
    R = np.zeros((3, 3))

    R[0, 0] = -np.cos(azimuth) * np.cos(swing) - np.sin(azimuth) * np.cos(tilt) * np.sin(swing)
    R[0, 1] = np.cos(swing) * np.sin(azimuth) - np.sin(swing) * np.cos(tilt) * np.cos(azimuth)
    R[0, 2] = -np.sin(swing) * np.sin(tilt)

    R[1, 0] = -np.sin(swing) * np.cos(azimuth) + np.cos(swing) * np.cos(tilt) * np.sin(azimuth)
    R[1, 1] = np.sin(swing) * np.sin(azimuth) + np.cos(swing) * np.cos(tilt) * np.cos(azimuth)
    R[1, 2] = np.cos(swing) * np.sin(tilt)

    R[2, 0] = np.sin(tilt) * np.sin(azimuth)
    R[2, 1] = np.sin(tilt) * np.cos(azimuth)
    R[2, 2] = -np.cos(tilt)

    return R

def intrinsics_extrinsics_to_P(intrinsics, extrinsics):
    # Section 1: Format IO into K matrix
    fx = intrinsics[4]
    fy = intrinsics[5]
    c0U = intrinsics[2]
    c0V = intrinsics[3]

    K = np.array([[-fx, 0, c0U],
                  [0, -fy, c0V],
                  [0, 0, 1]])

    # Section 2: Format EO into Rotation Matrix R
    azimuth = extrinsics[3]
    tilt = extrinsics[4]
    swing = extrinsics[5]
    R = CIRN_angles_to_R(azimuth, tilt, swing)

    # Section 3: Format EO into Translation Matrix
    x = extrinsics[0]
    y = extrinsics[1]
    z = extrinsics[2]

    IC = np.hstack((np.eye(3), np.array([[-x], [-y], [-z]])))

    # Section 4: Combine K, Rotation, and Translation Matrix into P
    P = K @ R @ IC
    P /= P[2, 3]  # Normalize for Homogeneous Coordinates

    return P, K, R, IC
