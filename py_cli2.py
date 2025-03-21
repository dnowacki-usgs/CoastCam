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

# %   extrinsics is the solved camera EO where R,J,CovB are metrics of the solution
# %   and can be explored in nlinfit documentation.
#
# extrinsicsError=sqrt(diag(CovB));
intrinsics
# xyz = real world coordinates
# UV = distorted UV coordinates
dfxyz = pd.read_csv('/Users/dnowacki/OneDrive - DOI/Alaska/norton2024/golovin/argus/extrinsic/glvc1_beach_gcps.csv',
            header=None, names=['gcp', 'x', 'y', 'z'])
# %%

IO = matlabtools.loadmat('/Users/dnowacki/OneDrive - DOI/Alaska/norton2024/golovin/argus/intrinsic/camb_23271994_101123025/glvcamb_IO.mat')

gcpUVdInitial = matlabtools.loadmat('/Users/dnowacki/OneDrive - DOI/Alaska/norton2024/golovin/argus/extrinsic/glvc1_gcpUVdInitial.mat')

ioeo = matlabtools.loadmat('/Users/dnowacki/OneDrive - DOI/Alaska/norton2024/golovin/argus/extrinsic/glvc1_IOEOInitial.mat')

ioeo

UVd = {}

for a in gcpUVdInitial['gcp']:
    for d in a._fieldnames:
        if d not in UVd:
            UVd[d] = []
        UVd[d].append(getattr(a, d))

for d in UVd:
    UVd[d] = np.stack(UVd[d])
assert np.all(dfxyz['gcp'] == UVd['num'])
# %%



gcpsused = [ 9, 10, 11, 12, 13, 15, 16, 17, 18, 19, 20, 21] # from matlab output

xyz = np.array(dfxyz[dfxyz['gcp'].isin(gcpsused)][['x', 'y', 'z']])  # Nx3 matrix of x,y,z values

idx = [np.where(UVd['num'] == x)[0][0] for x in gcpsused]
UV = np.array(UVd['UVd'][idx])   # Nx2 matrix of pixel coordinates
intrinsics = IO['intrinsics']
# below is using some weird point far away from the camera
# extrinsics_initial_guess = np.array([594225, 7158849, 8.872, np.deg2rad(160), np.deg2rad(68), np.deg2rad(0)])  # 1x6 matrix
# based off of GCP048
extrinsics_initial_guess = np.array([594225, 7158849, 8.872, np.deg2rad(160), np.deg2rad(68), np.deg2rad(0)])  # 1x6 matrix
# %%

# %%
import numpy as np
from scipy.optimize import least_squares

def extrinsics_solver(extrinsics_initial_guess, extrinsics_knowns_flag, intrinsics, UV, xyz):
    """
    Solves for camera geometry EO (extrinsics) and associated errors given specified known values.

    Args:
        extrinsics_initial_guess (np.ndarray): 1x6 array [x, y, z, azimuth, tilt, swing]
        extrinsics_knowns_flag (np.ndarray): 1x6 array of 1s and 0s marking known values
        intrinsics (np.ndarray): 1x11 array of intrinsics parameters
        UV (np.ndarray): Px2 array of image coordinates
        xyz (np.ndarray): Px3 array of world coordinates

    Returns:
        tuple: (extrinsics, extrinsics_error)
    """
    # Convert inputs to numpy arrays
    UV = np.asarray(UV)
    xyz = np.asarray(xyz)
    extrinsics_initial_guess = np.asarray(extrinsics_initial_guess)
    extrinsics_knowns_flag = np.asarray(extrinsics_knowns_flag)

    # Section 1: If All Values of extrinsics are Unknown
    if np.sum(extrinsics_knowns_flag) == 0:
        # Define residual function for least squares optimization
        def residual_func(extrinsics):
            UVd = xyz2_dist_UV(intrinsics, extrinsics, xyz)[0]
            return np.concatenate([UVd[0:len(UV)] - UV[:,0],
                                 UVd[len(UV):] - UV[:,1]])

        # Perform optimization
        result = least_squares(residual_func, extrinsics_initial_guess)
        extrinsics = result.x

        # Calculate errors (approximating MATLAB's nlinfit error estimation)
        J = result.jac
        residuals = result.fun
        mse = np.sum(residuals**2) / (len(residuals) - len(extrinsics))
        cov_matrix = mse * np.linalg.inv(J.T @ J)
        extrinsics_error = np.sqrt(np.diag(cov_matrix))

    # Section 2: If any values of extrinsics are known
    else:
        known_ind = np.where(extrinsics_knowns_flag == 1)[0]
        unknown_ind = np.where(extrinsics_knowns_flag == 0)[0]

        extrinsics_known = extrinsics_initial_guess[known_ind]
        extrinsics_unknown_initial_guess = extrinsics_initial_guess[unknown_ind]

        # Define residual function for partial optimization
        def residual_func_partial(extrinsics_unknown):
            return xyz2_dist_UV_for_nlinfit(extrinsics_knowns_flag,
                                          extrinsics_known,
                                          extrinsics_unknown,
                                          intrinsics,
                                          xyz)[0] - np.concatenate([UV[:,0], UV[:,1]])

        # Perform optimization on unknown parameters
        result = least_squares(residual_func_partial, extrinsics_unknown_initial_guess)
        e_unknown_sol = result.x

        # Calculate errors for unknown parameters
        J = result.jac
        residuals = result.fun
        mse = np.sum(residuals**2) / (len(residuals) - len(e_unknown_sol))
        cov_matrix = mse * np.linalg.inv(J.T @ J)
        e_unknown_sol_error = np.sqrt(np.diag(cov_matrix))

        # Format final results
        extrinsics = np.full(6, np.nan)
        extrinsics[known_ind] = extrinsics_known
        extrinsics[unknown_ind] = e_unknown_sol

        extrinsics_error = np.full(6, np.nan)
        extrinsics_error[known_ind] = 0  # Known parameters have zero error
        extrinsics_error[unknown_ind] = e_unknown_sol_error

    return extrinsics, extrinsics_error

def xyz2_dist_UV_for_nlinfit(extrinsics_knowns_flag, extrinsics_known,
                            extrinsics_unknown, intrinsics, xyz):
    """
    Helper function for partial extrinsics optimization.
    """
    k_ind = np.where(extrinsics_knowns_flag == 1)[0]
    uk_ind = np.where(extrinsics_knowns_flag == 0)[0]

    i_extrinsics = np.full(6, np.nan)
    i_extrinsics[k_ind] = extrinsics_known
    i_extrinsics[uk_ind] = extrinsics_unknown

    return xyz2_dist_UV(intrinsics, i_extrinsics, xyz)

def xyz2_dist_UV(intrinsics, extrinsics, xyz):
    """
    Computes distorted UV coordinates from world coordinates.
    """
    xyz = np.asarray(xyz)
    P, K, R, IC = intrinsics_extrinsics_2P(intrinsics, extrinsics)

    # Convert to homogeneous coordinates
    homog_coords = np.hstack([xyz, np.ones((xyz.shape[0], 1))]).T
    UV = P @ homog_coords
    UV = UV / UV[2]  # Normalize homogeneous coordinates

    # Calculate distorted coordinates
    Ud, Vd, flag = distort_UV(UV[0], UV[1], intrinsics)

    # Check for negative Z camera coordinates
    xyz_C = R @ IC @ homog_coords
    bind = xyz_C[2] <= 0
    flag[bind] = 0

    return np.concatenate([Ud, Vd]), flag

def intrinsics_extrinsics_2P(intrinsics, extrinsics):
    """
    Creates camera P matrix from intrinsics and extrinsics.
    """
    # Extract parameters
    fx, fy = intrinsics[4:6]
    c0U, c0V = intrinsics[2:4]

    # Create K matrix
    K = np.array([
        [-fx, 0, c0U],
        [0, -fy, c0V],
        [0, 0, 1]
    ])

    # Get rotation matrix
    R = CIRN_angles_2R(extrinsics[3], extrinsics[4], extrinsics[5])

    # Create translation matrix
    IC = np.eye(4)[:3]  # Take first 3 rows
    IC[:, 3] = -extrinsics[:3]

    # Combine into P matrix
    P = K @ R @ IC
    P = P / P[2, 3]  # Normalize for homogeneous coordinates

    return P, K, R, IC

def distort_UV(U, V, intrinsics):
    """
    Applies lens distortion to UV coordinates.
    """
    # Extract parameters
    NU, NV = intrinsics[:2]
    c0U, c0V = intrinsics[2:4]
    fx, fy = intrinsics[4:6]
    d1, d2, d3 = intrinsics[6:9]
    t1, t2 = intrinsics[9:11]

    # Normalize coordinates
    x = (U - c0U) / fx
    y = (V - c0V) / fy

    # Calculate radial distortion
    r2 = x*x + y*y
    fr = 1 + d1*r2 + d2*r2*r2 + d3*r2*r2*r2

    # Calculate tangential distortion
    dx = 2*t1*x*y + t2*(r2 + 2*x*x)
    dy = t1*(r2 + 2*y*y) + 2*t2*x*y

    # Apply distortion
    xd = x*fr + dx
    yd = y*fr + dy
    Ud = xd*fx + c0U
    Vd = yd*fy + c0V

    # Initialize validity flags
    flag = np.ones_like(U)

    # Check image bounds
    flag[(Ud <= 0) | (Vd <= 0)] = 0
    flag[(Ud >= NU) | (Vd >= NV)] = 0

    # Check tangential distortion bounds
    Um = np.array([0, 0, NU, NU])
    Vm = np.array([0, NV, NV, 0])

    xm = (Um - c0U) / fx
    ym = (Vm - c0V) / fy
    r2m = xm*xm + ym*ym

    dxm = 2*t1*xm*ym + t2*(r2m + 2*xm*xm)
    dym = t1*(r2m + 2*ym*ym) + 2*t2*xm*ym

    flag[np.abs(dy) > np.max(np.abs(dym))] = 0
    flag[np.abs(dx) > np.max(np.abs(dxm))] = 0

    return Ud, Vd, flag

def CIRN_angles_2R(azimuth, tilt, swing):
    """
    Converts CIRN angles to rotation matrix.
    Note: This is a placeholder - implement actual CIRN angle conversion
    """
    # Create rotation matrices for each angle
    Rz = np.array([
        [np.cos(azimuth), -np.sin(azimuth), 0],
        [np.sin(azimuth), np.cos(azimuth), 0],
        [0, 0, 1]
    ])

    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(tilt), -np.sin(tilt)],
        [0, np.sin(tilt), np.cos(tilt)]
    ])

    Ry = np.array([
        [np.cos(swing), 0, np.sin(swing)],
        [0, 1, 0],
        [-np.sin(swing), 0, np.cos(swing)]
    ])

    # Combine rotations
    R = Rz @ Rx @ Ry
    return R

import numpy as np
from scipy.optimize import curve_fit

def extrinsics_solver_curvefit(extrinsics_initial_guess, extrinsics_knowns_flag, intrinsics, UV, xyz):
    """
    Solves for camera geometry EO (extrinsics) and associated errors using scipy.optimize.curve_fit
    Modified to more closely match MATLAB's nlinfit behavior.
    """
    # Convert inputs to numpy arrays
    UV = np.asarray(UV)
    xyz = np.asarray(xyz)
    extrinsics_initial_guess = np.asarray(extrinsics_initial_guess)
    extrinsics_knowns_flag = np.asarray(extrinsics_knowns_flag)

    # Flatten UV for curve_fit
    UV_flat = np.concatenate([UV[:,0], UV[:,1]])

    # Section 1: If All Values of extrinsics are Unknown
    if np.sum(extrinsics_knowns_flag) == 0:
        # Define function for curve_fit
        def fit_func(xyz_flat, *params):
            extrinsics = np.array(params)
            UVd = xyz2_dist_UV(intrinsics, extrinsics, xyz_flat.reshape(-1, 3))[0]
            return UVd

        # Reshape xyz for curve_fit
        xyz_flat = xyz.reshape(-1, 3)

        try:
            # Perform optimization using Levenberg-Marquardt (like MATLAB)
            popt, pcov = curve_fit(
                f=fit_func,
                xdata=xyz_flat,
                ydata=UV_flat,
                p0=extrinsics_initial_guess,
                method='lm',  # Levenberg-Marquardt like MATLAB
                ftol=1e-8,
                xtol=1e-8
            )

            extrinsics = popt

            # Calculate errors, handling singular covariance matrix
            if np.any(np.diag(pcov) < 0):
                # Use a minimum error estimate if covariance matrix is singular
                xyz_range = np.ptp(xyz, axis=0)
                extrinsics_error = np.array([
                    xyz_range[0] * 0.01,  # 1% of range for positions
                    xyz_range[1] * 0.01,
                    xyz_range[2] * 0.01,
                    np.deg2rad(1),  # 1 degree for angles
                    np.deg2rad(1),
                    np.deg2rad(1)
                ])
            else:
                extrinsics_error = np.sqrt(np.diag(pcov))

        except (RuntimeError, ValueError) as e:
            print(f"Curve fit failed: {str(e)}")
            return None, None

    # Section 2: If any values of extrinsics are known
    else:
        known_ind = np.where(extrinsics_knowns_flag == 1)[0]
        unknown_ind = np.where(extrinsics_knowns_flag == 0)[0]

        extrinsics_known = extrinsics_initial_guess[known_ind]
        extrinsics_unknown_initial_guess = extrinsics_initial_guess[unknown_ind]

        # Define function for partial optimization
        def fit_func_partial(xyz_flat, *params):
            extrinsics_unknown = np.array(params)
            UVd = xyz2_dist_UV_for_nlinfit(
                extrinsics_knowns_flag,
                extrinsics_known,
                extrinsics_unknown,
                intrinsics,
                xyz_flat.reshape(-1, 3)
            )[0]
            return UVd

        # Reshape xyz for curve_fit
        xyz_flat = xyz.reshape(-1, 3)

        try:
            # Perform optimization on unknown parameters
            popt, pcov = curve_fit(
                f=fit_func_partial,
                xdata=xyz_flat,
                ydata=UV_flat,
                p0=extrinsics_unknown_initial_guess,
                method='lm',
                ftol=1e-8,
                xtol=1e-8
            )

            e_unknown_sol = popt

            # Calculate errors for unknown parameters
            if np.any(np.diag(pcov) < 0):
                xyz_range = np.ptp(xyz, axis=0)
                e_unknown_sol_error = np.array([
                    xyz_range[0] * 0.01 if i < 3 else np.deg2rad(1)
                    for i in range(len(unknown_ind))
                ])
            else:
                e_unknown_sol_error = np.sqrt(np.diag(pcov))

            # Format final results
            extrinsics = np.full(6, np.nan)
            extrinsics[known_ind] = extrinsics_known
            extrinsics[unknown_ind] = e_unknown_sol

            extrinsics_error = np.full(6, np.nan)
            extrinsics_error[known_ind] = 0  # Known parameters have zero error
            extrinsics_error[unknown_ind] = e_unknown_sol_error

        except (RuntimeError, ValueError) as e:
            print(f"Curve fit failed: {str(e)}")
            return None, None

    return extrinsics, extrinsics_error
# %%

extrinsics, errors = extrinsics_solver(
    extrinsics_initial_guess=extrinsics_initial_guess,
    extrinsics_knowns_flag=np.array([1, 1, 0, 0, 0, 0]),  # all unknown
    intrinsics=intrinsics,
    UV=UV,
    xyz=xyz
)

extrinsics, errors = extrinsics_solver_curvefit(
    extrinsics_initial_guess=np.array([594225, 7158849, 8.872, np.deg2rad(160), np.deg2rad(68), np.deg2rad(0)]),  # 1x6 matrix
    extrinsics_knowns_flag=np.array([0, 0, 0, 0, 0, 0]),  # all unknown
    intrinsics=intrinsics,
    UV=UV,
    xyz=xyz
)
# %%
import matplotlib.pyplot as plt
plt.plot(xyz[:,0], xyz[:,1], '.')
plt.plot(extrinsics_initial_guess[0], extrinsics_initial_guess[1], 'o')
plt.plot(extrinsics[0], extrinsics[1], '*')

plt.plot(ioeo['extrinsics'][0], ioeo['extrinsics'][1], 's')
plt.axis('equal')

# %%
extrinsics - ioeo['extrinsics']
