import numpy as np
from scipy.optimize import curve_fit, least_squares

def CIRNangles2R(azimuth, tilt, swing):
    """Convert CIRN angles to rotation matrix."""
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

def intrinsicsExtrinsics2P(intrinsics, extrinsics):
    """Create camera P matrix from intrinsics and extrinsics."""
    # Format IO into K matrix
    fx, fy = intrinsics[4:6]
    c0U, c0V = intrinsics[2:4]

    K = np.array([
        [-fx, 0, c0U],
        [0, -fy, c0V],
        [0, 0, 1]
    ])

    # Format EO into Rotation Matrix R
    azimuth, tilt, swing = extrinsics[3:6]
    R = CIRNangles2R(azimuth, tilt, swing)

    # Format EO into Translation Matrix
    x, y, z = extrinsics[0:3]
    IC = np.eye(4)[:3]  # Get first 3 rows of 4x4 identity matrix
    IC[:, 3] = [-x, -y, -z]

    # Combine K, Rotation, and Translation Matrix into P
    P = K @ R @ IC
    P = P / P[2, 3]  # Normalize for Homogeneous Coordinates

    return P, K, R, IC

def distortUV(U, V, intrinsics):
    """Distort UV coordinates using Caltech lens distortion models."""
    NU, NV, c0U, c0V, fx, fy, d1, d2, d3, t1, t2 = intrinsics

    # Normalize Distances
    x = (U - c0U) / fx
    y = (V - c0V) / fy

    # Radial Distortion
    r2 = x * x + y * y
    fr = 1 + d1 * r2 + d2 * r2 * r2 + d3 * r2 * r2 * r2

    # Tangential Distortion
    dx = 2 * t1 * x * y + t2 * (r2 + 2 * x * x)
    dy = t1 * (r2 + 2 * y * y) + 2 * t2 * x * y

    # Apply Correction
    xd = x * fr + dx
    yd = y * fr + dy
    Ud = xd * fx + c0U
    Vd = yd * fy + c0V

    # Initialize flag
    flag = np.ones_like(U, dtype=bool)

    # Check boundaries
    flag &= (np.round(Ud) > 0) & (np.round(Vd) > 0)
    flag &= (np.round(Ud) < NU) & (np.round(Vd) < NV)

    # Check tangential distortion
    Um = np.array([0, 0, NU, NU])
    Vm = np.array([0, NV, NV, 0])

    xm = (Um - c0U) / fx
    ym = (Vm - c0V) / fy
    r2m = xm * xm + ym * ym

    dxm = 2 * t1 * xm * ym + t2 * (r2m + 2 * xm * xm)
    dym = t1 * (r2m + 2 * ym * ym) + 2 * t2 * xm * ym

    flag &= (np.abs(dy) <= np.max(np.abs(dym)))
    flag &= (np.abs(dx) <= np.max(np.abs(dxm)))

    return Ud, Vd, flag

def xyz2DistUV(intrinsics, extrinsics, xyz):
    """Convert world coordinates to distorted UV coordinates."""
    P, K, R, IC = intrinsicsExtrinsics2P(intrinsics, extrinsics)

    # Convert to homogeneous coordinates
    xyz_h = np.hstack([xyz, np.ones((xyz.shape[0], 1))])

    # Find undistorted UV coordinates
    UV = P @ xyz_h.T
    UV = UV / UV[2]  # Make homogeneous

    # Distort UV coordinates
    Ud, Vd, flag = distortUV(UV[0], UV[1], intrinsics)

    # Check for negative Zc Camera Coordinates
    xyzC = R @ IC @ xyz_h.T
    flag &= (xyzC[2] > 0)

    return np.vstack([Ud, Vd]), flag

def objective_function(params, xyz, UV_target, intrinsics, known_params=None, known_mask=None):
    """Objective function for optimization."""
    if known_mask is not None:
        # Combine known and unknown parameters
        full_params = np.zeros(6)
        full_params[known_mask == 1] = known_params
        full_params[known_mask == 0] = params
    else:
        full_params = params

    UVd, _ = xyz2DistUV(intrinsics, full_params, xyz)
    return (UVd.flatten() - UV_target.flatten())

def extrinsicsSolver_leastsq(extrinsicsInitialGuess, extrinsicsKnownsFlag, intrinsics, UV, xyz):
    """Solve for camera extrinsics using least_squares."""
    # Convert all inputs to numpy arrays
    extrinsicsInitialGuess = np.array(extrinsicsInitialGuess, dtype=np.float64)
    extrinsicsKnownsFlag = np.array(extrinsicsKnownsFlag, dtype=int)
    intrinsics = np.array(intrinsics, dtype=np.float64)
    UV = np.array(UV, dtype=np.float64)
    xyz = np.array(xyz, dtype=np.float64)

    if np.sum(extrinsicsKnownsFlag) == 0:
        # All parameters unknown
        x0 = extrinsicsInitialGuess
        result = least_squares(
            objective_function,
            x0,
            args=(xyz, UV, intrinsics)
        )
        extrinsics = result.x
        J = result.jac
        extrinsicsError = np.sqrt(np.diag(np.linalg.inv(J.T @ J)))

    else:
        # Some parameters known
        known_mask = extrinsicsKnownsFlag == 1
        known_params = extrinsicsInitialGuess[known_mask]
        x0 = extrinsicsInitialGuess[~known_mask]  # Initial guess for unknown parameters only

        result = least_squares(
            objective_function,
            x0,
            args=(xyz, UV, intrinsics, known_params, extrinsicsKnownsFlag)
        )

        # Reconstruct full solution
        extrinsics = np.zeros(6)
        extrinsics[known_mask] = known_params
        extrinsics[~known_mask] = result.x

        # Calculate errors
        J = result.jac
        unknown_errors = np.sqrt(np.diag(np.linalg.inv(J.T @ J)))
        extrinsicsError = np.zeros(6)
        extrinsicsError[known_mask] = 0
        extrinsicsError[~known_mask] = unknown_errors

    return extrinsics, extrinsicsError

def extrinsicsSolver_curvefit(extrinsicsInitialGuess, extrinsicsKnownsFlag, intrinsics, UV, xyz):
    """Solve for camera extrinsics using curve_fit."""
    UV_flat = UV.reshape(-1)  # Flatten UV for optimization

    def fit_function(x, *params):
        return xyz2DistUV_wrapper(params, x, intrinsics)

    if np.sum(extrinsicsKnownsFlag) == 0:  # All parameters unknown
        popt, pcov = curve_fit(
            fit_function,
            xyz,
            UV_flat,
            p0=extrinsicsInitialGuess
        )
        extrinsics = popt
        extrinsicsError = np.sqrt(np.diag(pcov))
    else:  # Some parameters known
        known_idx = extrinsicsKnownsFlag == 1
        unknown_idx = extrinsicsKnownsFlag == 0

        extrinsics_known = extrinsicsInitialGuess[known_idx]
        initial_unknown = extrinsicsInitialGuess[unknown_idx]

        def fit_function_partial(x, *params):
            return xyz2DistUV_wrapper(params, x, intrinsics, extrinsics_known, extrinsicsKnownsFlag)

        popt, pcov = curve_fit(
            fit_function_partial,
            xyz,
            UV_flat,
            p0=initial_unknown
        )

        extrinsics = np.zeros(6)
        extrinsics[known_idx] = extrinsics_known
        extrinsics[unknown_idx] = popt

        extrinsicsError = np.zeros(6)
        extrinsicsError[known_idx] = 0
        extrinsicsError[unknown_idx] = np.sqrt(np.diag(pcov))

    return extrinsics, extrinsicsError

def xyz2DistUV_wrapper(params, xyz, intrinsics, known_params=None, known_mask=None):
    """Wrapper function for optimization."""
    params = np.asarray(params).ravel()

    if known_mask is not None:
        full_params = np.zeros(6)
        full_params[known_mask == 1] = known_params
        full_params[known_mask == 0] = params
        params = full_params

    UVd, _ = xyz2DistUV(intrinsics, params, xyz)
    return UVd.flatten()
# %%
extrinsicsInitialGuess=np.array([594225, 7158849, 8.872, np.deg2rad(160), np.deg2rad(68), np.deg2rad(0)]),  # 1x6 matrix
extrinsicsKnownsFlag=np.array([0, 0, 0, 0, 0, 0])
intrinsics = np.array([ 2.44800000e+03,  2.04800000e+03,  1.19782453e+03,  1.03812369e+03,        3.59842029e+03,  3.59765712e+03, -2.26643143e-01,  1.70268088e-01,        0.00000000e+00, -1.21848465e-03,  5.65668454e-04])
UV = np.array([[2150.82548424, 1261.71305166],
       [2169.57787691, 1090.63116616],
       [2200.48737813,  919.01652241],
       [ 538.62108668,  883.04428281],
       [ 356.97441526,  651.21577069],
       [1130.66918421,  562.20615256],
       [ 934.69524393,  494.2304069 ],
       [ 678.40929284,  488.92932789],
       [ 563.64838205,  521.35941833],
       [1422.7755111 ,  662.21407641],
       [1020.00373959,  631.08495896],
       [ 968.47078808,  697.06569212]])

xyz = np.array([[5.94252816e+05, 7.15884495e+06, 3.44400000e+00],
       [5.94261841e+05, 7.15884284e+06, 2.83600000e+00],
       [5.94279426e+05, 7.15883853e+06, 1.62000000e+00],
       [5.94258960e+05, 7.15885979e+06, 4.87800000e+00],
       [5.94276487e+05, 7.15886683e+06, 4.87500000e+00],
       [5.94333009e+05, 7.15885932e+06, 1.50400000e+00],
       [5.94357868e+05, 7.15886849e+06, 1.45800000e+00],
       [5.94337412e+05, 7.15887410e+06, 2.86500000e+00],
       [5.94315992e+05, 7.15887289e+06, 3.53400000e+00],
       [5.94309201e+05, 7.15885058e+06, 1.59100000e+00],
       [5.94300854e+05, 7.15885936e+06, 2.82500000e+00],
       [5.94285939e+05, 7.15885871e+06, 3.31300000e+00]])

# Using curve_fit

extrinsics, extrinsics_error = extrinsicsSolver_curvefit(
    extrinsicsInitialGuess,
    extrinsicsKnownsFlag,
    intrinsics,
    UV,
    xyz
)
# %%
# Or using least_squares
extrinsics, extrinsics_error = extrinsicsSolver_leastsq(
    extrinsicsInitialGuess,
    extrinsicsKnownsFlag,
    intrinsics,
    UV,
    xyz
)
