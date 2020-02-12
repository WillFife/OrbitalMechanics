import numpy as np


def acos(arg):
	if abs(arg-1.0) < 1e-12 and arg > 1.0:
		arg = 1.0

	elif abs(arg+1.0) < 1e-12 and arg < -1.0:
		arg = -1.0

	angle = np.arccos(arg)

	return angle


def asin(arg):
	if abs(arg-1.0) < 1e-12 and arg > 1.0:
		arg = 1.0

	elif abs(arg+1.0) < 1e-12 and arg < -1.0:
		arg = -1.0

	angle = np.arcsin(arg)

	return angle


def R3(angle):
	"""
	DCM for rotation about axis 3 by angle (rad)
	"""
	R = np.eye(3)
	R[0,0] = np.cos(angle)
	R[0,1] = np.sin(angle)
	R[1,0] = -np.sin(angle)
	R[1,1] = np.cos(angle)

	return R


def R2(angle):
	"""
	DCM for rotation about axis 2 by angle (rad)
	"""
	R = np.eye(3)
	R[0,0] = np.cos(angle)
	R[0,2] = -np.sin(angle)
	R[2,0] = np.sin(angle)
	R[2,2] = np.cos(angle)

	return R


def R1(angle):
	"""
	DCM for rotation about axis 1 by angle (rad)
	"""
	R = np.eye(3)
	R[1,1] = np.cos(angle)
	R[1,2] = np.sin(angle)
	R[2,1] = -np.sin(angle)
	R[2,2] = np.cos(angle)

	return R


def Q_ITRF_GCRF():
	pass