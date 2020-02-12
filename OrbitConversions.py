import numpy as np
from Transformations import *

def cart2kep_Fife(r, v, mu=1.0, tol=1e-12):
	"""
	Convert inertial position and velocity to classical orbital elements.

	If mu = 1.0, units are DU and TU
	If mu = 398600.4415, units are kilometers and seconds

	Inputs
	------
		r - Position vector (1x3 numpy array)
		v - Velocity vector (1x3 numpy array)

	Output
	------
		a     - Semimajor axis (km or DU)
		e     - Eccentricity
		i     - Inclination (radians)
		RAAN  - Right Ascension of ascending node (radians)
		argp  - Argument of periapsis (radians)
		theta - True anomaly (radians)

	if e = 0 and i = 0
		return true_lon (true longitude of periapsis) instead of theta

	if e = 0
		return argl (argument of latitude) instead of theta

	if i = 0
		return lonp (longitude of periapsis) instead of argp

	"""
	v_mag = np.linalg.norm(v)
	r_mag = np.linalg.norm(r)

	# Compute specific energy and determine orbit type
	eps = (v_mag**2)/2. - mu/r_mag

	if eps < 0.:
		orbit = 'Elliptical'
		a     = -mu/(2.*eps)
	elif eps == 0.:
		orbit = 'Parabolic'
		a     = np.inf
	else:
		orbit = 'Hyperbolic'
		a     = mu/(2.*eps)

	# Compute eccentricity and its magnitude
	e = (1./mu)*((v_mag**2 - mu/r_mag)*r - np.inner(r,v)*v)
	e_mag = np.linalg.norm(e)

	# Compute angular momentum and its magnitude
	h     = np.cross(r,v)
	h_mag = np.linalg.norm(h) 

	# Compute inclination
	i = acos(np.inner(h, np.array([0.,0.,1.]))/h_mag)

	# Handle singularities
	if e_mag < tol and i < tol:
		RAAN     = 0.
		argp     = 0.
		true_lon = acos(r[1]/r_mag)
		if r[1] < 0.:
			true_lon = 2*np.pi - true_lon

		return a, e_mag, np.rad2deg(i), np.rad2deg(RAAN), np.rad2deg(argp), np.rad2deg(true_lon)

	elif e_mag < tol:
		nodes = np.cross(np.array([0.,0.,1.]),h)
		n_hat = nodes/np.linalg.norm(nodes)
		RAAN  = acos(nodes[0]/np.linalg.norm(nodes))
		if nodes[1] < 0.:
			RAAN = 2*np.pi - RAAN

		argp  = 0.
		argl  = acos(np.inner(n_hat, r)/r_mag)
		if r[2] < 0.:
			argl = 2*np.pi - argl

		return a, e_mag, np.rad2deg(i), np.rad2deg(RAAN), np.rad2deg(argp), np.rad2deg(argl)

	elif i < tol:
		RAAN = 0.
		lonp = acos(np.inner(e/e_mag, np.array([1.,0.,0.])))
		if e[1] < 0.:
			lonp = 2*np.pi - lonp

		theta = acos(np.inner(r/r_mag, e/e_mag))
		if np.inner(r,v) < 0.:
			theta = 2*np.pi - theta

		return a, e_mag, np.rad2deg(i), np.rad2deg(RAAN), np.rad2deg(lonp), np.rad2deg(theta)

	# If no singularites, return regular classical elements
	nodes = np.array([0.,0.,1.])*h
	n_hat = nodes/np.linalg.norm(nodes)
	RAAN  = acos(nodes[0]/np.linalg.norm(nodes))
	if nodes[1] < 0.:
		RAAN = 2*np.pi - RAAN

	argp  = acos(np.inner(n_hat, e/e_mag))
	if e[2] < 0.:
		argp = 2*np.pi - argp

	theta = acos(np.inner(r/r_mag, e/e_mag))
	if np.inner(r,v) < 0.:
		theta = 2*np.pi - theta

	return np.array([a, e_mag, np.rad2deg(i), np.rad2deg(RAAN), np.rad2deg(argp), np.rad2deg(theta)])


def kep2cart_Fife(a, e_mag, i, RAAN, argp, theta, mu=1.0, tol=1e-12):
	"""
	Convert classical orbital elements into cartesian position and velocity.

	If mu = 1.0, units are DU and TU
	If mu = 398600.4415, units are kilometers and seconds

	Inputs
	------
		a     - Semimajor axis (km or DU)
		e     - Eccentricity
		i     - Inclination                       (degrees)
		RAAN  - Right Ascension of ascending node (degrees)
		argp  - Argument of periapsis             (degrees)
		theta - True anomaly                      (degrees)

	Outputs
	-------
		r     - Inertial position
		v     - Inertial velocity

	"""

	# solve for r and v in PQW frame
	i     = np.deg2rad(i)
	RAAN  = np.deg2rad(RAAN)
	argp  = np.deg2rad(argp)
	theta = np.deg2rad(theta)
	p = a*(e_mag**2 - 1)
	if e_mag < 1.0:
		p = a*(1-e_mag**2.)
		
	r_mag_pqw = p/(1 + e_mag*np.cos(theta))
	r         = r_mag_pqw*np.array([np.cos(theta), np.sin(theta), 0.])
	#print r
	v_mag_pqw = np.sqrt(mu/p)
	v         = v_mag_pqw*np.array([-np.sin(theta), e_mag + np.cos(theta), 0.])
	#print v

	# Handle special cases for PQW -> IJK transformation
	if e_mag < tol and i < tol:
		Q     = np.eye(3)
		r_ijk = np.dot(Q, r)
		v_ijk = np.dot(Q, v)
		return r_ijk, v_ijk

	elif e_mag < tol:
		Q     = np.dot(R3(-RAAN), R1(-i))
		r_ijk = np.dot(Q, r)
		v_ijk = np.dot(Q, v)
		return r_ijk, v_ijk

	elif i < tol:
		Q     = R3(-argp)
		r_ijk = np.dot(Q, r)
		v_ijk = np.dot(Q, v)
		return r_ijk, v_ijk

	Q     = np.dot(np.dot(R3(-RAAN), R1(-i)), R3(-argp))
	r_ijk = np.dot(Q, r)
	v_ijk = np.dot(Q, v)

	return r_ijk, v_ijk 