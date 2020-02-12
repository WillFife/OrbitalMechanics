import numpy as np
from OrbitConversions import cart2kep_Fife, kep2cart_Fife
from scipy.integrate import ode
import matplotlib.pyplot as plt

def give_me_orbital_elements(r, v, mu=398600.4415):
	"""
	Given position and velocity VECTORS, find:
	1. Angular momentum vector & magnitude
	2. specific energy
	3. eccentricity vector
	4. true anomaly
	5. eccentricity magnitude
	6. semimajor axis
	7. periapsis
	8. apoapsis
	"""
	r_mag = np.linalg.norm(r)
	v_mag = np.linalg.norm(v)

	# Angular momentum
	h = np.cross(r, v)
	h_mag = np.linalg.norm(h)

	# specific energy
	eps = (v_mag**2)/2 - (mu/r_mag)

	# eccentricity vector
	e = np.cross(v, h)/mu - r/r_mag

	# semimajor axis
	a = -mu/(2*eps)

	# semiparameter
	p = (h_mag**2)/mu

	# eccentricity magnitude
	e_mag = np.sqrt(1 - (p/a))

	# periapsis
	r_p = a*(1-e_mag)

	# apoapsis
	r_a = a*(1+e_mag)

	# true anomaly
	theta = np.arccos((p-r_mag)/(r_mag*e_mag))
	if sum(r*v) < 0.0:
		theta = np.rad2deg(2*np.pi - theta)

	else:
		theta = np.rad2deg(theta)

	print('h = ', h)
	print('eps = ', eps)
	print('e = ', e)
	print('theta = ', theta)
	print('e_mag = ', e_mag)
	print('a = ', a)
	print('r_p = ', r_p)
	print('r_a = ', r_a)

	return h, a, e, e_mag, eps, np.deg2rad(theta), h_mag


def r_orbit_eq(theta, h_mag=0., e_mag=0., mu=398600.4415):
	"""
	given e_mag and h, use orbit equation to solve for r
	"""
	p  = (h_mag**2)/mu
	c2 = 1./(1 + e_mag*np.cos(np.deg2rad(theta)))

	return p*c2


def specific_potential_energy(r, mu=398600.4415):
	return mu/r


def specific_kinetic_energy_from_eps(eps, r, mu=1):
	specific_potential = specific_potential_energy(r)

	return eps + specific_potential

def specific_kinetic_energy(v):
	return 0.5*v**2


def E_newton(M, e, tol=10e-10, max_iter=20):
	"""
	Use newton-raphson method to solve for E
	"""
	if M < np.pi:
		E_prev = M + e/2
	else:
		E_prev = M - e/2

	for i in range(max_iter):
		E_next = E_prev - (E_prev - e*np.sin(E_prev) - M)/(1 - e*np.cos(E_prev))
		if E_next - E_prev < tol:
			return E_next
		
		E_prev = E_next
		if i==max_iter-1:
			return E_next


def F_newton(M, e, tol=10e-10, max_iter=20):
	"""
	Use newton-raphson method to solve for F
	"""
	F_prev = M

	for i in range(max_iter):
		F_next = F_prev - (e*np.sinh(F_prev) - F_prev - M)/(e*np.cosh(F_prev) - 1)
		if F_next - F_prev < tol:
			return F_next

		F_prev = F_next
		if i==max_iter-1:
			return F_next


def E_from_theta(theta, e):
	E = 2*np.arctan2(np.sqrt(1 - e)*np.tan(theta/2.), np.sqrt(1+e))
	return E

def theta_from_E(E, e):
	theta = 2*np.arctan2(np.sqrt(1 + e)*np.tan(E/2.), np.sqrt(1-e))
	return theta

def M_from_F(F, e):
	return -F + e*np.sinh(F)

def M_from_E(E, e):
	return E - e*np.sin(E)

def F_from_theta(theta, e):
	F = 2*np.arctanh(np.sqrt(e - 1)*np.tan(theta/2.)/(np.sqrt(e+1)))
	return F

def theta_from_F(F, e):
	return 2*np.arctan2(np.sqrt(e+1)*np.tanh(F/2.), np.sqrt(e-1))

def speed_from_r(r, a, mu=398600.4415):
	return np.sqrt(mu*((2/r) - (1./a)))

def orbit_propagation(times, n, e, tp, a, h_mag, mu=398600.4415, vector=False, **kw_args):
	"""
	Propagate an orbit in time.

	Inputs:
		times - time array (np.array)
		n     - mean motion (float)
		e     - eccentricity (float)
		tp    - time at periapsis (float)

	Output:
		thetas - True anomalies at those times (deg)
		Es     - Eccentric anomalies 
		Ms     - Mean anomalies (deg)
		rs     - positions at those times
		vs     - velocity at those times

	"""
	Elliptical = True
	if vector:
		i    = kw_args['i']
		RAAN = kw_args['RAAN']
		argp = kw_args['argp']
		print(argp)

	if e > 1.0:
		Elliptical = False

	thetas = []
	Es     = []
	Ms     = []
	rs     = []
	vs     = []
	energies = []

	if Elliptical:
		for i in range(len(times)):
			M = n*(times[i] - tp)
			M = np.mod(M, 2*np.pi)
			Ms.append(M)

			# solve for E
			E = E_newton(M, e)
			Es.append(E)

			# solve for theta
			theta = theta_from_E(E, e)
			thetas.append(theta)

			# solve for r
			if vector:
				 r, v = kep2cart_Fife(a, e, i, RAAN, argp, np.rad2deg(theta), mu=mu)
				 rs.append(r)
				 vs.append(v)

			else:
				r = r_orbit_eq(np.rad2deg(theta), h_mag=h_mag, e_mag=e)
				rs.append(r)

				# solve for v
				v = speed_from_r(r, a)
				vs.append(v)

			if vector:
				energy = specific_kinetic_energy(np.linalg.norm(v)) - specific_potential_energy(np.linalg.norm(r))
				energies.append(energy)

			else:
				energy = specific_kinetic_energy(v) - specific_potential_energy(r)
				energies.append(energy)

		return np.rad2deg(thetas), Es, np.rad2deg(Ms), rs, vs, energies

	Fs = []

	for i in range(len(times)):
		M = n*(times[i] - tp)
		Ms.append(M)

		# solve for F
		F = F_newton(M, e)
		Fs.append(F)

		# solve for theta
		theta = theta_from_F(F, e)
		thetas.append(theta)

		# solve for r
		if vector:
			r, v = kep2cart_Fife(a, e, i, RAAN, argp, theta, mu=mu)
			rs.append(r)
			vs.append(v)

		else:
			r = r_orbit_eq(theta, h_mag=h_mag, e_mag=e)
			rs.append(r)

			# solve for v
			v = speed_from_r(r, a)
			vs.append(v)

		if vector:
			energy = specific_kinetic_energy(np.linalg.norm(v)) - specific_potential_energy(np.linalg.norm(r))
			energies.append(energy)

		else:
			energy = specific_kinetic_energy(v) - specific_potential_energy(r)
			energies.append(energy)

	return np.rad2deg(thetas), Fs, Ms, rs, vs, energies


def two_body_dynamics(t, x, j2, mu=398600.4415):
	"""
	Two body dynamics function used as input for numerical integrator.

	if J2 is not None, pertubation of J2 is taken into account.
	"""
	r = x[:3]
	v = x[3:6]
	r_mag = np.linalg.norm(r)
	if j2 != 0.:
		C1   = 1 - (3*j2/2.)*((R_E/r_mag)**2)*(5*(r[2]/r_mag)**2 - 1)
		C2   = 1 - (3*j2/2.)*((R_E/r_mag)**2)*(5*(r[2]/r_mag)**2 - 3)
		a    = np.zeros(3,)
		a[0] = (-mu/r_mag**3)*C1*r[0]
		a[1] = (-mu/r_mag**3)*C1*r[1]
		a[2] = (-mu/r_mag**3)*C2*r[2]

		xdot = np.append(v, a)

		return xdot

	a    = (-mu/r_mag**3)*r
	xdot = np.append(v, a)

	return xdot


def r_v_prop(r0, v0, t, dt, j2=0., mu=398600.4415):
	# Create initial state
	x0 = np.append(r0, v0)

	# Set integrator and initial conditions
	rv = ode(two_body_dynamics)
	rv.set_integrator('dopri5', rtol=1e-12, atol=1e-20)
	rv.set_initial_value(x0, t[0])
	rv.set_f_params(j2)


	# Define output array
	eps  = []
	eps0 = specific_kinetic_energy(np.linalg.norm(v0)) - specific_potential_energy(np.linalg.norm(r0))
	eps.append(0.)
	out  = []
	out.append(x0)
	acc  = []
	acc.append(two_body_dynamics(t[0], x0, j2))

	orb  = []
	orb.append(cart2kep_Fife(r0, v0, mu=mu))

	# Start propagation
	while rv.successful() and rv.t < t[-1]:
		acc.append(two_body_dynamics(rv.t, rv.y, j2))
		rv.integrate(rv.t + dt)
		eps.append(specific_kinetic_energy(np.linalg.norm(rv.y[3:6])) - specific_potential_energy(np.linalg.norm(rv.y[:3])) - eps0)
		orb.append(cart2kep_Fife(rv.y[:3], rv.y[3:6], mu=mu))
		out.append(rv.y)

	# list comprehend the output
	rs   = [x[:3] for x in out]
	vs   = [x[3:6] for x in out]
	accs = [np.linalg.norm(x[3:6]) for x in acc]
	a    = [x[0] for x in orb]
	e    = [x[1] for x in orb]	
	i    = [x[2] for x in orb]
	RAAN = [x[3] for x in orb]
	argp = [x[4] for x in orb]

	return rs, np.asarray(eps), vs


def plot_energy_change(times, delta_eps, ylim_min, ylim_max):
	plt.figure('Energy Difference')
	plt.title(r'$\Delta \epsilon$' + ' vs Time')
	plt.ylabel(r'$\Delta \epsilon$' + ' (km/s)^2')
	plt.xlabel('Time (min)')
	plt.ylim(ylim_min, ylim_max)
	plt.plot(times, delta_eps, color='black')
	plt.grid()
	plt.show()