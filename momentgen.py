# J. W. Thiesing 2025/09/17

import numpy as np

# Ameya's
def ccf(X1, X2=None, lag=0):
	if X2 is None:
		X2 = X1 # acf
	if X1.shape != X2.shape:
		raise ValueError("Two array shapes not equal.")

	shape = X1.shape
	axis_to_avg = len(shape)-1

	padding = np.zeros(shape=tuple(
		(shape[i] if i < axis_to_avg else np.abs(lag))
		for i in range(len(shape))))

	if lag >= 0:
		prod = X1[lag:] * np.conjugate(X2[:(None if lag == 0 else -lag)])
	else:
		prod = X1[:lag] * np.conjugate(X2[-lag:])

	return np.mean(np.concatenate([prod, padding], axis=axis_to_avg), axis=axis_to_avg)

"""def ccf(x1, x2, l):
	summ = 0 + 0j
	M = x1.size
	if not x1.size == x2.size:
		print("Timeseries sizes different")
	for m in range(0, M-np.abs(l)):
		summ = summ + np.dot(x1[m],np.conj(x2[m-l]))
	return summ/(M-np.abs(l)-1)"""

def acf(x, l):
	return ccf(x, x, l)

def get_moments(X_ho, X_vo, N_h, N_v, R, va, C, Cd, Cp):
	# X_h and X_v have dims (range, ray, pulse)

	# C = C[0][0]
	# Cd = Cd[0]
	# Cp = Cp[0]

	if not np.isfinite(N_h) or N_h < 0:
		N_h = 0
	if not np.isfinite(N_v) or N_v < 0:
		N_v = 0

	C = np.nanmean(C[:,0])
	Cd = np.nanmean(Cd)
	Cp = np.nanmean(Cp)

	moments = {'DBZ': None, 'VEL': None, 'WIDTH': None, 'ZDR': None, 'RHOHV': None, 'PHIDP': None, 'SNRH': None, 'SNRV': None}

	DBZOUT = np.full((X_ho.shape[1], X_ho.shape[0]), np.nan)
	VELOUT, WIDTHOUT, ZDROUT, RHOHVOUT, PHIDPOUT, SNRHOUT, SNRVOUT = DBZOUT.copy(), DBZOUT.copy(), DBZOUT.copy(), DBZOUT.copy(), DBZOUT.copy(), DBZOUT.copy(), DBZOUT.copy()

	for it in range(X_ho.shape[1]):
		for ir in range(X_ho.shape[0]):
			X_h = X_ho[ir,it]
			X_v = X_vo[ir,it]
			r = R[ir]

			P_h = np.real(acf(X_h, 0)) # Rxx(V,0)
			P_v = np.real(acf(X_v, 0))
			S_h = P_h - N_h
			S_v = P_v - N_v

			cross = ccf(X_h, X_v, 0)
			lag1h = acf(X_h,1)
			# print(P_h, N_h, P_v, N_v)

			DBZOUT[it,ir] = 10*np.log10(S_h) + 20*np.log10(r if not r==0 else 1e-10) + 10*np.log10(C if not C==0 else 1e-10)
			VELOUT[it,ir] = (va/np.pi)*np.angle(lag1h)
			WIDTHOUT[it,ir] = (np.sqrt(2)*va/np.pi)*np.sqrt(np.abs(np.log(S_h/np.abs(lag1h))))
			ZDROUT[it,ir] = 10*np.log10(S_h/S_v) # + 10*np.log10(Cd)
			RHOHVOUT[it,ir] = np.abs(cross)/np.sqrt(S_h*S_v)
			PHIDPOUT[it,ir] = np.rad2deg(np.angle(cross)) + Cp # np.atan2(np.imag(S_h),np.real(S_h))-np.atan2(np.imag(S_v),np.real(S_v)) + Cp
			SNRHOUT[it,ir] = 10*np.log10(S_h/N_h)
			SNRVOUT[it,ir] = 10*np.log10(S_v/N_v)

	PHIDPOUT[PHIDPOUT < -180] += 360
	PHIDPOUT[PHIDPOUT > 180] -= 360

	moments['DBZ'] = DBZOUT
	moments['VEL'] = VELOUT
	moments['WIDTH'] = WIDTHOUT
	moments['ZDR'] = ZDROUT
	moments['RHOHV'] = RHOHVOUT
	moments['PHIDP'] = PHIDPOUT
	moments['SNRH'] = SNRHOUT
	moments['SNRV'] = SNRVOUT

	return moments