from scipy.linalg import solve, toeplitz
try:
    from scipy.linalg import solve_toeplitz
    can_solve_toeplitz = True
except ImportError:
    can_solve_toeplitz = False
import numpy as np
import math


def lorentzian(broad, root, osc_str, freq):
    """
    Calculates and returns a lorentzian

    The lorentzian is centered at root, integrates to osc_str, and has a
    half-width at half-max of broad.
    """
    ones = np.ones(freq.shape)
    # 1/(pi*broad*(1+((w-root)/broad)^2))
    l_denom = broad * np.pi * (1 + np.square((freq - root * ones) / broad))
    return osc_str * np.divide(ones, l_denom)


def lorentzian_2(broad, root, osc_str, freq_matrix):
    lorentz = 1 / (broad * np.pi * (1 + np.square((freq_matrix - root) / broad)))
    return np.dot(lorentz, osc_str)


def sigmoidal(broad, root, osc_str, freq_matrix):
    sigmoid = 1 / (1 + np.exp(-broad * (freq_matrix - root)))
    return np.dot(sigmoid, osc_str)


def gaussian_2(broad, root, osc_str, freq_matrix):
    stddev = broad / np.sqrt(2.0 * np.log(2.0))
    frac = 1 / (np.sqrt(2 * np.pi) * stddev)
    gauss = frac * np.exp(-1 * np.square(freq_matrix - root) / (2 * np.square(stddev)))
    return np.dot(gauss, osc_str)


def gaussian(broad, root, osc_str, freq):
    """
    Calculates and returns a gaussian

    The gaussian is centered at root, integrates to osc_str, and has a
    half-width at half-max of broad.
    """
    ones = np.ones(freq.shape)
    # Convert from HWHM to std dev
    stddev = broad / np.sqrt(2.0 * np.log(2.0))
    # 1/((2*pi*broad^2)^(1/2))*e^(-(w-root)^2/(2*broad^2)
    g_power = -1 * np.square(freq - root * ones) / (2 * np.square(stddev))
    gauss = 1 / (np.sqrt(2 * np.pi) * stddev) * np.exp(g_power)
    return osc_str * gauss





def fourier_tx(data, dt, res=3000, **kwargs):
    """
    Easy fourier transform wrapper to numpy.fft. Assumes signal is real.

    Parameters
    ----------
    data : numpy.NDArray
        Should be a vector or array whose first dimension is the time series.
    dt
        Time step size

    Returns
    -------
    tuple of (numpy.NDArray, numpy.NDArray)
        Contains the frequency, then the fourier transform 

    Notes
    -----
    Takes arbitrary kwargs to provide the same call signature as pade_tx. These
    have no effect on the calculation.
    """

    # Compatibility with call signature of pade_tx
    for kw in ['wlim', 'do_toeplitz']:
        try:
            kwargs.pop(kw)
        except KeyError:
            pass

    if len(kwargs) > 0:
        raise TypeError(
            "fourier_tx() got an unexpected keyword argument " +
            "'{}'".format(kwargs.keys()[0]))
    
    curres = 2*np.pi/(dt*len(data))
    if res > curres:
        if np.abs(data[-1])/max(data) > 1e-7:
            print("WARNING: increasing resolution in a FFT on a not-fully " +
                  "damped signal can create artifacts.")
        npts = int(2*np.pi/dt*res)
        print(npts)
        data = np.hstack([data, np.zeros((npts-len(data),))])

    w = np.fft.rfftfreq(len(data))/dt*2*np.pi
    spec = -1 * np.fft.rfft(data)

    return (w,spec)


def pade_tx(data, dt, wlim=(0,2), res=3000, do_toeplitz=True):
    """
    Gives the Pade approximant to the fourier transform of the data at a given
    range of frequencies.

    Parameters
    ----------
    data : numpy.NDArray
        Data to be transformed. Should be a vector containing time series data.
    dt
        Time step
    wlim : tuple, optional
        Frequency range over which to transform in the format (start, stop).
    res : integer, optional
        Resolution in points/(freq. unit).
    do_toeplitz : bool, optional
        Whether to do `scipy.linalg.solve_toeplitz` or general linear solve. 

    Returns
    -------
    tuple of (numpy.NDArray, numpy.NDArray)
        Contains the frequency, then the fourier transform 

    Notes
    -----
    A description of the Pade transform used here can be found in [1]_

    References
    ----------
    .. [1] Bruner, Adam, Daniel LaMaster, and Kenneth Lopata. "Accelerated
       broadband spectra using transition dipole decomposition and Pade
       approximants." Journal of chemical theory and computation 12.8 (2016):
       3741-3750
    """


    if do_toeplitz and not can_solve_toeplitz:
        print("SciPy < 0.17.0 does not have 'linalg.solve_toeplitz'")
        print("Falling back to general linear solve.")
        do_toeplitz = False
    
    M = len(data)
    N = M // 2

    # d_k = -c_{N+k}, k \in [1,N]
    d = -data[N+1:2*N]

    if not do_toeplitz:
        # G_{k,m} = c_{N-m+k}, m,k \in [1,N]
        G = data[N + np.arange(1,N)[:,None] - np.arange(1,N)]
 
        # solve b = G^{-1} d
        b = solve(G, d, check_finite=False)

    else:
        # Because G is toeplitz, we can solve using just the first column and
        # row (Levinson recursion)
        c = data[N:2*N-1]  # Column
        r = np.hstack([data[1], data[N-1:1:-1]])  # Row
        b = solve_toeplitz((c,r), d, check_finite=False)

    # Assert that b0 = 1
    b = np.hstack([1, b])
    # a_k = \sum_{m=0}^k b_m c_{k-m}
    a = np.dot(np.tril(toeplitz(data[0:N])),b)

    # Form \sum_{k=0}^M a_k z^k (and b_k)
    p = np.poly1d(a)
    q = np.poly1d(b)

    # Get complex coordinates
    w = np.linspace(wlim[0], wlim[1], math.ceil(res*(wlim[1]-wlim[0])))
    z = np.exp(-1j*w*dt)

    # Plug in z's and evaluate Pade approximant
    spec = p(z)/q(z)

    return (w, spec)
