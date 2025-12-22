import numpy as np



from astropy.convolution import convolve_fft
from astropy.convolution import Gaussian1DKernel

def rebin_spectrum(wvls, flux, R, w_sample=1):
    '''
    Smears a model spectrum with a gaussian kernel to the given resolution, R.

    Parameters
    -----------

    sp: SourceSpectrum
        Pysynphot object that we willsmear

    R: int
        The resolution (dL/L) to smear to

    w_sample: int
        Oversampling factor for smoothing

    Returns
    -----------

    sp: PySynphot Source Spectrum
        The smeared spectrum
    '''

    # Generate logarithmic wavelength grid for smoothing
    w_logmin = np.log10(np.nanmin(wvls))
    w_logmax = np.log10(np.nanmax(wvls))
    n_w = np.size(wvls)*w_sample
    w_log = np.logspace(w_logmin, w_logmax, num=n_w)

    # Find stddev of Gaussian kernel for smoothing
    R_grid = (w_log[1:-1]+w_log[0:-2])/(w_log[1:-1]-w_log[0:-2])/2
    sigma = np.median(R_grid)/R
    if sigma < 1:
        sigma = 1

    # Interpolate on logarithmic grid
    f_log = np.interp(w_log, wvls, flux)

    # Smooth convolving with Gaussian kernel
    gauss = Gaussian1DKernel(stddev=sigma)
    f_conv = convolve_fft(f_log, gauss)

    # Interpolate back on original wavelength grid
    f_sm = np.interp(wvls, w_log, f_conv)

    # Write smoothed spectrum back into Spectrum object
    return wvls, f_sm