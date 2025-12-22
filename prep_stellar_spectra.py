"""
    This script use stsynphot to generate stellar SED and save as SOCRATES version
    https://synphot.readthedocs.io/en/latest/index.html
    phoenix model grid: http://ssb.stsci.edu/trds/tarfiles/synphot5.tar.gz
    sphinx model grid: https://zenodo.org/records/11392341
"""

import numpy as np
import os
import matplotlib.pyplot as plt

import astropy.constants as aconst
from synphot.utils import download_data
from astropy.config import generate_config
# find the path of this py file
root = os.path.dirname(os.path.abspath(__file__))
os.environ["PYSYN_CDBS"] = os.path.join(root,"stellar_spectra/synphot_trds")
import stsynphot as S
from astropy import units as u
from util.sphinx_icat import grid_to_spec_sphinx
from util.tools import rebin_spectrum
from typing import Literal, Tuple, List
# generate_config(pkgname='synphot')
# file_list = download_data('/work/home/ac9b0k6rio/SocSpecGen/stellar_spectra/synphot_trds')



def write_to_socrates_input(output_filename,star_name,wavelen_m,irradiance_wm3,model_type:Literal['phoenix','sphinx1','sphinx2-condcloud','sphinx2-graycloud','sphinx2-graycloud_MLT0.5']):
    # start here
    """
        wavelen_m: wavelength in meter
        irradiance_wm3: irradiance in W/m3
    """
    with open(output_filename, 'w') as output_file:
        # write header
        output_file.write(f'SOLAR SPECTRUM OF {star_name} ({model_type} interpolate)\n')
        output_file.write('      WAVELENGTH        IRRADIANCE\n')
        output_file.write('          (m)               (W/m3)\n')
        output_file.write('*BEGIN_DATA\n')
        for wavelen, irradiance in zip(wavelen_m, irradiance_wm3):
            output_file.write(f'      {wavelen:.16e}      {irradiance:.16e}\n')
        output_file.write('*END\n')
        
def interp_sed(Teff,MH,logg_cgs,model_type:Literal['phoenix','sphinx1','sphinx2-condcloud','sphinx2-graycloud','sphinx2-graycloud_MLT0.5'],c_to_o=0.5):
    if model_type == 'phoenix':
        sp = S.catalog.grid_to_spec('phoenix', Teff,MH,logg_cgs)    
    elif model_type == 'sphinx1':
        sp = grid_to_spec_sphinx(Teff, MH, logg_cgs, c_to_o, sphinx_dir = os.path.join(os.environ['PYSYN_CDBS'], 'grid', 'sphinx-I'))
    elif model_type == 'sphinx2-condcloud':
        sp = grid_to_spec_sphinx(Teff, MH, logg_cgs, c_to_o, sphinx_dir = os.path.join(os.environ['PYSYN_CDBS'], 'grid', 'sphinx-II','SPHINXII_condcloud','SPECTRA'))
    elif model_type == 'sphinx2-graycloud':
        sp = grid_to_spec_sphinx(Teff, MH, logg_cgs, c_to_o, sphinx_dir = os.path.join(os.environ['PYSYN_CDBS'], 'grid', 'sphinx-II','SPHINXII_graycloud','SPECTRA'))
    elif model_type == 'sphinx2-graycloud_MLT0.5':
        sp = grid_to_spec_sphinx(Teff, MH, logg_cgs, c_to_o, sphinx_dir = os.path.join(os.environ['PYSYN_CDBS'], 'grid', 'sphinx-II','SPHINXII_graycloud_MLT0.5','SPECTRA'))
    else:
        raise ValueError('model_type must be phoenix or sphinx')
    wave_ang = sp.waveset
    flux_flam = sp(wave_ang, flux_unit='flam')
    wavelen_m = wave_ang.value * 1e-10
    irradiance_wm3 = flux_flam.value * 1e7
    return wavelen_m, irradiance_wm3


stellar_const = {
    'GJ3929': {'Teff': 3384, 'MH': -0.02, 'stellar_mass_to_solar': 0.313, 'stellar_radius_to_solar': 0.32},
    'LTT1445A': {'Teff': 3340, 'MH': -0.34, 'stellar_mass_to_solar': 0.257, 'stellar_radius_to_solar': 0.265},
    'LHS1140': {'Teff': 3096, 'MH': -0.15, 'stellar_mass_to_solar': 0.1844, 'stellar_radius_to_solar': 0.2159},
    'TOI198': {'Teff': 3650, 'MH': -0.7, 'stellar_mass_to_solar': 0.467, 'stellar_radius_to_solar': 0.441},
    'TOI406': {'Teff': 3392, 'MH': -0.02, 'stellar_mass_to_solar': 0.408, 'stellar_radius_to_solar': 0.41},
    'TOI771': {'Teff': 3370, 'MH': -0.13, 'stellar_mass_to_solar': 0.22, 'stellar_radius_to_solar': 0.232},
    'HD260655': {'Teff': 3803, 'MH': -0.43, 'stellar_mass_to_solar': 0.439, 'stellar_radius_to_solar': 0.439},
    'TOI244': {'Teff': 3433, 'MH': -0.39, 'stellar_mass_to_solar': 0.427, 'stellar_radius_to_solar': 0.428},
}

g_sun_cgs = aconst.GM_sun.value/aconst.R_sun.value**2 * 1e2 # [cm/s^2]
for star in ['GJ3929', 'LTT1445A', 'LHS1140', 'TOI198', 'TOI406', 'TOI771', 'HD260655', 'TOI244']:
    params = stellar_const[star]
    Teff = params['Teff']
    MH = params['MH']
    stellar_mass_to_solar = params['stellar_mass_to_solar']
    stellar_radius_to_solar = params['stellar_radius_to_solar']
    logg_cgs = np.log10(stellar_mass_to_solar/stellar_radius_to_solar**2*g_sun_cgs)

    # phoenix model
    # wavelen_m, irradiance_wm3 = interp_sed(Teff, MH, logg_cgs, model_type='phoenix')
    # output_filename = f'stellar_spectra/{star}_phoenix.socrates'
    # write_to_socrates_input(output_filename, star, wavelen_m, irradiance_wm3, model_type='phoenix')

    # sphinx model
    # target_ctoo = 0.5 # unknown, set to Sun-like
    # wavelen_m, irradiance_wm3 = interp_sed(Teff, MH, logg_cgs, model_type='sphinx1', c_to_o=target_ctoo)
    # output_filename = os.path.join(root,f'stellar_spectra/soc_in/{star}_sphinx')
    # write_to_socrates_input(output_filename, star, wavelen_m, irradiance_wm3, model_type='sphinx1')

### a comparison here for test
# GJ-3929 parameters
Teff = 3384; MH = -0.02; stellar_mass_to_solar = 0.313; stellar_radius_to_solar = 0.32
g_sun_cgs = aconst.GM_sun.value/aconst.R_sun.value**2 * 1e2 # [cm/s^2]
logg_cgs = np.log10(stellar_mass_to_solar/stellar_radius_to_solar**2*g_sun_cgs)

fig,ax = plt.subplots(1,1,figsize=(6,4))
target_R = 1000
# generate phoenix spectrum
sp = S.catalog.grid_to_spec('phoenix', Teff,MH,logg_cgs)
wave_angstrom = sp.waveset * u.Angstrom
flux_flam = sp(sp.waveset, flux_unit='flam') * u.erg / (u.s * u.cm**2 * u.Angstrom)
wave_values = wave_angstrom.value/1e4 # in micron
flux_values = flux_flam.value* 1e7
wave_values, flux_values = rebin_spectrum(wave_values, flux_values, target_R)
ax.plot(wave_values, flux_values, label=f'PHOENIX')

# generate sphinx spectrum
target_ctoo = 0.5 # unknown, set to Sun-like
wavelen_m, irradiance_wm3 = interp_sed(Teff, MH, logg_cgs, model_type='sphinx1', c_to_o=target_ctoo)
ax.plot(wavelen_m*1e6, irradiance_wm3, label=f'SPHINX-I')

wavelen_m, irradiance_wm3 = interp_sed(Teff, MH, logg_cgs, model_type='sphinx2-condcloud', c_to_o=target_ctoo)
ax.plot(wavelen_m*1e6, irradiance_wm3, label=f'SPHINX-II-condcloud')

wavelen_m, irradiance_wm3 = interp_sed(Teff, MH, logg_cgs, model_type='sphinx2-graycloud', c_to_o=target_ctoo)
ax.plot(wavelen_m*1e6, irradiance_wm3, label=f'SPHINX-II-graycloud')

wavelen_m, irradiance_wm3 = interp_sed(Teff, MH, logg_cgs, model_type='sphinx2-graycloud_MLT0.5', c_to_o=target_ctoo)
ax.plot(wavelen_m*1e6, irradiance_wm3, label=f'SPHINX-II-graycloud_MLT0.5')

ax.set_xlim(0, 5.0)
ax.set_xlabel('Wavelength (micron)')
ax.set_ylabel(r'Flux (W/m3)')
ax.legend()
plt.savefig('GJ3929_stellar_spectra_comparison.png')