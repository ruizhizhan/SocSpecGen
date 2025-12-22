import numpy as np
import os

# ==========================================
# 1. Environment & Basic Configuration
# ==========================================
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
SLURM_DIR = os.path.join(ROOT_DIR, 'slurms')
NT = '62'
NP = '22'
RES = '001'

# ==========================================
# 2. Gas Library (Absorption & UV only)
# ==========================================
# Keys are molecule names. Values contain IDs, HDF5 paths, and UV settings.
GAS_LIBRARY = {
    'CO2': {
        'molecule': '12C-16O2',
        'gas_id': '2',
        'gas_abs_config': {
            'hdf5_rel_path': f'ExoMol/hdf5/CO2/CO2_UCL-4000_1.0-20000.0_T{NT}xP{NP}_{RES}.h5',
            'datasource': f'CO2_T{NT}xP{NP}_{RES}',
            'lower_wn': 1.0,
            'upper_wn': 20000.0,
        },
        'uv_config': {
            # 'pt_file_name': 'pt_co2_uv',
            'xuv_rel_path': 'ExoMol/12C_16O2/XUV/',
            'xuv_file': 'CO2_DTU.uvxsc',
            'lower_wn': 1e7/280, # approx 35000 cm-1
            'upper_wn': 1e7/108, # approx 100000 cm-1
            't_grid': [305.0,550.0,800.0,1160.0,1630.0],
            'p_grid': [1e-6,1e-5,1e-4,1e-3], # in bar, from 0.1 Pa to 100 Pa
        }
    },
    'CO': {
        'molecule': '12C-16O',
        'gas_id': '5',
        'gas_abs_config': {
            'hdf5_rel_path': f'ExoMol/hdf5/CO/CO_Li2015_1.0-23000.0_T{NT}xP{NP}_{RES}.h5',
            'datasource': f'CO_T{NT}xP{NP}_{RES}',
            'lower_wn': 1.0,
            'upper_wn': 23000.0,
        },
        'uv_config': {
            # 'pt_file_name': 'pt_co_uv',
            'xuv_rel_path': 'ExoMol/12C_16O/XUV/',
            'xuv_file': 'CO_DTU.uvxsc',
            'lower_wn': 43706,
            'upper_wn': 85441,
            't_grid': [305.0,1630.0],
            'p_grid': [1e-6,1e-5,1e-4,1e-3], # in bar, from 0.1 Pa to 100 Pa
        }
    },
    'N2': {
        'molecule': '14N2',#
        'gas_id': '13',
        'gas_abs_config': {
            'hdf5_rel_path': f'ExoMol/hdf5/N2/N2_WCCRMT_1.0-38200.0_T{NT}xP{NP}_{RES}.h5',
            'datasource': f'N2_T{NT}xP{NP}_{RES}',
            'lower_wn': 1.0,
            'upper_wn': 38200.0,
        },
        'uv_config': { # ExoMol/14N2/XUV/N2_combine.uvxsc, 1-226nm
            #'pt_file_name': 'pt_n2_uv',
            'xuv_rel_path': 'ExoMol/14N2/XUV/',
            'xuv_file': 'N2_combine.uvxsc',
            'lower_wn': 1e7/226, # approx 44248 cm-1
            'upper_wn': 1e7/1,   # approx 10000000 cm-1
            't_grid': [295.0,298.0],
            'p_grid': [1e-6,1e-5,1e-4,1e-3], # in bar, from 0.1 Pa to 100 Pa
        }
    },
    'H2O': {
        'molecule': '1H2_16O',#
        'gas_id': '1',
        'gas_abs_config': {
            'hdf5_rel_path': f'ExoMol/hdf5/H2O/H2O_POZAKATEL_1.0-41200.0_T{NT}xP{NP}_{RES}.h5',
            'datasource': f'H2O_T{NT}xP{NP}_{RES}',
            'lower_wn': 1.0,
            'upper_wn': 41200.0,
        },
        'uv_config': { # ExoMol/1H2_16O/XUV/H2O.uvxsc, 108-237 nm
            'pt_file_name': 'pt_h2o_uv',
            'xuv_rel_path': 'ExoMol/1H2_16O/XUV/',
            'xuv_file': 'H2O.uvxsc',
            'lower_wn': 1e7/237, # approx 42194 cm-1
            'upper_wn': 1e7/108, # approx 92592 cm-1
            't_grid': [423.15,573.15,1630.0,1773.15],
            'p_grid': [1e-6,1e-5,1e-4,1e-3], # in bar, from 0.1 Pa to 100 Pa
        }
    },
    'SiO': {
        'molecule': '28Si_16O',
        'gas_id': '17',
        'gas_abs_config': {
            'hdf5_rel_path': 'ExoMol/hdf5/SiO/SiO_UVenIR_1.0-66500.0_T62xP22_001.h5',
            'datasource': 'SiO_T62xP22_001',
            'lower_wn': 1.0,
            'upper_wn': 66500.0,
        },
        'uv_config': None
    },
    # Add other gases (H2O, CH4, etc.) here...
}

# ==========================================
# 3. CIA Library (Independent)
# ==========================================
# Keys are the pair names (Must match structure MolA-MolB).
# Used to determine if a CIA file should be included based on selected gases.
CIA_LIBRARY = { # single pressure grid for each CIA
    'CO2-CO2': {
        'cia_rel_path': 'hitran/CO2-CO2_2024/',# 'hitran/CO2-CO2_2018/',
        'cia_file': 'CO2-CO2_2024.cia',# 'CO2-CO2_2018.cia',
        'lower_wn': [1,1120,2510,4000],
        'upper_wn': [750,1850,3250,4500],
        't_grid': np.arange(200, 801, 100), 
        'p_grid': [1.0]
    },
    'N2-N2':{ # hitran/N2-N2_2021/N2-N2_2021.cia; 70K-400K
        'cia_rel_path': 'hitran/N2-N2_2021/',
        'cia_file': 'N2-N2_2021.cia',
        'lower_wn': [1.0],
        'upper_wn': [5000.],
        't_grid': [70,100,150,200,250,300,350,400], 
        'p_grid': [1.0]
    },
    'CO2-H2O':{
        'cia_rel_path': 'hitran/CO2-H2O_2024',
        'cia_file': 'CO2-H2O_2024.cia',
        'lower_wn': [5700.],
        'upper_wn': [6300.],
        't_grid': [296], 
        'p_grid': [1.0]
    },
    'N2-H2O':{
        'cia_rel_path': 'hitran/N2-H2O_2018',
        'cia_file': 'N2-H2O_2018.cia',
        'lower_wn': [1930.],
        'upper_wn': [2830.],
        't_grid': [250,300,350],
        'p_grid': [1.0]
    }
}

# ==========================================
# 4. Simulation Parameters
# ==========================================
STAR_NAME = '55CancriA'
NUM_KTERM = 20
INCLUDE_CIA = True
INCLUDE_SOLAR_SED = True

# New Option: Ultra Hot Atmosphere (Calculate UV/Rayleigh for 'lw' if True)
ULTRA_HOT_ATMOSPHERE = True 

# ==========================================
# 5. Wavenumber Edges Generation
# ==========================================
def get_wnedges():
    # 1-25 um: delta=1 um; 25-100 um: delta=5um; 0-1um: use wavenumber
    _wnedges = np.sort(np.concatenate([np.ones(1),
                                      np.array(1e4/np.arange(100,25,-5)),
                                      np.array(1e4/np.arange(25,1,-1)),
                                      np.arange(10000,20000,400),
                                      np.arange(20000,40000,1000),
                                      np.arange(40000,100000,5000)]))
    _wnedges = np.unique(np.round(_wnedges, 10))
    return _wnedges

WNEDGES = get_wnedges()