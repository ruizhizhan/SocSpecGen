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
# choose the correct hdf5 file (different broadening, ...)
MOLEULES_TO_INCLUDE = ['CO2','O2']
STAR_NAME = 'K7_5V_phoenix' 
# K7_5V_phoenix M2_5V_sphinx M8_5V_sphinx LHS3844_sphinx LTT1445A_sphinx Trappist-1_sphinx GJ486_sphinx GJ1132_sphinx GJ3929_sphinx 
TEST_NAME = f'{STAR_NAME}_CO2_O2'

# ==========================================
# 2. Gas Library (Absorption & UV only)
# ==========================================
# Keys are molecule names. Values contain IDs, HDF5 paths, and UV settings.
# The "Ill-conditioned division in Newton-Raphson iteration" error is fundamentally triggered by 
# unphysical non-positive values entering the optical depth calculations, 
# stemming either from negative cross-sections in your CIA (Collision-Induced Absorption) data 
# or from $0/0$ NaNs caused by Planck function underflow at short wavelengths and extreme low temperatures.
GAS_LIBRARY = {
    'CO2': {
        'molecule': '12C-16O2',
        'gas_id': '2',
        'gas_abs_config': {
            'hdf5_rel_path': 'ExoMol/hdf5/CO2/CO2_Dozen_1.0-20000.0_T62xP22_001.h5',
            #f'ExoMol/hdf5/CO2/CO2_UCL-4000_1.0-20000.0_T{NT}xP{NP}_{RES}.h5', # this is too old
            'datasource': f'CO2_T{NT}xP{NP}_{RES}',
            'lower_wn': 1.0,
            'upper_wn': 20000.0,
        },
        'uv_config': {
            'xuv_rel_path': 'ExoMol/12C_16O2/XUV/',
            'xuv_file': 'CO2_DTU.uvxsc',
            'lower_wn': 1e7/280, # approx 35000 cm-1
            'upper_wn': 1e7/119, # approx 100000 cm-1
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
            'xuv_rel_path': 'ExoMol/14N2/XUV/',
            'xuv_file': 'N2_combine.uvxsc',
            'lower_wn': 1e7/226, # approx 44248 cm-1
            'upper_wn': 1e7/1,   # approx 10000000 cm-1
            't_grid': [295.0,298.0],
            'p_grid': [1e-6,1e-5,1e-4,1e-3], # in bar, from 0.1 Pa to 100 Pa
        }
    },
    'H2O': {
        'molecule': '1H2_16O', 
        'gas_id': '1', # cfc113 in GCM!
        'gas_abs_config': {
            'hdf5_rel_path': f'ExoMol/hdf5/H2O/H2O_POZAKATEL_1.0-41200.0_T{NT}xP{NP}_{RES}.h5',
            'datasource': f'H2O_T{NT}xP{NP}_{RES}',
            'lower_wn': 1.0,
            'upper_wn': 41200.0,
        },
        'uv_config': { # ExoMol/1H2_16O/XUV/H2O.uvxsc, 108-237 nm
            'xuv_rel_path': 'ExoMol/1H2_16O/XUV/',
            'xuv_file': 'H2O.uvxsc',
            'lower_wn': 1e7/237, # approx 42194 cm-1
            'upper_wn': 1e7/108, # approx 92592 cm-1
            't_grid': [423.15,573.15,1630.0,1773.15],
            'p_grid': [1e-6,1e-5,1e-4,1e-3], # in bar, from 0.1 Pa to 100 Pa
        }
    },
    'O2': {
        'molecule': '16O2',
        'gas_id': '7',
        'gas_abs_config': {
            'hdf5_rel_path': 'ExoMol/hdf5/O2/O2_SWYT_1.0-20000.0_T62xP22_001.h5',
            'datasource': 'O2_T62xP22_001',
            'lower_wn': 1.0,
            'upper_wn': 20000.0,
        },
        'uv_config': { # ExoMol/16O2/PhoMol/O2.uvxsc
            'xuv_rel_path': 'ExoMol/16O2/PhoMol/',
            'xuv_file': 'O2.uvxsc',
            'lower_wn': 1e7/500, # approx 20000 cm-1
            'upper_wn': 1e7/50, # approx 200000 cm-1
            't_grid': np.arange(200,4201,800), # uv no need to be such high res
            'p_grid': [1e-6,1e-5,1e-4,1e-3], # in bar, from 0.1 Pa to 100 Pa
        }
    },
    'CH4': {},
    'H2': {},
    'N2O': {},
    'NH3': {},
    'SO2': {
        'molecule': '32S_16O2',
        'gas_id': '9',
        'gas_abs_config': {
            'hdf5_rel_path': 'ExoMol/hdf5/SO2/SO2_ExoAmes_H2broad_1.0-8000.0_T37xP22_001.h5',
            'datasource': 'SO2_T37xP22_001',
            'lower_wn': 1.0,
            'upper_wn': 8000.0,
        },
        'uv_config': { # ExoMol/16O2/PhoMol/O2.uvxsc
            'xuv_rel_path': 'ExoMol/32S_16O2/XUV/',
            'xuv_file': 'SO2.uvxsc',
            'lower_wn': 1e7/230, # approx 20000 cm-1
            'upper_wn': 1e7/110, # approx 200000 cm-1
            't_grid': [423], # uv no need to be such high res
            'p_grid': [1e-6,1e-5,1e-4,1e-3], # in bar, from 0.1 Pa to 100 Pa
        }
    },
    'H2S': {},
    'HCl': {
        'uv_config': { 
            'xuv_rel_path': 'ExoMol/1H_35Cl/PTY/',
            'xuv_file': 'HCl.uvxsc',
            'lower_wn': 1e7/400, # approx 25000 cm-1
            'upper_wn': 1e7/100, # approx 100000 cm-1
            't_grid': np.arange(200,4201,800), # uv no need to be such high res
            'p_grid': [1e-6,1e-5,1e-4,1e-3], # in bar, from 0.1 Pa to 100 Pa
        }
    },
    # rock and metal vapor below
    'SiO': {
        'molecule': '28Si_16O',
        'gas_id': '17', # hcfc22
        'gas_abs_config': {
            'hdf5_rel_path': 'ExoMol/hdf5/SiO/SiO_UVenIR_CO2broad_1.0-66500.0_T62xP22_001.h5',
            'datasource': 'SiO_T62xP22_001',
            'lower_wn': 1.0,
            'upper_wn': 66500.0,
        },
        'uv_config': None
    },
    'SiO2': {
        'molecule': '28Si_16O2',
        'gas_id': '18', # hfc125
        'gas_abs_config': {
            'hdf5_rel_path': 'ExoMol/hdf5/SiO2/SiO2_OYT3_CO2broad_1.0-6000.0_T47xP22_001.h5',
            # 'ExoMol/hdf5/SiO2/SiO2_OYT3_1.0-6000.0_T47xP22_001.h5', # self broad
            'datasource': 'SiO2_T47xP22_001',
            'lower_wn': 1.0,
            'upper_wn': 6000.0,
        },
        'uv_config': None
    },
    'MgO': {
        'molecule': '24Mg_16O',
        'gas_id': '18', # hfc125
        'gas_abs_config': {
            'hdf5_rel_path': 'ExoMol/hdf5/MgO/MgO_LiTY_1.0-30000.0_T62xP22_001.h5',
            # 'ExoMol/hdf5/MgO/MgO_LiTY_CO2broad_1.0-30000.0_T62xP22_001.h5'
            'datasource': 'MgO_T62xP22_001',
            'lower_wn': 1.0,
            'upper_wn': 30000.0,
        },
        'uv_config': { 
            'xuv_rel_path': 'ExoMol/24Mg_16O/PhoMol',
            'xuv_file': 'MgO.uvxsc',
            'lower_wn': 30000, # raw: approx 5400 cm-1, but should not overwrite the line absorption
            'upper_wn': 1e7/50, # approx 200000 cm-1
            't_grid': np.arange(200,4201,800), # uv no need to be such high res
            'p_grid': [1e-6,1e-5,1e-4,1e-3], # in bar, from 0.1 Pa to 100 Pa
        }
    },
    
    # Add other gases (H2O, CH4, etc.) here...
    # SiO2 custom T grid
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
    },
    'O2-O2':{
        'cia_rel_path': 'hitran/O2-O2_2024',
        'cia_file': 'O2-O2_2024.cia',
        'lower_wn': [1150.000,7545,7583,9091.0000,10512.000,12600.120,14206.000,15290.000,15292.000,16645.000,16658.000,16668.000,16780.000,16791.000,20000.000],
        'upper_wn': [1950.000,8355,8355,9596.0000,11228.000,13839.642,14898.000,16664.000,16664.000,29784.000,29748.000,29802.000,29757.000,29837.000,33670.000],
        't_grid': [150,200,250,300],
        'p_grid': [1.0]
    },
    'O2-N2':{
        'cia_rel_path': 'hitran/O2-N2_2024',
        'cia_file': 'O2-N2_2024.cia',
        'lower_wn': [1300, 1850, 2000, 7583, 12600],
        'upper_wn': [1850, 3000, 2698, 8355, 13840], 
        't_grid': [150,200,250,300,350,400],
        'p_grid': [1.0]
    },
    'O2-CO2':{
        'cia_rel_path': 'hitran/O2-CO2_2024',
        'cia_file': 'O2-CO2_2024.cia',
        'lower_wn': [9105, 12600],
        'upper_wn': [9545, 13840],
        't_grid': [250,300],
        'p_grid': [1.0]
    }
    # parse_cia.py
}

# ==========================================
# 4. Simulation Parameters
# ==========================================
NUM_KTERM = 20
INCLUDE_CIA = True
INCLUDE_SOLAR_SED = True

# New Option: Ultra Hot Atmosphere (Calculate UV/Rayleigh for 'lw' if True)
ULTRA_HOT_ATMOSPHERE = False 

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