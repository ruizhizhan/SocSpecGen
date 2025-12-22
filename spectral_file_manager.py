import os
import numpy as np
import stat
from typing import Literal, List, Dict, Tuple, Set

# Import configuration
import config_data as cfg

# ==========================================
# Helper Functions (Logic Pre-calculation)
# ==========================================

def get_active_cias(selected_gases: List[str]) -> List[Tuple[str, str, str]]:
    """
    Identifies active CIA pairs based on the selected molecules.
    """
    active_cias = []
    selected_set = set(selected_gases)
    name_to_id = {name: cfg.GAS_LIBRARY[name]['gas_id'] for name in selected_gases}
    
    for pair_name in cfg.CIA_LIBRARY.keys():
        parts = pair_name.split('-')
        if len(parts) != 2: continue 
        mol1, mol2 = parts[0], parts[1]
        
        if mol1 in selected_set and mol2 in selected_set:
            id1 = name_to_id[mol1]
            id2 = name_to_id[mol2]
            active_cias.append((id1, id2, pair_name))
            
    return active_cias

def calculate_band_occupancy(
    wnedges: np.ndarray, 
    selected_gases: List[str], 
    active_cia_tuples: List[Tuple[str, str, str]]
) -> List[List[str]]:
    """
    Determines which gases are present in each spectral band.
    """
    band_map = []
    num_bands = len(wnedges) - 1
    gas_configs = {name: cfg.GAS_LIBRARY[name] for name in selected_gases}
    
    for i in range(num_bands):
        band_min = wnedges[i]
        band_max = wnedges[i+1]
        current_band_gas_ids: Set[str] = set()
        
        # 1. Check Standard Absorption and UV
        for gas_name in selected_gases:
            gas_conf = gas_configs[gas_name]
            gas_id = gas_conf['gas_id']
            ranges_to_check = []
            
            if gas_conf.get('gas_abs_config'):
                ranges_to_check.append((gas_conf['gas_abs_config']['lower_wn'], gas_conf['gas_abs_config']['upper_wn']))
            if gas_conf.get('uv_config'):
                ranges_to_check.append((gas_conf['uv_config']['lower_wn'], gas_conf['uv_config']['upper_wn']))
            
            for r_min, r_max in ranges_to_check:
                if max(band_min, r_min) < min(band_max, r_max):
                    current_band_gas_ids.add(gas_id)
                    break 
        
        # 2. Check Active CIAs
        for id1, id2, pair_name in active_cia_tuples:
            cia_conf = cfg.CIA_LIBRARY[pair_name]
            cia_min = cia_conf['lower_wn']
            cia_max = cia_conf['upper_wn']
            if max(band_min, cia_min) < min(band_max, cia_max):
                current_band_gas_ids.add(id1)
                current_band_gas_ids.add(id2)
        
        band_map.append(sorted(list(current_band_gas_ids)))
        
    return band_map

def get_solar_wn_range(file_path: str) -> Tuple[float, float]:
    """
    Reads the Solar Spectrum file and determines the wavenumber range coverage.
    Converts Wavelength (m) to Wavenumber (cm-1).
    """
    min_wl_m = float('inf')
    max_wl_m = float('-inf')
    in_data = False
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Solar spectrum file not found at: {file_path}")

    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if '*BEGIN_DATA' in line:
                in_data = True
                continue
            if '*END' in line:
                break
            
            if in_data and line:
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        wl_m = float(parts[0]) # Wavelength in meters
                        if wl_m < min_wl_m: min_wl_m = wl_m
                        if wl_m > max_wl_m: max_wl_m = wl_m
                    except ValueError:
                        continue
    
    if min_wl_m == float('inf') or max_wl_m == float('-inf'):
        raise ValueError(f"Could not parse data from solar file: {file_path}")

    # Conversion: nu (cm-1) = 1 / (lambda (m) * 100)
    # Note: Min Wavelength -> Max Wavenumber
    max_wn = 1.0 / (min_wl_m * 100.0)
    min_wn = 1.0 / (max_wl_m * 100.0)
    
    return min_wn, max_wn

# ==========================================
# 2. Worker Script Generator 
# ==========================================

def write_worker_script(filename, test_name, selected_gases_list, spec_type: Literal['lw','sw']):
    """
    Writes a Python script dynamically configured with injected logic and data.
    """
    file_path = os.path.join(cfg.SLURM_DIR, filename)
    
    # 1. Prepare Data
    current_gas_configs = [cfg.GAS_LIBRARY[g] for g in selected_gases_list]
    molecules_str = [g['molecule'] for g in current_gas_configs]
    gas_ids = [g['gas_id'] for g in current_gas_configs]
    
    active_cia_tuples = get_active_cias(selected_gases_list)
    relevant_cia_configs = {pair: cfg.CIA_LIBRARY[pair] for _, _, pair in active_cia_tuples}

    # --- Wavenumber Logic (Modified) ---
    target_wnedges = cfg.WNEDGES # Default to full edges
    
    # If SW, restrict WNEDGES to Solar Data Range
    if spec_type == 'sw':
        # Construct path to solar file (Using path logic from Block 3/Section 6)
        solar_path = os.path.join(cfg.ROOT_DIR, 'stellar_spectra/soc_in', cfg.STAR_NAME)
        print(f"Reading solar spectrum from: {solar_path}")
        
        try:
            min_solar_wn, max_solar_wn = get_solar_wn_range(solar_path)
            print(f"Solar Data Range (cm-1): {min_solar_wn:.2f} - {max_solar_wn:.2f}")
            
            # Filter global WNEDGES
            # We keep edges that are within the solar range (approximate overlap)
            # Logic: keep edges where edge >= min and edge <= max
            # Note: We need at least 2 edges to form a band.
            
            # Using slightly relaxed bounds to ensure we don't cut off a band that partially overlaps if desired,
            # but strictly "solar data available" usually means we restrict to the range.
            valid_indices = np.where((cfg.WNEDGES >= min_solar_wn) & (cfg.WNEDGES <= max_solar_wn))[0]
            
            if len(valid_indices) < 2:
                # Fallback or strict check
                # Try to include bands that *overlap* the range
                print("Warning: Solar range strictly contained very few edges. expanding to overlaps.")
                valid_indices = np.where((cfg.WNEDGES >= min_solar_wn) & (cfg.WNEDGES <= max_solar_wn))[0]

            if len(valid_indices) >= 2:
                target_wnedges = cfg.WNEDGES[valid_indices]
                print(f"Restricted WNEDGES to {len(target_wnedges)} edges based on Solar Data.")
            else:
                print("Warning: Solar data range does not overlap sufficiently with WNEDGES. Using full edges.")
                
        except Exception as e:
            print(f"Error reading solar file: {e}. Using full WNEDGES.")

    # Calculate Band Map using the TARGET edges (Important: Must use the edges that will be injected)
    band_gas_map = calculate_band_occupancy(target_wnedges, selected_gases_list, active_cia_tuples)
    
    with open(file_path, 'w') as f:
        # --- Imports ---
        f.write("import os\n")
        f.write("import numpy as np\n")
        f.write("from numpy import array\n") 
        f.write("import netCDF4\n")
        f.write("from typing import Literal\n")
        f.write("from tools import read_wnedges, generate_LBL_from_ExoMol_hdf5, find_index\n\n")

        # --- Dynamic Configuration Injection ---
        f.write(f"# --- Configuration injected by Manager ---\n")
        f.write(f"test_name = '{test_name}'\n")
        f.write(f"Molecule_str_list = {molecules_str}\n")
        f.write(f"gas_id_list = {gas_ids}\n")
        f.write(f"spec_type = '{spec_type}'\n")
        f.write(f"star_name = '{cfg.STAR_NAME}'\n")
        f.write(f"root = '{cfg.ROOT_DIR}'\n")
        f.write(f"num_kterm = {cfg.NUM_KTERM}\n")
        f.write(f"include_cia = {cfg.INCLUDE_CIA}\n")
        f.write(f"include_solar_sed = {cfg.INCLUDE_SOLAR_SED}\n")
        f.write(f"ultra_hot_atmosphere = {cfg.ULTRA_HOT_ATMOSPHERE}\n")
        # add solar path
        f.write(f"solar_path = os.path.join(root, 'stellar_spectra', 'soc_in', '{cfg.STAR_NAME}')\n\n")
        
        # Inject computed maps and configs
        f.write(f"BAND_GAS_MAP = {band_gas_map}\n") 
        f.write(f"ACTIVE_CIA_TUPLES = {active_cia_tuples}\n")
        f.write(f"RELEVANT_CIA_CONFIGS = {relevant_cia_configs}\n")
        
        # Inject the (potentially filtered) WNEDGES
        f.write(f"wnedges = np.array({target_wnedges.tolist()})\n")
        f.write(f"band_num = len(wnedges)-1\n")
        f.write(f"outputfilename = f'sp_{{spec_type}}_b{{band_num}}_{{star_name}}_{{test_name}}'\n\n")
        
        f.write(f"# Full configuration for gases\n")
        f.write(f"GAS_CONFIGS = {current_gas_configs}\n\n")

        # --- Worker Logic ---
        f.write(r"""
update_library = True
output_path_list = []; mon_path_list = []; LbL_path_list = []

# 1. Processing xsec tables and generating PT files
for config in GAS_CONFIGS:
    Molecule_str = config['molecule']
    gas_id = config['gas_id']
    path_in = config['gas_abs_config']['hdf5_rel_path']
    datasource = config['gas_abs_config']['datasource']
    
    hdf5_path = os.path.join(root, path_in)
    print(f"Processing xsec tables for {Molecule_str}...")
    
    output_path, mon_path, LbL_path, T_grid, P_grid = generate_LBL_from_ExoMol_hdf5(
        root, hdf5_path, Molecule_str, datasource, update_library, test_name
    )
    output_path_list.append(output_path)
    mon_path_list.append(mon_path)
    LbL_path_list.append(LbL_path)
    
    # Generate PT file per gas
    pt_file_path = os.path.join(root, f'block5/pt_file_{test_name}_{gas_id}')
    if os.path.exists(pt_file_path):
        os.remove(pt_file_path)
    with open(pt_file_path, "a") as file:
        file.write('*PTVAL\n')
        for P in P_grid:
            file.write(str(P*1e+5))
            for T in T_grid:
                file.write(f' {T}')
            file.write('\n')
        file.write('*END')

# 2. Reference PT file
ref_pt_path = os.path.join(root, f'block5/ref_pt_file{test_name}')
if os.path.exists(ref_pt_path):
    os.remove(ref_pt_path)
with open(ref_pt_path, "a") as file:
    for gas_id in gas_id_list:
        file.write(f'*REF 1 {band_num} {gas_id} 1e+5 300.0\n')
    file.write('*END')

# 3. Generate skeleton spectral file using prep_spec
exec_file_name = f"sp_exec{test_name}.sh"
if os.path.exists(exec_file_name):
    os.remove(exec_file_name)
skeleton_file_name = f'sp_b{band_num}_{test_name}'
if os.path.exists(skeleton_file_name):
    os.remove(skeleton_file_name)

with open(exec_file_name, "w+") as f:
    f.write('prep_spec <<EOF\n')
    f.write(f'{skeleton_file_name}\n')
    f.write(f'{band_num}\n')                      # Number of bands
    f.write(f'{len(gas_id_list)}\n')              # Number of absorbing gases
    for gas_id in gas_id_list:
        f.write(f'{gas_id}\n')
    
    # --- CIA Section ---
    if len(ACTIVE_CIA_TUPLES) > 0:
        f.write(f'{len(ACTIVE_CIA_TUPLES)}\n')    # Number of continuum pairs
        for id1, id2, _ in ACTIVE_CIA_TUPLES:
            f.write(f'{id1} {id2}\n')             # Write pairs (e.g. "2 2")
    else:
        f.write('0\n')                            # No Continua
        
    f.write('0\n')                                # Number of aerosols
    f.write('c\n')                                # Band units (c for cm-1)

    f.write(f'{wnedges[0]} ')                     # Write band edges
    for band in wnedges[1:-1]:
        f.write(f'{band}\n')
        f.write(f'{band} ')
    f.write(f'{wnedges[-1]}\n')

    # --- Band Mapping Section ---
    # Determine which gases are active in each band using BAND_GAS_MAP
    for i, gases_in_band in enumerate(BAND_GAS_MAP):
        if len(gases_in_band) > 0:
            f.write(' '.join(gases_in_band) + '\n')
        else:
            f.write('0\n')

    # Continua IDs in each band
    # Currently default to '0' for all bands
    for _ in wnedges[:-1]:
        f.write('0\n')
        
    f.write('n\n')                                    # Exclude no bands
    f.write('-1\n')                                   # Close prep_spec
    f.write('EOF\n')

os.chmod(exec_file_name, 0o777)
os.system(f'./{exec_file_name}')
os.system(f'rm {exec_file_name}')

# 4. Generate corrk data
# Processing corrk logic
for config, output_path, mon_path, LbL_path in zip(GAS_CONFIGS, output_path_list, mon_path_list, LbL_path_list):
    gas_id = config['gas_id']
    lower = config['gas_abs_config']['lower_wn']
    upper = config['gas_abs_config']['upper_wn']
        
    print(f"Running corr_k for Gas ID {gas_id}...")
    exec_file_corrk = f"corr_k_ExoMol_{test_name}.sh"
    if os.path.exists(exec_file_corrk):
        os.remove(exec_file_corrk)
        
    with open(exec_file_corrk, "w+") as f:
        # Note: wnedges here refers to the injected (potentially subset) array
        idx_lower, idx_upper = find_index(wnedges[:-1], wnedges[1:], lower, upper)
        f.write('Ccorr_k ')
        f.write(f'-s {skeleton_file_name} ')
        f.write(f'-R {idx_lower} {idx_upper} ')
        f.write(f'-F {root}/block5/pt_file_{test_name}_{gas_id} ')
        f.write(f'-r {root}/block5/ref_pt_file{test_name} ')
        f.write(f'-l {gas_id} 1.0e5 ') # a limit for grey gas approximation
        f.write(f'-n {num_kterm} ')
        f.write('-lk'+' ') 
        if spec_type == 'sw':
            f.write(f'+S {solar_path} ')                     # Solar spectrum
        else:
            f.write('+p'+' ')                                # Planckian Weighting
        f.write(f'-o {output_path} ')
        f.write(f'-m {mon_path} ')
        f.write(f'-L {LbL_path} ')
        f.write('-np 1')

    os.chmod(exec_file_corrk, 0o777)
    os.system(f'./{exec_file_corrk}')
    os.system(f'rm {exec_file_corrk}')

# 5. Include CIA (Hitran)
if include_cia and len(ACTIVE_CIA_TUPLES) > 0:
    exec_file_CIA = f"corr_k_CIA_{test_name}.sh"
    if os.path.exists(exec_file_CIA):
        os.remove(exec_file_CIA)

    with open(exec_file_CIA, "w+") as f:
        # Update: unpack id1 and id2 from the tuple
        for id1, id2, pair_name in ACTIVE_CIA_TUPLES:
            cia_conf = RELEVANT_CIA_CONFIGS.get(pair_name)
            if not cia_conf:
                print(f"Warning: Configuration for CIA pair {pair_name} missing.")
                continue

            # 5a. Retrieve T_grid
            if 't_grid' in cia_conf:
                T_cia_grid = np.array(cia_conf['t_grid'])
            else:
                raise ValueError(f"No t_grid found for {pair_name}")
            if 'p_grid' in cia_conf:
                P_cia_grid = np.array(cia_conf['p_grid'])
            else:
                raise ValueError(f"No p_grid found for {pair_name}")

            # 5b. Generate unique PT file
            pt_cia_path = os.path.join(root, f'block19/pt_cia_{pair_name}_{test_name}')
            if os.path.exists(pt_cia_path):
                os.remove(pt_cia_path)
            
            with open(pt_cia_path, "w") as pt_file:
                pt_file.write('*PTVAL\n')
                for P_0 in P_cia_grid:
                    pt_file.write(f'{P_0 * 1e+5}')
                    for T in T_cia_grid:
                        pt_file.write(f' {T}')
                    pt_file.write('\n')
                pt_file.write('*END')

            # 5c. Run corr_k
            cia_file_path = os.path.join(root, cia_conf['cia_rel_path'], cia_conf['cia_file'])
            cia_lower = cia_conf['lower_wn']
            cia_upper = cia_conf['upper_wn']

            idx_lower, idx_upper = find_index(wnedges[:-1], wnedges[1:], int(cia_lower), int(cia_upper))
            
            cia_out_base = f"output_CIA_{pair_name}_{test_name}"
            
            # Modified line: Use dynamic id1 and id2 instead of hardcoded '2 2'
            f.write(f'Ccorr_k -CIA {cia_file_path} -R {idx_lower} {idx_upper} ')
            f.write(f'-F {pt_cia_path} -ct {id1} {id2} 1000.0 -i 1.0 -t 1.0e-2 ')
            f.write(f'-s {skeleton_file_name} ')
            if spec_type == 'sw':
                f.write(f'+S {solar_path} ')                     # Solar spectrum
            else:
                f.write('+p'+' ')                                # Planckian Weighting
            f.write('-lk ')
            f.write(f'-o {root}/block19/{cia_out_base} ')
            if os.path.exists(f'{root}/block19/{cia_out_base}'):
                os.remove(f'{root}/block19/{cia_out_base}')
            if os.path.exists(f'{root}/block19/{cia_out_base}.nc'):
                os.remove(f'{root}/block19/{cia_out_base}.nc')
            f.write(f'-m {root}/block19/monitoring_CIA_{pair_name}_{test_name} ')
            if os.path.exists(f'{root}/block19/monitoring_CIA_{pair_name}_{test_name}'):
                os.remove(f'{root}/block19/monitoring_CIA_{pair_name}_{test_name}')
            f.write(f'-L {root}/block19/LBL_CIA_{pair_name}_{test_name}.nc\n')
            if os.path.exists(f'{root}/block19/LBL_CIA_{pair_name}_{test_name}.nc'):
                os.remove(f'{root}/block19/LBL_CIA_{pair_name}_{test_name}.nc')

    os.chmod(exec_file_CIA, 0o777)
    os.system(f'./{exec_file_CIA}')
    os.system(f'rm {exec_file_CIA}')

# 6. Add UV lines to SW files
output_path_xuv_list = []
if spec_type == 'sw' or ultra_hot_atmosphere:
    for config in GAS_CONFIGS:
        uv_conf = config['uv_config']
        if uv_conf is None:
            continue
            
        gas_id = config['gas_id']
        Molecule_str = config['molecule']
        
        # uv_pt_file = uv_conf['pt_file_name']
        # UV_gas_path = os.path.join(root, uv_conf['xuv_rel_path'])
        UV_gas = uv_conf['xuv_file']
        lower = uv_conf['lower_wn']
        upper = uv_conf['upper_wn']
        
        # os.system(f'cp {UV_gas_path}{uv_pt_file} {root}/block5')
        pt_uv_path = os.path.join(root, f'block5',f'pt_uv_{Molecule_str}_{test_name}')
        if os.path.exists(pt_uv_path):
            os.remove(pt_uv_path)
        if 't_grid' in uv_conf:
            T_uv_grid = np.array(uv_conf['t_grid'])
        else:
            raise ValueError(f"No t_grid found for UV config of {Molecule_str}")
        if 'p_grid' in uv_conf:
            P_uv_grid = np.array(uv_conf['p_grid'])
        else:
            raise ValueError(f"No p_grid found for UV config of {Molecule_str}")
        
        with open(pt_uv_path, "w") as pt_file:
            pt_file.write('*PTVAL\n')
            for P_0 in P_uv_grid:
                pt_file.write(f'{P_0 * 1e+5}')
                for T in T_uv_grid:
                    pt_file.write(f' {T}')
                pt_file.write('\n')
            pt_file.write('*END')
        
        exec_file_corrk_xuv = f"corr_k_ExoMol_{test_name}_xuv.sh"
        if os.path.exists(exec_file_corrk_xuv):
            os.remove(exec_file_corrk_xuv)
            
        output_path_xuv = os.path.join(root, f'block5/output_xuv_{Molecule_str}_{test_name}')
        mon_path_xuv = os.path.join(root, f'block5/monitoring_xuv_{Molecule_str}_{test_name}')
        LBL_path_xuv = os.path.join(root, f'block5/LBL_xuv_{Molecule_str}_{test_name}.nc')
        
        if os.path.exists(output_path_xuv): os.remove(output_path_xuv)
        if os.path.exists(output_path_xuv + '.nc'): os.remove(output_path_xuv + '.nc')
        if os.path.exists(mon_path_xuv): os.remove(mon_path_xuv)
        if os.path.exists(LBL_path_xuv): os.remove(LBL_path_xuv)

        with open(exec_file_corrk_xuv, "w+") as f:
            f.write(f'Ccorr_k -s {skeleton_file_name} ')
            f.write(f'-UVX {os.path.join(root, uv_conf["xuv_rel_path"], UV_gas)} ')
            idx_lower, idx_upper = find_index(wnedges[:-1], wnedges[1:], lower, upper)
            f.write(f'-R {idx_lower} {idx_upper} ')
            f.write(f'-F {pt_uv_path} ')
            f.write('-i 1.0 ')
            f.write(f'-l {gas_id} 1.0e1 ')
            f.write('-t 1.0e-3 ')
            if spec_type == 'sw':
                f.write(f'+S {solar_path} ')                     # Solar spectrum
            else:
                f.write('+p ')                                # Planckian Weighting
            f.write(f'-o {output_path_xuv} -m {mon_path_xuv} -L {LBL_path_xuv} -np 1')

        os.chmod(exec_file_corrk_xuv, 0o777)
        os.system(f'./{exec_file_corrk_xuv}')
        os.system(f'rm {exec_file_corrk_xuv}')
        output_path_xuv_list.append(output_path_xuv)

# 7. Final Assembly: Fill in Spectral File
exec_file_sp = f"fill_in_sp_{test_name}.sh"
with open(exec_file_sp, "w+") as f:
    f.write('prep_spec <<EOF\n')
    f.write(f'{skeleton_file_name}\n')
    f.write('a\n') # append

    # Block 3: Rayleigh (SW or ultrahot)
    if spec_type == 'sw' or ultra_hot_atmosphere:
        f.write('3\n')
        f.write(f'{solar_path}\n')
        f.write('C\n') # Custom composition
        f.write('A\n') # All gases

    # Block 5: k-coeff (Normal)
    f.write('5\n')
    f.write(f'{output_path_list[0]}\n')
    for output_path in output_path_list[1:]:
        f.write('5\n')
        f.write('y\n')
        f.write(f'{output_path}\n')
    
    # Block 5: k-coeff (UV, SW or ultra-hot)
    if spec_type == 'sw' or ultra_hot_atmosphere:
        for output_path_xuv in output_path_xuv_list:
            f.write('5\n')
            f.write('y\n')
            f.write(f'{output_path_xuv}\n')

    # Block 6: Thermal source (LW only)
    if spec_type == 'lw':
        f.write('6\n')
        f.write('n\n')
        f.write('T\n')
        f.write('50 4500\n')
        f.write('445\n')
        
    # Block 2: Solar spectrum (SW only)
    if include_solar_sed and spec_type == 'sw':
        f.write('2\n')
        f.write('n\n')
        f.write('y\n')

    # Block 19: Continuum (CIA)
    if include_cia and len(ACTIVE_CIA_TUPLES) > 0:
        f.write('19\n')
        first_pair = ACTIVE_CIA_TUPLES[0][2] 
        f.write(f'{root}/block19/output_CIA_{first_pair}_{test_name}\n')
        
        for _, _, pair in ACTIVE_CIA_TUPLES[1:]:
             f.write('19\n')
             f.write(f'{root}/block19/output_CIA_{pair}_{test_name}\n')
             
    f.write('-1\n')
    f.write('EOF\n')

os.chmod(exec_file_sp, 0o777)
os.system(f'./{exec_file_sp}')
os.system(f'rm {exec_file_sp}')

# 8. Move files
final_dir = os.path.join(root, f'spectral_files/sp_b{band_num}')
if not os.path.exists(final_dir):
    os.makedirs(final_dir)
    
print(f"Moving {outputfilename} to {final_dir}")
os.system(f'mv sp_b{band_num}_{test_name} {os.path.join(final_dir, outputfilename)}')
if os.path.exists(f'sp_b{band_num}_{test_name}_k'):
    os.system(f'mv sp_b{band_num}_{test_name}_k {os.path.join(final_dir, outputfilename)}_k')
""")

# ==========================================
# 3. Slurm Script Generator 
# ==========================================

def write_slurm_script(job_name, case_name_list):
    slurm_path = os.path.join(cfg.SLURM_DIR, f'{job_name}.sh')
    ncores = 8 
    with open(slurm_path, 'w') as f:
        f.write('#!/bin/bash\n')
        f.write(f'#SBATCH --job-name={job_name}\n')
        f.write(f'#SBATCH --output=output_%j.out\n')
        f.write(f'#SBATCH --error=error_%j.err\n')
        f.write(f'#SBATCH -N 1\n')
        f.write(f'#SBATCH -c {ncores}\n')
        f.write(f'#SBATCH --partition=wzhcnormal\n\n')
        
        f.write(f'cd {cfg.SLURM_DIR}\n')
        f.write('source /work/home/ac9b0k6rio/miniconda3/etc/profile.d/conda.sh\n')
        f.write('source /work/home/ac9b0k6rio/miniconda3/bin/activate exo-k\n\n')
        
        for case_name in case_name_list:
            f.write(f'# Processing Case: {case_name}\n')
            f.write(f'python -u {case_name}_lw.py\n')
            f.write(f'python -u {case_name}_sw.py\n')

# ==========================================
# 4. Main Execution Loop
# ==========================================

if __name__ == "__main__":
    
    # ----------------------------------------
    # Select gases for this run
    # ----------------------------------------
    SELECTED_MOLECULES = ['CO2']
    
    # Validate selection against Gas Library
    for m in SELECTED_MOLECULES:
        if m not in cfg.GAS_LIBRARY:
            raise ValueError(f"Molecule {m} not found in config_data.GAS_LIBRARY")
            
    primary_gas = SELECTED_MOLECULES[0]
    num_bands = len(cfg.WNEDGES) - 1
    job_identifier = f"sp_b{num_bands}_{cfg.STAR_NAME}"
    test_name = "CO2_T62xP22_001_nk20"
    
    print(f"Generating scripts for: {test_name}")
    print(f"Selected Gases: {SELECTED_MOLECULES}")
    
    # Generate Python Worker Scripts
    write_worker_script(f"{test_name}_lw.py", test_name, SELECTED_MOLECULES, 'lw')
    write_worker_script(f"{test_name}_sw.py", test_name, SELECTED_MOLECULES, 'sw')
    
    # Generate Slurm Submission Script
    case_name_list = [test_name]
    write_slurm_script(job_identifier, case_name_list)
    
    slurm_file = f'{job_identifier}.sh'
    os.chmod(os.path.join(cfg.SLURM_DIR, slurm_file), 0o755)
    print(f"To submit job, run:\n cd {cfg.SLURM_DIR} && sbatch {slurm_file}")