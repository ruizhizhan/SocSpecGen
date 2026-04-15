import os
import numpy as np
import stat
from typing import Literal, List, Dict, Tuple, Set, Optional

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

def get_cia_segments(pair_name: str, cia_conf: Dict) -> List[Tuple[int, float, float]]:
    """
    Returns validated CIA segment bounds while preserving floating-point limits.
    """
    if 'lower_wn' not in cia_conf or 'upper_wn' not in cia_conf:
        raise ValueError(f"CIA pair {pair_name} is missing lower_wn/upper_wn")

    lowers = cia_conf['lower_wn']
    uppers = cia_conf['upper_wn']

    if not isinstance(lowers, (list, np.ndarray)):
        lowers = [lowers]
    if not isinstance(uppers, (list, np.ndarray)):
        uppers = [uppers]

    if len(lowers) != len(uppers):
        raise ValueError(
            f"CIA pair {pair_name} has mismatched lower/upper segment counts: "
            f"{len(lowers)} vs {len(uppers)}"
        )

    segments: List[Tuple[int, float, float]] = []
    for seg_idx, (lower, upper) in enumerate(zip(lowers, uppers)):
        lower_wn = float(lower)
        upper_wn = float(upper)
        if lower_wn >= upper_wn:
            raise ValueError(
                f"CIA pair {pair_name} segment {seg_idx} has invalid bounds: "
                f"{lower_wn} >= {upper_wn}"
            )
        segments.append((seg_idx, lower_wn, upper_wn))

    return segments

def resolve_cia_t_grids(
    active_cia_tuples: List[Tuple[str, str, str]],
    cia_configs: Dict[str, Dict],
    policy: str = 'native',
    shared_t_grid: Optional[List[float]] = None,
    forced_by_pair: Optional[Dict[str, List[float]]] = None
) -> Tuple[Dict[str, List[float]], List[str]]:
    """
    Return the temperature grid to use for each active CIA pair.

    The manager can either use the pair-local native grids or a single shared
    grid for all active pairs. Individual pairs may still be overridden
    explicitly via a mapping keyed by pair name.
    """
    resolved_grids: Dict[str, List[float]] = {}
    forced_by_pair = forced_by_pair or {}
    warnings: List[str] = []

    if policy not in {'native', 'shared'}:
        raise ValueError(f"Unsupported CIA_T_GRID_POLICY: {policy}")

    if policy == 'shared':
        if shared_t_grid is None:
            raise ValueError("CIA_SHARED_T_GRID must be set when CIA_T_GRID_POLICY='shared'")
        shared_grid = [float(value) for value in np.asarray(shared_t_grid, dtype=float).tolist()]
        if len(shared_grid) < 2:
            raise ValueError(
                "CIA_SHARED_T_GRID must contain at least two temperatures. "
                f"Got: {shared_grid}"
            )

        for _, _, pair_name in active_cia_tuples:
            cia_conf = cia_configs.get(pair_name)
            if cia_conf is None or 't_grid' not in cia_conf:
                raise ValueError(f"No t_grid found for CIA pair: {pair_name}")
            native_grid = [float(value) for value in np.asarray(cia_conf['t_grid'], dtype=float).tolist()]
            grid_override = forced_by_pair.get(pair_name)
            grid = shared_grid if grid_override is None else [float(value) for value in np.asarray(grid_override, dtype=float).tolist()]
            if len(grid) < 2:
                raise ValueError(
                    f"CIA pair {pair_name} must use at least two temperatures. Got: {grid}"
                )
            if not set(shared_grid).issubset(set(native_grid)):
                warnings.append(
                    f"{pair_name}: shared grid {shared_grid} is not a subset of native grid {native_grid}; "
                    "this will require extrapolation or model fitting."
                )
            resolved_grids[pair_name] = grid

        return resolved_grids, warnings

    for _, _, pair_name in active_cia_tuples:
        cia_conf = cia_configs.get(pair_name)
        if cia_conf is None or 't_grid' not in cia_conf:
            raise ValueError(f"No t_grid found for CIA pair: {pair_name}")

        native_grid = [float(value) for value in np.asarray(cia_conf['t_grid'], dtype=float).tolist()]
        grid_override = forced_by_pair.get(pair_name)
        if grid_override is None:
            grid = native_grid
        else:
            grid = [float(value) for value in np.asarray(grid_override, dtype=float).tolist()]

        if len(grid) < 2:
            raise ValueError(
                f"CIA pair {pair_name} must use at least two temperatures. Got: {grid}"
            )

        resolved_grids[pair_name] = grid

    return resolved_grids, warnings

def analyze_cia_band_coverage(
    wnedges: np.ndarray,
    active_cia_tuples: List[Tuple[str, str, str]],
    cia_configs: Dict[str, Dict]
) -> Tuple[Dict[str, Dict], List[int], Dict[int, Set[str]]]:
    """
    Computes per-pair and per-segment CIA band coverage and validates that every
    active CIA segment maps onto at least one spectral band.
    """
    pair_band_coverage: Dict[str, Dict] = {}
    bands_with_cia: Set[int] = set()
    required_gases_by_band: Dict[int, Set[str]] = {}

    for id1, id2, pair_name in active_cia_tuples:
        cia_conf = cia_configs.get(pair_name)
        if cia_conf is None:
            raise ValueError(f"Missing CIA configuration for pair: {pair_name}")

        pair_bands: Set[int] = set()
        segment_reports = []

        for seg_idx, seg_lower, seg_upper in get_cia_segments(pair_name, cia_conf):
            covered_bands = []
            for band_idx in range(len(wnedges) - 1):
                band_number = band_idx + 1
                band_min = float(wnedges[band_idx])
                band_max = float(wnedges[band_idx + 1])
                if max(band_min, seg_lower) < min(band_max, seg_upper):
                    covered_bands.append(band_number)
                    pair_bands.add(band_number)
                    bands_with_cia.add(band_number)
                    required_gases_by_band.setdefault(band_number, set()).update({id1, id2})

            if not covered_bands:
                raise ValueError(
                    f"CIA pair {pair_name} segment {seg_idx} ({seg_lower}, {seg_upper}) "
                    "does not overlap any configured spectral band."
                )

            segment_reports.append({
                'segment_index': seg_idx,
                'lower_wn': seg_lower,
                'upper_wn': seg_upper,
                'bands': covered_bands,
            })

        if not pair_bands:
            raise ValueError(
                f"CIA pair {pair_name} is active but all of its segments were dropped by band coverage."
            )

        pair_band_coverage[pair_name] = {
            'gas_ids': [id1, id2],
            'bands': sorted(pair_bands),
            'segments': segment_reports,
        }

    return pair_band_coverage, sorted(bands_with_cia), required_gases_by_band

def calculate_band_occupancy(
    wnedges: np.ndarray, 
    selected_gases: List[str], 
    include_uv: bool = True  
) -> Tuple[List[List[str]], Dict[str, List[int]], List[int]]:
    """
    Determines which gases are present in each spectral band for BLOCK 4 only.
    CIA is intentionally excluded here because CIA is handled separately by
    BLOCK 18/19 and must not affect the gas occupancy map.
    """
    band_map = []
    num_bands = len(wnedges) - 1
    gas_configs = {name: cfg.GAS_LIBRARY[name] for name in selected_gases}
    
    # New: Dictionary to track missing bands for each gas
    missing_bands_report = {name: [] for name in selected_gases}
    
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
            
            # check if uv included
            if include_uv and gas_conf.get('uv_config'):
                ranges_to_check.append((gas_conf['uv_config']['lower_wn'], gas_conf['uv_config']['upper_wn']))
            
            for r_min, r_max in ranges_to_check:
                if max(band_min, r_min) < min(band_max, r_max):
                    current_band_gas_ids.add(gas_id)
                    break 

        # Record missing gases for this band.
        for gas_name in selected_gases:
            gid = gas_configs[gas_name]['gas_id']
            if gid not in current_band_gas_ids:
                missing_bands_report[gas_name].append(i + 1)

        band_map.append(sorted(list(current_band_gas_ids)))

    empty_bands = [idx + 1 for idx, gases in enumerate(band_map) if not gases]
    
    # New: Print the report
    print("\n" + "="*40)
    print("      GAS BAND OCCUPANCY REPORT")
    print("="*40)
    for gas_name, missing_list in missing_bands_report.items():
        if missing_list:
            # Shorten output if too long
            if len(missing_list) > 20:
                missing_str = f"{missing_list[:10]} ... {missing_list[-5:]} (Total {len(missing_list)} bands)"
            else:
                missing_str = str(missing_list)
            print(f"Gas {gas_name:<5} is MISSING in Bands: {missing_str}")
        else:
            print(f"Gas {gas_name:<5} is present in ALL bands.")
    if empty_bands:
        print(f"Bands with no gas support: {empty_bands}")
    else:
        print("All bands have at least one gas contributor.")
    print("="*40 + "\n")

    return band_map, missing_bands_report, empty_bands

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

# Changed: Added job_dir parameter
def write_worker_script(job_dir, filename, test_name, selected_gases_list, spec_type: Literal['lw','sw']):
    """
    Writes a Python script dynamically configured with injected logic and data.
    """
    file_path = os.path.join(job_dir, filename)
    
    # 1. Prepare Data
    current_gas_configs = [cfg.GAS_LIBRARY[g] for g in selected_gases_list]
    molecules_str = [g['molecule'] for g in current_gas_configs]
    gas_ids = [g['gas_id'] for g in current_gas_configs]
    cia_t_grid_policy = getattr(cfg, 'CIA_T_GRID_POLICY', 'native')
    cia_shared_t_grid = getattr(cfg, 'CIA_SHARED_T_GRID', None)
    
    candidate_cia_tuples = get_active_cias(selected_gases_list)
    active_cia_tuples = candidate_cia_tuples if cfg.INCLUDE_CIA else []
    relevant_cia_configs = {pair: cfg.CIA_LIBRARY[pair] for _, _, pair in active_cia_tuples}
    forced_cia_t_grids = getattr(cfg, 'CIA_T_GRID_OVERRIDE_BY_PAIR', None)
    cia_t_grids_by_pair, cia_grid_warnings = resolve_cia_t_grids(
        active_cia_tuples,
        relevant_cia_configs,
        policy=cia_t_grid_policy,
        shared_t_grid=cia_shared_t_grid,
        forced_by_pair=forced_cia_t_grids,
    )
    if active_cia_tuples:
        print(
            f"[{spec_type}] CIA T-grid policy={cia_t_grid_policy} for {test_name}: {cia_t_grids_by_pair}"
        )
        for warning in cia_grid_warnings:
            print(f"[{spec_type}] WARNING: {warning}")
    elif cfg.INCLUDE_CIA:
        print(f"[{spec_type}] No active CIA pairs for {test_name}.")
    else:
        print(f"[{spec_type}] CIA generation disabled by config for {test_name}.")

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
            valid_indices = np.where((cfg.WNEDGES >= min_solar_wn) & (cfg.WNEDGES <= max_solar_wn))[0]
            
            if len(valid_indices) < 2:
                print("Warning: Solar range strictly contained very few edges. expanding to overlaps.")
                valid_indices = np.where((cfg.WNEDGES >= min_solar_wn) & (cfg.WNEDGES <= max_solar_wn))[0]

            if len(valid_indices) >= 2:
                target_wnedges = cfg.WNEDGES[valid_indices]
                print(f"Restricted to {len(target_wnedges)-1} bands based on Solar Data. Original bands: {len(cfg.WNEDGES)-1}")
            else:
                print("Warning: Solar data range does not overlap sufficiently with WNEDGES. Using full edges.")
                
        except Exception as e:
            print(f"Error reading solar file: {e}. Using full WNEDGES.")

    cia_band_coverage, bands_with_cia, _cia_required_gases_by_band = analyze_cia_band_coverage(
        target_wnedges,
        active_cia_tuples,
        relevant_cia_configs,
    )

    # Calculate BLOCK 4 band occupancy using the TARGET edges.
    # CIA is excluded here on purpose; it is handled separately in BLOCK 18/19.
    if spec_type == 'sw' or cfg.ULTRA_HOT_ATMOSPHERE:
        include_uv = True
    else:
        include_uv = False
    band_gas_map, missing_gas_bands, empty_band_numbers = calculate_band_occupancy(
        target_wnedges,
        selected_gases_list,
        include_uv=include_uv,
    )
    
    with open(file_path, 'w') as f:
        # --- Imports ---
        f.write("import os\n")
        f.write("import json\n")
        f.write("import numpy as np\n")
        f.write("import re\n")
        f.write("import shutil\n")
        f.write("import subprocess\n")
        f.write("from numpy import array\n") 
        f.write("import netCDF4\n")
        f.write("from typing import Literal\n")
        f.write("import sys\n")
        f.write(f"sys.path.append('{cfg.ROOT_DIR}')\n") 
        f.write("from util.tools import read_wnedges, generate_LBL_from_ExoMol_hdf5, find_index, fix_socrates_nan, check_absorption_mismatches\n\n")

        # --- Dynamic Configuration Injection ---
        f.write(f"# --- Configuration injected by Manager ---\n")
        f.write(f"test_name = '{test_name}'\n")
        f.write(f"Molecule_str_list = {molecules_str}\n")
        f.write(f"gas_id_list = {gas_ids}\n")
        f.write(f"spec_type = '{spec_type}'\n")
        f.write(f"star_name = '{cfg.STAR_NAME}'\n")
        f.write(f"root = '{cfg.ROOT_DIR}'\n")
        f.write(f"num_kterm = {cfg.NUM_KTERM}\n")
        f.write(f"cia_t_grid_policy = '{cia_t_grid_policy}'\n")
        f.write(f"cia_shared_t_grid = {cia_shared_t_grid}\n")
        f.write(f"cia_grid_warnings = {cia_grid_warnings}\n")
        f.write(f"include_cia = {cfg.INCLUDE_CIA}\n")
        f.write(f"include_solar_sed = {cfg.INCLUDE_SOLAR_SED}\n")
        f.write(f"ultra_hot_atmosphere = {cfg.ULTRA_HOT_ATMOSPHERE}\n")
        # add solar path
        f.write(f"solar_path = os.path.join(root, 'stellar_spectra', 'soc_in', '{cfg.STAR_NAME}')\n\n")
        
        # Inject computed maps and configs
        f.write(f"BAND_GAS_MAP = {band_gas_map}\n") 
        f.write(f"ACTIVE_CIA_TUPLES = {active_cia_tuples}\n")
        f.write(f"RELEVANT_CIA_CONFIGS = {relevant_cia_configs}\n")
        f.write(f"CIA_T_GRID_OVERRIDE_BY_PAIR = {forced_cia_t_grids}\n")
        f.write(f"CIA_T_GRIDS_BY_PAIR = {cia_t_grids_by_pair}\n")
        f.write(f"CIA_BAND_COVERAGE = {cia_band_coverage}\n")
        f.write(f"BANDS_WITH_CIA = {bands_with_cia}\n")
        f.write(f"MISSING_GAS_BANDS = {missing_gas_bands}\n")
        f.write(f"EMPTY_BAND_NUMBERS = {empty_band_numbers}\n")
        
        # Inject the (potentially filtered) WNEDGES
        f.write(f"wnedges = np.array({target_wnedges.tolist()})\n")
        f.write(f"band_num = len(wnedges)-1\n")
        f.write(f"outputfilename = f'sp_{{spec_type}}_b{{band_num}}_{{test_name}}'\n\n")
        
        f.write(f"# Full configuration for gases\n")
        f.write(f"GAS_CONFIGS = {current_gas_configs}\n\n")

        # --- Worker Logic ---
        f.write(r"""
update_library = True
output_path_list = []
mon_path_list = []
LbL_path_list = []
output_path_xuv_list = []
generated_cia_files = []

final_dir = os.path.join(root, f'spectral_files/sp_b{band_num}')
outputfile = os.path.join(final_dir, outputfilename)
outputfile_k = f"{outputfile}_k"
qa_summary_path = os.path.join(final_dir, f"{outputfilename}_qa_summary.json")

qa_summary = {
    'selected_gases': list(Molecule_str_list),
    'gas_ids': list(gas_id_list),
    'cia_t_grid_policy': cia_t_grid_policy,
    'cia_shared_t_grid': cia_shared_t_grid,
    'cia_grid_warnings': cia_grid_warnings,
    'active_cia_pairs': [
        {'pair_name': pair_name, 'gas_ids': [id1, id2]}
        for id1, id2, pair_name in ACTIVE_CIA_TUPLES
    ],
    'cia_t_grids_by_pair': {
        pair_name: [float(value) for value in grid]
        for pair_name, grid in CIA_T_GRIDS_BY_PAIR.items()
    },
    'cia_t_grid_override_by_pair': CIA_T_GRID_OVERRIDE_BY_PAIR,
    'per_pair_band_coverage': CIA_BAND_COVERAGE,
    'bands_with_cia': BANDS_WITH_CIA,
    'missing_gas_bands': MISSING_GAS_BANDS,
    'empty_band_numbers': EMPTY_BAND_NUMBERS,
}

BLOCK_TYPE_PATTERN = re.compile(r'^\*BLOCK:\s+TYPE =\s*(\d+)\s*:')
CONTINUUM_ENTRY_PATTERN = re.compile(r'^Band:\s*(\d+), continuum:\s*(\d+), k-terms:\s*(\d+)\s*$')


def remove_if_exists(path):
    if os.path.exists(path):
        os.remove(path)


def ensure_nonempty_file(path, description):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing {description}: {path}")
    if os.path.getsize(path) <= 0:
        raise ValueError(f"Empty {description}: {path}")


def ensure_nonempty_files(paths, description):
    for path in paths:
        ensure_nonempty_file(path, description)


def run_generated_script(script_path, description):
    print(f"[RUN] {description}: {script_path}")
    with open(script_path, 'r', encoding='utf-8') as handle:
        print(handle.read())
    try:
        subprocess.run(['bash', script_path], check=True)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"{description} failed with exit code {exc.returncode}: {script_path}") from exc
    remove_if_exists(script_path)


def parse_int_tokens(line):
    return [int(token) for token in line.split() if re.fullmatch(r'-?\d+', token)]


def parse_float_tokens(line):
    values = []
    for token in line.split():
        try:
            values.append(float(token.replace('D', 'E').replace('d', 'e')))
        except ValueError:
            continue
    return values


def extract_typed_block(lines, block_type):
    start = None
    for idx, line in enumerate(lines):
        match = BLOCK_TYPE_PATTERN.match(line)
        if match and int(match.group(1)) == block_type:
            start = idx + 1
            break
    if start is None:
        raise ValueError(f"BLOCK {block_type} not found")
    for idx in range(start, len(lines)):
        if lines[idx].strip() == '*END':
            return lines[start:idx]
    raise ValueError(f"BLOCK {block_type} is missing *END")


def extract_named_block(lines, block_header):
    start = None
    for idx, line in enumerate(lines):
        if line.strip() == block_header:
            start = idx + 1
            break
    if start is None:
        raise ValueError(f"{block_header} not found")
    for idx in range(start, len(lines)):
        if lines[idx].strip() == '*END':
            return lines[start:idx]
    raise ValueError(f"{block_header} is missing *END")


def parse_block0_summary(sp_path):
    lines = open(sp_path, 'r', encoding='utf-8').read().splitlines()
    block = extract_typed_block(lines, 0)
    absorber_index_to_gas_id = {}
    continuum_index_to_pair = {}
    in_absorber_section = False
    in_continuum_section = False

    for line in block:
        stripped = line.strip()
        if stripped.startswith('Index       Absorber'):
            in_absorber_section = True
            in_continuum_section = False
            continue
        if stripped.startswith('Listing of continuum indexing numbers and gases.'):
            in_absorber_section = False
            continue
        if stripped.startswith('Index     Gas 1'):
            in_continuum_section = True
            continue

        numeric_tokens = parse_int_tokens(line)
        if in_absorber_section and len(numeric_tokens) >= 2:
            absorber_index_to_gas_id[numeric_tokens[0]] = str(numeric_tokens[1])
        elif in_continuum_section and len(numeric_tokens) >= 3:
            continuum_index = numeric_tokens[0]
            gas1_index = numeric_tokens[1]
            gas2_index = numeric_tokens[2]
            if gas1_index not in absorber_index_to_gas_id or gas2_index not in absorber_index_to_gas_id:
                raise ValueError("Continuum block references an undefined absorber index")
            continuum_index_to_pair[continuum_index] = (
                absorber_index_to_gas_id[gas1_index],
                absorber_index_to_gas_id[gas2_index],
            )

    return {
        'absorber_index_to_gas_id': absorber_index_to_gas_id,
        'continuum_index_to_pair': continuum_index_to_pair,
    }


def parse_block18_active_continua(sp_path):
    lines = open(sp_path, 'r', encoding='utf-8').read().splitlines()
    block = extract_typed_block(lines, 18)
    band_to_continua = {}
    active_continua = set()
    idx = 0

    while idx < len(block):
        line = block[idx]
        if re.match(r'^\s*\d+\s+\d+\s+[TF]\s*$', line):
            numeric_tokens = parse_int_tokens(line)
            band_number = numeric_tokens[0]
            n_active = numeric_tokens[1]
            idx += 1
            continua = []
            while len(continua) < n_active and idx < len(block):
                if block[idx].strip():
                    continua.extend(parse_int_tokens(block[idx]))
                idx += 1
            continua = continua[:n_active]
            band_to_continua[band_number] = continua
            active_continua.update(continua)
            continue
        idx += 1

    return {
        'band_to_continua': band_to_continua,
        'active_continua': sorted(active_continua),
    }


def parse_block19_entries(sp_path):
    lines = open(sp_path, 'r', encoding='utf-8').read().splitlines()
    block = extract_typed_block(lines, 19)
    n_t_lookup = None
    entries = []
    idx = 0

    while idx < len(block):
        line = block[idx]
        if 'Number of temperatures in look-up table' in line:
            numeric_tokens = parse_int_tokens(line)
            if not numeric_tokens:
                raise ValueError("Could not parse BLOCK 19 lookup temperature count")
            n_t_lookup = numeric_tokens[-1]
        elif re.match(r'^\s*\d+\s+\d+\s+\d+\s+-?\d+\s*$', line):
            numeric_tokens = parse_int_tokens(line)
            band_number, continuum_index, n_terms, overlap = numeric_tokens[:4]
            term_rows = []
            for _ in range(n_terms):
                idx += 1
                while idx < len(block) and not block[idx].strip():
                    idx += 1
                if idx >= len(block):
                    raise ValueError("BLOCK 19 ended before all k-term rows were read")
                term_rows.append(block[idx].strip())
            entries.append({
                'band': band_number,
                'continuum_index': continuum_index,
                'kterms': n_terms,
                'overlap': overlap,
                'term_rows': term_rows,
            })
        idx += 1

    if n_t_lookup is None:
        raise ValueError("BLOCK 19 temperature count is missing")

    return {
        'n_t_lookup': n_t_lookup,
        'entries': entries,
        'active_continua': sorted({entry['continuum_index'] for entry in entries}),
    }


def parse_continuum_k_table(spk_path):
    lines = open(spk_path, 'r', encoding='utf-8').read().splitlines()
    block = extract_named_block(lines, '*BLOCK: continuum k-table')
    declared_n_t = None
    temperatures = []
    entries = []
    idx = 0

    while idx < len(block):
        line = block[idx].strip()
        if not line:
            idx += 1
            continue
        if line.startswith('Lookup table:'):
            match = re.search(r'Lookup table:\s*(\d+)\s+temperatures', line)
            if not match:
                raise ValueError("Could not parse continuum k-table temperature count")
            declared_n_t = int(match.group(1))
            idx += 1
            while idx < len(block) and block[idx].strip():
                temperatures.extend(parse_float_tokens(block[idx]))
                idx += 1
            break
        idx += 1

    if declared_n_t is None:
        raise ValueError("continuum k-table header is missing lookup temperature count")
    if len(temperatures) != declared_n_t:
        raise ValueError(
            f"continuum k-table temperature header declares {declared_n_t} values but contains {len(temperatures)}"
        )

    while idx < len(block):
        line = block[idx].strip()
        if not line:
            idx += 1
            continue
        match = CONTINUUM_ENTRY_PATTERN.match(line)
        if not match:
            raise ValueError(f"Unexpected continuum k-table line: {line}")

        band_number = int(match.group(1))
        continuum_index = int(match.group(2))
        n_terms = int(match.group(3))
        idx += 1
        rows = []

        for term_idx in range(n_terms):
            row_values = []
            while idx < len(block):
                current_line = block[idx].strip()
                if not current_line:
                    if row_values:
                        raise ValueError(
                            f"Band {band_number}, continuum {continuum_index}, term {term_idx + 1} "
                            "ended before reaching the declared temperature count"
                        )
                    idx += 1
                    continue
                if CONTINUUM_ENTRY_PATTERN.match(current_line):
                    raise ValueError(
                        f"Band {band_number}, continuum {continuum_index}, term {term_idx + 1} "
                        "is missing lookup values before the next entry header"
                    )

                numeric_values = parse_float_tokens(current_line)
                if not numeric_values:
                    raise ValueError(f"Malformed continuum k-table data line: {current_line}")
                row_values.extend(numeric_values)
                idx += 1

                if len(row_values) == declared_n_t:
                    break
                if len(row_values) > declared_n_t:
                    raise ValueError(
                        f"Band {band_number}, continuum {continuum_index}, term {term_idx + 1} "
                        f"contains {len(row_values)} values but the header declares {declared_n_t}"
                    )

            if len(row_values) != declared_n_t:
                raise ValueError(
                    f"Band {band_number}, continuum {continuum_index}, term {term_idx + 1} "
                    f"contains {len(row_values)} values but the header declares {declared_n_t}"
                )
            rows.append(row_values)

        entries.append({
            'band': band_number,
            'continuum_index': continuum_index,
            'kterms': n_terms,
            'rows': rows,
        })

    return {
        'declared_n_t': declared_n_t,
        'temperatures': temperatures,
        'entries': entries,
    }


def normalize_pair(gas_id_1, gas_id_2):
    return tuple(sorted((str(gas_id_1), str(gas_id_2))))


def validate_final_outputs(sp_path, spk_path):
    messages = []
    mismatch_count = 0
    mismatch_preview = []

    ensure_nonempty_file(sp_path, 'final spectral file')
    ensure_nonempty_file(spk_path, 'final spectral k-table file')

    try:
        if ACTIVE_CIA_TUPLES:
            block0 = parse_block0_summary(sp_path)
            block18 = parse_block18_active_continua(sp_path)
            block19 = parse_block19_entries(sp_path)
            continuum_k = parse_continuum_k_table(spk_path)

            if block19['n_t_lookup'] != continuum_k['declared_n_t']:
                messages.append(
                    f"BLOCK 19 declares {block19['n_t_lookup']} temperatures but continuum k-table "
                    f"declares {continuum_k['declared_n_t']}."
                )

            if continuum_k['declared_n_t'] != len(continuum_k['temperatures']):
                messages.append(
                    f"continuum k-table header declares {continuum_k['declared_n_t']} temperatures but "
                    f"contains {len(continuum_k['temperatures'])} values."
                )

            block19_signatures = {
                (entry['band'], entry['continuum_index'], entry['kterms'])
                for entry in block19['entries']
            }
            continuum_k_signatures = {
                (entry['band'], entry['continuum_index'], entry['kterms'])
                for entry in continuum_k['entries']
            }
            if block19_signatures != continuum_k_signatures:
                messages.append("BLOCK 19 entries do not match continuum k-table entries.")

            if block18['active_continua'] != block19['active_continua']:
                messages.append(
                    f"Active continua differ between BLOCK 18 ({block18['active_continua']}) and "
                    f"BLOCK 19 ({block19['active_continua']})."
                )

            expected_pairs = sorted({
                normalize_pair(id1, id2)
                for id1, id2, _ in ACTIVE_CIA_TUPLES
            })
            actual_pairs = sorted({
                normalize_pair(*block0['continuum_index_to_pair'][continuum_index])
                for continuum_index in block19['active_continua']
            })
            if actual_pairs != expected_pairs:
                messages.append(
                    f"Active continuum pairs in the final spectral file ({actual_pairs}) do not match "
                    f"the configured active CIA pairs ({expected_pairs})."
                )
    except Exception as exc:
        messages.append(f"Structural continuum validation failed: {exc}")

    try:
        mismatch_results = check_absorption_mismatches(sp_path)
        mismatch_count = len(mismatch_results)
        mismatch_preview = [str(item) for item in mismatch_results[:5]]
        if mismatch_count > 0:
            messages.append(f"Absorption mismatch checker reported {mismatch_count} issue(s).")
    except Exception as exc:
        messages.append(f"Absorption mismatch checker failed: {exc}")

    return {
        'passed': not messages,
        'messages': messages,
        'absorption_mismatch_count': mismatch_count,
        'absorption_mismatch_preview': mismatch_preview,
    }


def write_qa_summary():
    os.makedirs(final_dir, exist_ok=True)
    with open(qa_summary_path, 'w', encoding='utf-8') as handle:
        json.dump(qa_summary, handle, indent=2, sort_keys=True)
        handle.write('\n')


remove_if_exists(outputfile)
remove_if_exists(outputfile_k)
remove_if_exists(qa_summary_path)

print(f"Using CIA T-grids by pair: {CIA_T_GRIDS_BY_PAIR}")
print(f"CIA T-grid policy: {cia_t_grid_policy}")
if CIA_T_GRID_OVERRIDE_BY_PAIR:
    print(f"CIA_T_GRID_OVERRIDE_BY_PAIR applied: {CIA_T_GRID_OVERRIDE_BY_PAIR}")

# 1. Processing xsec tables and generating PT files
for config in GAS_CONFIGS:
    Molecule_str = config['molecule']
    gas_id = config['gas_id']
    path_in = config['gas_abs_config']['hdf5_rel_path']
    datasource = config['gas_abs_config']['datasource']

    hdf5_path = os.path.join(root, path_in)
    ensure_nonempty_file(hdf5_path, f"LBL source for {Molecule_str}")
    print(f"Processing xsec tables for {Molecule_str}...")

    output_path, mon_path, LbL_path, T_grid, P_grid = generate_LBL_from_ExoMol_hdf5(
        root, hdf5_path, Molecule_str, datasource, update_library, test_name
    )
    output_path_list.append(output_path)
    mon_path_list.append(mon_path)
    LbL_path_list.append(LbL_path)

    pt_file_path = os.path.join(root, f'block5/pt_file_{test_name}_{gas_id}')
    remove_if_exists(pt_file_path)
    with open(pt_file_path, "w", encoding='utf-8') as file:
        file.write('*PTVAL\n')
        for P in P_grid:
            file.write(str(P * 1e+5))
            for T in T_grid:
                file.write(f' {T}')
            file.write('\n')
        file.write('*END')
    ensure_nonempty_file(pt_file_path, f"PT file for gas {gas_id}")

# 2. Reference PT file
ref_pt_path = os.path.join(root, f'block5/ref_pt_file_{test_name}')
remove_if_exists(ref_pt_path)
with open(ref_pt_path, "w", encoding='utf-8') as file:
    for gas_id in gas_id_list:
        file.write(f'*REF 1 {band_num} {gas_id} 1e+5 300.0\n')
    file.write('*END')
ensure_nonempty_file(ref_pt_path, 'reference PT file')

# 3. Generate skeleton spectral file using prep_spec
exec_file_name = f"sp_exec{test_name}.sh"
remove_if_exists(exec_file_name)
skeleton_file_name = f'sp_b{band_num}_{test_name}'
remove_if_exists(skeleton_file_name)

with open(exec_file_name, "w", encoding='utf-8') as f:
    f.write('prep_spec <<EOF\n')
    f.write(f'{skeleton_file_name}\n')
    f.write(f'{band_num}\n')
    f.write(f'{len(gas_id_list)}\n')
    for gas_id in gas_id_list:
        f.write(f'{gas_id}\n')

    if len(ACTIVE_CIA_TUPLES) > 0:
        f.write(f'{len(ACTIVE_CIA_TUPLES)}\n')
        for id1, id2, _ in ACTIVE_CIA_TUPLES:
            f.write(f'{id1} {id2}\n')
    else:
        f.write('0\n')

    f.write('0\n')
    f.write('c\n')
    f.write(f'{wnedges[0]} ')
    for band in wnedges[1:-1]:
        f.write(f'{band}\n')
        f.write(f'{band} ')
    f.write(f'{wnedges[-1]}\n')

    for gases_in_band in BAND_GAS_MAP:
        if len(gases_in_band) > 0:
            f.write(' '.join(gases_in_band) + '\n')
        else:
            f.write('0\n')

    for _ in wnedges[:-1]:
        f.write('0\n')

    f.write('n\n')
    f.write('-1\n')
    f.write('EOF\n')

os.chmod(exec_file_name, 0o777)
run_generated_script(exec_file_name, 'prep_spec skeleton generation')
ensure_nonempty_file(skeleton_file_name, 'skeleton spectral file')

# 4. Generate corrk data for gases
for config, output_path, mon_path, LbL_path in zip(GAS_CONFIGS, output_path_list, mon_path_list, LbL_path_list):
    gas_id = config['gas_id']
    lower = config['gas_abs_config']['lower_wn']
    upper = config['gas_abs_config']['upper_wn']

    print(f"Running corr_k for Gas ID {gas_id}...")
    exec_file_corrk = f"corr_k_ExoMol_{test_name}_{gas_id}.sh"
    remove_if_exists(exec_file_corrk)

    with open(exec_file_corrk, "w", encoding='utf-8') as f:
        idx_lower, idx_upper = find_index(wnedges[:-1], wnedges[1:], lower, upper, strict_band_edges=True)
        f.write('Ccorr_k ')
        f.write(f'-s {skeleton_file_name} ')
        f.write(f'-R {idx_lower} {idx_upper} ')
        f.write(f'-F {root}/block5/pt_file_{test_name}_{gas_id} ')
        f.write(f'-r {root}/block5/ref_pt_file_{test_name} ')
        f.write(f'-l {gas_id} 1.0e5 ')
        f.write(f'-n {num_kterm} ')
        f.write('-lk ')
        if spec_type == 'sw':
            f.write(f'+S {solar_path} ')
        else:
            f.write('+p ')
        f.write(f'-o {output_path} ')
        f.write(f'-m {mon_path} ')
        f.write(f'-L {LbL_path} ')
        f.write('-np 1\n')

    os.chmod(exec_file_corrk, 0o777)
    run_generated_script(exec_file_corrk, f'gas corr_k generation for {gas_id}')
    ensure_nonempty_file(output_path, f'gas corr_k output for {gas_id}')

ensure_nonempty_files(output_path_list, 'gas corr_k outputs')

# 5. Include CIA (Hitran)
if include_cia and len(ACTIVE_CIA_TUPLES) > 0:
    for id1, id2, pair_name in ACTIVE_CIA_TUPLES:
        cia_conf = RELEVANT_CIA_CONFIGS.get(pair_name)
        if not cia_conf:
            raise ValueError(f"Missing CIA configuration for pair {pair_name}")

        T_cia_grid = np.array(CIA_T_GRIDS_BY_PAIR[pair_name], dtype=float)
        P_cia_grid = np.array(cia_conf.get('p_grid', [1.0]), dtype=float)

        pt_cia_path = os.path.join(root, f'block19/pt_cia_{pair_name}_{test_name}')
        remove_if_exists(pt_cia_path)
        with open(pt_cia_path, "w", encoding='utf-8') as pt_file:
            pt_file.write('*PTVAL\n')
            for P_0 in P_cia_grid:
                pt_file.write(f'{P_0 * 1e+5}')
                for T in T_cia_grid:
                    pt_file.write(f' {T}')
                pt_file.write('\n')
            pt_file.write('*END')
        ensure_nonempty_file(pt_cia_path, f"CIA PT file for {pair_name}")

        cia_file_path = os.path.join(root, cia_conf['cia_rel_path'], cia_conf['cia_file'])
        ensure_nonempty_file(cia_file_path, f"CIA input file for {pair_name}")

        cia_lowers = cia_conf['lower_wn']
        cia_uppers = cia_conf['upper_wn']
        if not isinstance(cia_lowers, list):
            cia_lowers = [cia_lowers]
        if not isinstance(cia_uppers, list):
            cia_uppers = [cia_uppers]

        for seg_idx, (seg_lower, seg_upper) in enumerate(zip(cia_lowers, cia_uppers)):
            cia_out_base = f"output_CIA_{pair_name}_{test_name}_seg{seg_idx}"
            full_cia_out_path = f"{root}/block19/{cia_out_base}"
            monitoring_cia_path = f"{root}/block19/monitoring_CIA_{pair_name}_{test_name}_seg{seg_idx}"
            lbl_cia_path = f"{root}/block19/LBL_CIA_{pair_name}_{test_name}_seg{seg_idx}.nc"

            generated_cia_files.append(full_cia_out_path)

            idx_lower, idx_upper = find_index(wnedges[:-1], wnedges[1:], float(seg_lower), float(seg_upper))

            remove_if_exists(full_cia_out_path)
            remove_if_exists(full_cia_out_path + '.nc')
            remove_if_exists(monitoring_cia_path)
            remove_if_exists(lbl_cia_path)

            exec_file_CIA_seg = f"corr_k_CIA_{pair_name}_{test_name}_seg{seg_idx}.sh"
            remove_if_exists(exec_file_CIA_seg)

            with open(exec_file_CIA_seg, "w", encoding='utf-8') as f:
                f.write(f'Ccorr_k -CIA {cia_file_path} -R {idx_lower} {idx_upper} ')
                f.write(f'-F {pt_cia_path} -ct {id1} {id2} 1000.0 -i 1.0 -t 1.0e-2 ')
                f.write(f'-s {skeleton_file_name} ')
                if spec_type == 'sw':
                    f.write(f'+S {solar_path} ')
                else:
                    f.write('+p ')
                f.write('-lk ')
                f.write(f'-o {full_cia_out_path} ')
                f.write(f'-m {monitoring_cia_path} ')
                f.write(f'-L {lbl_cia_path}\n')

            os.chmod(exec_file_CIA_seg, 0o777)
            print(f"Running corr_k for CIA {pair_name} segment {seg_idx}...")
            run_generated_script(exec_file_CIA_seg, f'CIA corr_k generation for {pair_name} segment {seg_idx}')
            ensure_nonempty_file(full_cia_out_path, f'CIA corr_k output for {pair_name} segment {seg_idx}')

ensure_nonempty_files(generated_cia_files, 'CIA corr_k outputs')

# 6. Add UV lines to SW files
if spec_type == 'sw' or ultra_hot_atmosphere:
    for config in GAS_CONFIGS:
        uv_conf = config['uv_config']
        if uv_conf is None:
            continue

        gas_id = config['gas_id']
        Molecule_str = config['molecule']
        UV_gas = uv_conf['xuv_file']
        lower = uv_conf['lower_wn']
        upper = uv_conf['upper_wn']
        uv_source_path = os.path.join(root, uv_conf['xuv_rel_path'], UV_gas)
        ensure_nonempty_file(uv_source_path, f'UV input file for {Molecule_str}')

        pt_uv_path = os.path.join(root, 'block5', f'pt_uv_{Molecule_str}_{test_name}')
        remove_if_exists(pt_uv_path)
        if 't_grid' not in uv_conf:
            raise ValueError(f"No t_grid found for UV config of {Molecule_str}")
        if 'p_grid' not in uv_conf:
            raise ValueError(f"No p_grid found for UV config of {Molecule_str}")
        T_uv_grid = np.array(uv_conf['t_grid'])
        P_uv_grid = np.array(uv_conf['p_grid'])

        with open(pt_uv_path, "w", encoding='utf-8') as pt_file:
            pt_file.write('*PTVAL\n')
            for P_0 in P_uv_grid:
                pt_file.write(f'{P_0 * 1e+5}')
                for T in T_uv_grid:
                    pt_file.write(f' {T}')
                pt_file.write('\n')
            pt_file.write('*END')
        ensure_nonempty_file(pt_uv_path, f'UV PT file for {Molecule_str}')

        exec_file_corrk_xuv = f"corr_k_ExoMol_{test_name}_{gas_id}_xuv.sh"
        remove_if_exists(exec_file_corrk_xuv)

        output_path_xuv = os.path.join(root, f'block5/output_xuv_{Molecule_str}_{test_name}')
        mon_path_xuv = os.path.join(root, f'block5/monitoring_xuv_{Molecule_str}_{test_name}')
        LBL_path_xuv = os.path.join(root, f'block5/LBL_xuv_{Molecule_str}_{test_name}.nc')

        remove_if_exists(output_path_xuv)
        remove_if_exists(output_path_xuv + '.nc')
        remove_if_exists(mon_path_xuv)
        remove_if_exists(LBL_path_xuv)

        with open(exec_file_corrk_xuv, "w", encoding='utf-8') as f:
            f.write(f'Ccorr_k -s {skeleton_file_name} ')
            f.write(f'-UVX {uv_source_path} ')
            idx_lower, idx_upper = find_index(wnedges[:-1], wnedges[1:], lower, upper)
            f.write(f'-R {idx_lower} {idx_upper} ')
            f.write(f'-F {pt_uv_path} ')
            f.write('-i 1.0 ')
            f.write(f'-l {gas_id} 1.0e1 ')
            f.write('-t 1.0e-3 ')
            if spec_type == 'sw':
                f.write(f'+S {solar_path} ')
            else:
                f.write('+p ')
            f.write(f'-o {output_path_xuv} -m {mon_path_xuv} -L {LBL_path_xuv} -np 1\n')

        os.chmod(exec_file_corrk_xuv, 0o777)
        run_generated_script(exec_file_corrk_xuv, f'UV corr_k generation for {Molecule_str}')
        ensure_nonempty_file(output_path_xuv, f'UV corr_k output for {Molecule_str}')
        output_path_xuv_list.append(output_path_xuv)

ensure_nonempty_files(output_path_xuv_list, 'UV corr_k outputs')

# 7. Final Assembly: Fill in Spectral File
exec_file_sp = f"fill_in_sp_{test_name}.sh"
remove_if_exists(exec_file_sp)

ensure_nonempty_files(output_path_list, 'gas corr_k outputs')
ensure_nonempty_files(output_path_xuv_list, 'UV corr_k outputs')
ensure_nonempty_files(generated_cia_files, 'CIA corr_k outputs')

with open(exec_file_sp, "w", encoding='utf-8') as f:
    f.write('prep_spec <<EOF\n')
    f.write(f'{skeleton_file_name}\n')
    f.write('a\n')

    if spec_type == 'sw' or ultra_hot_atmosphere:
        f.write('3\n')
        f.write(f'{solar_path}\n')
        f.write('C\n')
        f.write('A\n')

    f.write('5\n')
    f.write(f'{output_path_list[0]}\n')
    for output_path in output_path_list[1:]:
        f.write('5\n')
        f.write('y\n')
        f.write(f'{output_path}\n')

    if spec_type == 'sw' or ultra_hot_atmosphere:
        for output_path_xuv in output_path_xuv_list:
            f.write('5\n')
            f.write('y\n')
            f.write(f'{output_path_xuv}\n')

    if spec_type == 'lw':
        f.write('6\n')
        f.write('n\n')
        f.write('T\n')
        f.write('50 4500\n')
        f.write('445\n')

    if include_solar_sed and spec_type == 'sw':
        f.write('2\n')
        f.write('n\n')
        f.write('y\n')

    if include_cia and len(generated_cia_files) > 0:
        f.write('19\n')
        f.write(f'{generated_cia_files[0]}\n')
        for path in generated_cia_files[1:]:
            f.write('19\n')
            f.write('y\n')
            f.write(f'{path}\n')

    f.write('-1\n')
    f.write('EOF\n')

os.chmod(exec_file_sp, 0o777)
run_generated_script(exec_file_sp, 'final spectral assembly')

# 8. Move files
os.makedirs(final_dir, exist_ok=True)
source_output = f'sp_b{band_num}_{test_name}'
source_output_k = f'{source_output}_k'
ensure_nonempty_file(source_output, 'assembled spectral file')
ensure_nonempty_file(source_output_k, 'assembled spectral k-table file')

print(f"Moving {outputfilename} to {final_dir}")
shutil.move(source_output, outputfile)
shutil.move(source_output_k, outputfile_k)

# 9. Fix NaNs and validate final outputs
fix_socrates_nan(outputfile)
fix_socrates_nan(outputfile_k)

validation_result = validate_final_outputs(outputfile, outputfile_k)
qa_summary['validator'] = validation_result
write_qa_summary()

if not validation_result['passed']:
    raise RuntimeError(
        "Final spectral validation failed: " + "; ".join(validation_result['messages'])
    )
""")

# Additional: modify H2O
# Changed: Added job_dir parameter
def modify_h2o_to_cfc113(job_dir, filename):
    file_path = os.path.join(job_dir, filename)
    # 使用原始多行字符串 (r"")，这样正则表达式里的 \s, \b 等就不会被转义
    append_script = r"""

# Post-processing: Modify H2O to CFC113 in the final spectral file
import re

if os.path.exists(outputfile):
    with open(outputfile, 'r', encoding='utf-8') as f:
        content = f.read()

    # 1. 查找 "1 [若干空格] Water Vapour"，将 1 改为 16，Water Vapour 改为 CFC113，并保留中间的空格 (\1 代表第一个括号匹配到的空格)
    content = re.sub(r'\b1(\s+)Water Vapour', r'16\1CFC113', content)
    
    # 2. 将文件中其他所有的 "Water Vapour" 修改成 "CFC113"
    content = content.replace("Water Vapour", "CFC113")

    # 3. 将 "Index of water =     1" 修改为 "Index of water =     0"
    # 使用正则 \s* 兼容等号两边可能出现的不定数量的空格，\g<1> 代表保留前面的 "Index of water = [空格]" 部分
    content = re.sub(r'(Index of water\s*=\s*)1\b', r'\g<1>0', content)

    # 将修改后的内容写回文件
    with open(outputfile, 'w', encoding='utf-8') as f:
        f.write(content)

    fix_socrates_nan(outputfile)
    validation_result = validate_final_outputs(outputfile, outputfile_k)
    qa_summary['post_processing'] = {
        'h2o_to_cfc113_applied': True,
        'validator': validation_result,
    }
    qa_summary['validator'] = validation_result
    write_qa_summary()

    if not validation_result['passed']:
        raise RuntimeError(
            "Final spectral validation failed after H2O->CFC113 post-processing: "
            + "; ".join(validation_result['messages'])
        )
else:
    print(f"File not found: {outputfile}")
"""

    # 以追加模式('a')打开目标 .py 文件，并将代码写入到文件末尾
    with open(file_path, 'a', encoding='utf-8') as f:
        # 写入前先加两行换行，确保不会和原文件最后一行代码粘连
        f.write('\n\n' + append_script)
    print("Warning: Post-processing code to modify H2O to CFC113 has been appended to the worker script. Please ensure CFC113 in gas_list_pcf.F90 has been modified.")

# ==========================================
# 3. Slurm Script Generator 
# ==========================================
def get_file_size_in_gb(file_path):
    try:
        size_bytes = os.path.getsize(file_path)
        size_gb = size_bytes / (1024 * 1024 * 1024)
        return size_gb
    except OSError as e:
        print(f"Error: {e}")
        return None

# Changed: Added job_dir parameter
def write_slurm_script(job_dir, job_name, case_name_list, gas_lbl_file_list):
    slurm_path = os.path.join(job_dir, f'{job_name}.sh')
    cache_size_gb = 0.0
    for gas_lbl_file in gas_lbl_file_list:
        full_path = os.path.join(cfg.ROOT_DIR, gas_lbl_file)
        size_gb = get_file_size_in_gb(full_path)
        cache_size_gb += size_gb if size_gb is not None else 0.0
    gb_per_core = 2 # Wuzhen cluster
    ncores = int(np.ceil(cache_size_gb / gb_per_core))+1
    with open(slurm_path, 'w') as f:
        f.write('#!/bin/bash\n')
        f.write(f'#SBATCH --job-name={job_name}\n')
        f.write(f'#SBATCH --output=output_%j.out\n')
        f.write(f'#SBATCH --error=error_%j.err\n')
        f.write(f'#SBATCH -N 1\n')
        f.write(f'#SBATCH -c {ncores}\n')
        f.write(f'#SBATCH --partition=wzhcnormal\n\n')
        
        # Changed: switch into the specific job_identifier directory
        f.write(f'cd {job_dir}\n')
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
    SELECTED_MOLECULES = cfg.MOLEULES_TO_INCLUDE
    
    # Validate selection against Gas Library
    for m in SELECTED_MOLECULES:
        if m not in cfg.GAS_LIBRARY:
            raise ValueError(f"Molecule {m} not found in config_data.GAS_LIBRARY")
            
    num_bands = len(cfg.WNEDGES) - 1
    test_name = cfg.TEST_NAME
    job_identifier = f"sp_b{num_bands}_{test_name}"
    
    # New: Create the specific sub-directory for this job
    job_dir = os.path.join(cfg.SLURM_DIR, job_identifier)
    os.makedirs(job_dir, exist_ok=True)
    
    print(f"Generating scripts for: {test_name}")
    print(f"Selected Gases: {SELECTED_MOLECULES}")
    print(f"Output Directory: {job_dir}")
    
    # Generate Python Worker Scripts
    write_worker_script(job_dir, f"{job_identifier}_lw.py", test_name, SELECTED_MOLECULES, 'lw')
    write_worker_script(job_dir, f"{job_identifier}_sw.py", test_name, SELECTED_MOLECULES, 'sw')
    
    # if 'H2O' in SELECTED_MOLECULES, modify H2O to CFC113
    if 'H2O' in SELECTED_MOLECULES:
        modify_h2o_to_cfc113(job_dir, f"{job_identifier}_lw.py")
        modify_h2o_to_cfc113(job_dir, f"{job_identifier}_sw.py")
        
    # Generate Slurm Submission Script
    case_name_list = [job_identifier]
    gas_lbl_file_list = []
    for gas in SELECTED_MOLECULES:
        gas_conf = cfg.GAS_LIBRARY[gas]
        gas_lbl_file_list.append(gas_conf['gas_abs_config']['hdf5_rel_path'])
        
    write_slurm_script(job_dir, job_identifier, case_name_list, gas_lbl_file_list)
    
    slurm_file = f'{job_identifier}.sh'
    os.chmod(os.path.join(job_dir, slurm_file), 0o755)
    
    # Changed: Terminal output to show the new working directory
    print(f"To submit job, run:\n cd {job_dir} && sbatch {slurm_file}")
