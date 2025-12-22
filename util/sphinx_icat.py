import os
import glob
import numpy as np
from astropy import units as u

# SYNPHOT / STSYNPHOT
from synphot import exceptions as synexceptions
from synphot import units
from synphot.spectrum import SourceSpectrum
from synphot.models import Empirical1D 
from synphot.utils import validate_totalflux
from stsynphot import exceptions, stio

# 设置缓存
_SPHINX_CACHE = {}
_PARAM_NAMES = ['T_eff', 'metallicity', 'log_g', 'C_to_O']

def reset_sphinx_cache():
    """Empty the sphinx grid cache."""
    global _SPHINX_CACHE
    _SPHINX_CACHE.clear()

def _par_from_parser(x):
    """Convert parser string to parameter value."""
    if not isinstance(x, (np.floating, float, int, u.Quantity)):
        x = float(x)
    return x

def _load_sphinx_indices(sphinx_dir):
    """
    Scan the directory and parse filenames to build the index list.
    Expects filename format: "Teff_3900.0_logg_4.25_logZ_-0.75_CtoO_0.3.txt"
    Returns list of [T_eff, Metallicity, log_g, C_to_O, filename]
    """
    if not os.path.exists(sphinx_dir):
        raise IOError(f"Sphinx grid directory not found: {sphinx_dir}")

    files = glob.glob(os.path.join(sphinx_dir, "*.txt"))
    if not files:
        raise IOError(f"No .txt files found in Sphinx grid directory: {sphinx_dir}")

    indices = []
    for fpath in files:
        fname = os.path.basename(fpath)
        try:
            # 解析文件名: Teff_Val_logg_Val_logZ_Val_CtoO_Val.txt
            clean_name = fname.replace('.txt', '')
            parts = clean_name.split('_')
            
            # 提取参数 (顺序必须对应 _break_list 中的 index)
            # 0: Teff, 1: logZ(Met), 2: logg, 3: CtoO
            t_val = float(parts[1])      # Teff
            g_val = float(parts[3])      # logg
            z_val = float(parts[5])      # logZ (Metallicity)
            c_val = float(parts[7])      # CtoO

            # 存入列表: [Teff, Metallicity, log_g, C_to_O, filename]
            indices.append([t_val, z_val, g_val, c_val, fname])
            
        except (IndexError, ValueError):
            print(f"Warning: Skipping malformed filename: {fname}")
            continue
    
    return indices

def get_sphinx_index(sphinx_dir):
    """
    Get or load the Sphinx grid index for a specific directory.
    Cache is keyed by the directory path.
    """
    global _SPHINX_CACHE
    
    # 如果该目录的索引不在缓存中，则加载
    if sphinx_dir not in _SPHINX_CACHE:
        indices = _load_sphinx_indices(sphinx_dir)
        _SPHINX_CACHE[sphinx_dir] = indices
    
    return _SPHINX_CACHE[sphinx_dir]

def _break_list(in_list, index, parameter):
    """
    Break input list into upper and lower lists for a specific dimension.
    index: 0=Teff, 1=Met, 2=logg, 3=CtoO
    """
    parameter = float(parameter)
    
    if not in_list:
        raise exceptions.ParameterOutOfBounds("Grid search failed. Grid may be discontinuous.")

    # 提取当前维度的所有值
    array = np.array([row[index] for row in in_list], dtype=np.float64)

    upper_array = array[array >= parameter]
    lower_array = array[array <= parameter]

    if upper_array.size == 0:
        raise exceptions.ParameterOutOfBounds(
            f"Parameter '{_PARAM_NAMES[index]}' exceeds data. "
            f"Max allowed={array.max()}, entered={parameter}.")

    if lower_array.size == 0:
        raise exceptions.ParameterOutOfBounds(
            f"Parameter '{_PARAM_NAMES[index]}' exceeds data. "
            f"Min allowed={array.min()}, entered={parameter}.")

    upper_bound = upper_array.min()
    lower_bound = lower_array.max()

    upper_list = []
    lower_list = []

    # 精确匹配浮点数，使用小 epsilon
    epsilon = 1e-7
    for row in in_list:
        val = row[index]
        if abs(val - upper_bound) < epsilon:
            upper_list.append(row)
        if abs(val - lower_bound) < epsilon:
            lower_list.append(row)

    return lower_list, upper_list

def _get_spectrum(parlist, sphinx_dir):
    """
    Load spectrum from .txt file with header.
    Requires sphinx_dir to locate the file.
    """
    # parlist: [Teff, Met, logg, CtoO, filename]
    filename = parlist[-1]
    fullpath = os.path.join(sphinx_dir, filename) # 使用传入的 sphinx_dir
    
    # 1. 使用 numpy 手动读取，跳过第一行 header
    try:
        data = np.loadtxt(fullpath, skiprows=1)
    except Exception as e:
        raise IOError(f"Failed to read file {filename} in {sphinx_dir}: {e}")

    # 获取原始数据 (假设两列结构固定)
    wave_um = data[:, 0]    # 微米
    flux_wm3 = data[:, 1]   # W/m^3

    # 2. 单位转换
    # 2.1 波长: 微米 -> 埃 (Angstrom)
    wave_ang = wave_um * 10000.0

    # 2.2 通量: W/m^3 -> FLAM (erg/s/cm^2/A)
    # 转换因子: 1e-7
    # 必须构建为 Astropy Quantity，否则 convert_flux 不知道这是 FLAM
    flux_flam_val = flux_wm3 * 1e-7
    flux_flam_qty = flux_flam_val * u.erg / (u.s * u.cm**2 * u.AA)

    # 3. 转换为 PHOTLAM
    # synphot.units.convert_flux(wavelength, flux, out_flux_unit)
    flux_photlam = units.convert_flux(wave_ang, flux_flam_qty, 'photlam')

    # 4. 构建 SourceSpectrum 对象
    sp = SourceSpectrum(Empirical1D, points=wave_ang, lookup_table=flux_photlam.value)

    # 验证通量有效性 (可选)
    totflux = sp.integrate()
    if totflux <= 0 or np.isnan(totflux):
         raise exceptions.ParameterOutOfBounds(
            f"Parameter '{parlist[:-1]}' yielded invalid spectrum (Flux <= 0 or NaN).")

    # 返回结构: [Teff, Met, logg, CtoO, SpectrumObject]
    result = list(parlist)
    result.pop() # 移除文件名
    result.append(sp) # 添加光谱对象
    
    return result

def _interpolate_spectrum(sp1_list, sp2_list, target_val):
    """
    Linearly interpolate between two spectrum objects based on target_val.
    Lists are [param1, param2, ..., Spectrum]
    """
    # 从列表尾部弹出：Spectrum 和 当前维度的 Parameter
    spectrum_low = sp1_list.pop()
    spectrum_high = sp2_list.pop()
    
    val_low = sp1_list.pop()
    val_high = sp2_list.pop()

    if val_low == val_high:
        sp = spectrum_low
    else:
        # 线性插值
        a = (val_low - target_val) / (val_low - val_high)
        b = 1.0 - a
        sp = a * spectrum_high + b * spectrum_low

    # 构建新列表
    result = list(sp1_list) # 剩余参数
    result.append(sp)       # 新光谱

    return result

def grid_to_spec_sphinx(t_eff, metallicity, log_g, c_to_o, sphinx_dir):
    """
    Extract spectrum from Sphinx grid (4 dimensions).
    
    Parameters
    ----------
    t_eff : float or Quantity
        Effective temperature (K).
    metallicity : float
        LogZ.
    log_g : float
        Log gravity.
    c_to_o : float
        Carbon to Oxygen ratio.
    sphinx_dir : str
        Path to the directory containing the Sphinx grid .txt files.
        
    Returns
    -------
    sp : synphot.spectrum.SourceSpectrum
        Interpolated spectrum.
    """
    # 1. 参数标准化
    t_eff = units.validate_quantity(_par_from_parser(t_eff), u.K).value
    metallicity = _par_from_parser(metallicity)
    log_g = _par_from_parser(log_g)
    c_to_o = _par_from_parser(c_to_o)

    # 2. 获取索引 (传入 sphinx_dir)
    indices = get_sphinx_index(sphinx_dir)

    # 3. 分层筛选 (4层)
    
    # Dimension 0: Teff
    list_t_low, list_t_high = _break_list(indices, 0, t_eff)

    # Dimension 1: Metallicity (logZ)
    list_m_ll, list_m_lh = _break_list(list_t_low, 1, metallicity)  # Teff Low 分支
    list_m_hl, list_m_hh = _break_list(list_t_high, 1, metallicity) # Teff High 分支

    # Dimension 2: log_g (4 pairs -> 8 lists)
    l_ll_g_l, l_ll_g_h = _break_list(list_m_ll, 2, log_g)
    l_lh_g_l, l_lh_g_h = _break_list(list_m_lh, 2, log_g)
    h_hl_g_l, h_hl_g_h = _break_list(list_m_hl, 2, log_g)
    h_hh_g_l, h_hh_g_h = _break_list(list_m_hh, 2, log_g)

    # Dimension 3: C_to_O (8 pairs -> 16 lists)
    # 命名: [Teff][Met][Logg][CtoO] (0=Low, 1=High)
    p0000, p0001 = _break_list(l_ll_g_l, 3, c_to_o)
    p0010, p0011 = _break_list(l_ll_g_h, 3, c_to_o)
    p0100, p0101 = _break_list(l_lh_g_l, 3, c_to_o)
    p0110, p0111 = _break_list(l_lh_g_h, 3, c_to_o)
    p1000, p1001 = _break_list(h_hl_g_l, 3, c_to_o)
    p1010, p1011 = _break_list(h_hl_g_h, 3, c_to_o)
    p1100, p1101 = _break_list(h_hh_g_l, 3, c_to_o)
    p1110, p1111 = _break_list(h_hh_g_h, 3, c_to_o)

    # 4. 加载光谱 (16个)
    spectra_params = [
        p0000, p0001, p0010, p0011, p0100, p0101, p0110, p0111,
        p1000, p1001, p1010, p1011, p1100, p1101, p1110, p1111
    ]
    
    loaded_spectra = []
    for p_list in spectra_params:
        if not p_list:
             raise exceptions.ParameterOutOfBounds(
                "Sphinx grid incomplete. Could not bracket parameters.")
        # 传递 sphinx_dir 到 _get_spectrum
        loaded_spectra.append(_get_spectrum(p_list[0], sphinx_dir))

    # 5. 插值 (Interpolation)
    
    # Layer 1: Interpolate C_to_O (16 -> 8)
    sp_layer1 = []
    for i in range(0, 16, 2):
        sp = _interpolate_spectrum(loaded_spectra[i], loaded_spectra[i+1], c_to_o)
        sp_layer1.append(sp)

    # Layer 2: Interpolate log_g (8 -> 4)
    sp_layer2 = []
    for i in range(0, 8, 2):
        sp = _interpolate_spectrum(sp_layer1[i], sp_layer1[i+1], log_g)
        sp_layer2.append(sp)

    # Layer 3: Interpolate Metallicity (4 -> 2)
    sp_layer3 = []
    for i in range(0, 4, 2):
        sp = _interpolate_spectrum(sp_layer2[i], sp_layer2[i+1], metallicity)
        sp_layer3.append(sp)

    # Layer 4: Interpolate Teff (2 -> 1)
    final_obj = _interpolate_spectrum(sp_layer3[0], sp_layer3[1], t_eff)

    sp = final_obj[0]
    
    # 添加元数据
    sp.meta['expr'] = (f'Sphinx(T_eff={t_eff:g}, metallicity={metallicity:g}, '
                       f'log_g={log_g:g}, C_to_O={c_to_o:g})')
    
    return sp