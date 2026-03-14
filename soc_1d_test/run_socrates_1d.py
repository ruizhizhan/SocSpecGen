import numpy as np
import matplotlib.pyplot as plt
import netCDF4 as nc
import os
import sys     
import re
from typing import Literal

root = f"/work/home/ac9b0k6rio/SocSpecGen/soc_1d_test"
os.chdir(root)
sys.path.append(os.path.dirname(root))
from util.tools import compute_gaussian
from read_write_socrates_netcdf import write_latlon, read_latlon
GCM_root = f'/work/home/ac9b0k6rio/isca_17/data'


def write_input(soc_base,file_type,ps,Ts,p,pmid,T,Tmid,
                co2_mmr = 0,co_mmr = 0,sio_mmr = 0,n2_mmr = 0,h2o_mmr = 0,real_h2o = 0,o2_mmr=0,
                stoa = 0, sza = 90.0):
    """
        Write input files for socrates offline 1d runs.
    """
    write_latlon('.', Tmid, soc_base, 'tl', 'tl', 'K','Temperature on levels', p_lev=pmid)
    write_latlon('.', T, soc_base, 't', 't', 'K','Temperature', p_lev=p)
    if file_type == 'lw':
        write_latlon('.', np.asarray([Ts]), soc_base, 'tstar', 'tstar', 'K','Surface temperature', p_lev=np.asarray([ps]))
    if file_type == 'sw':
        write_latlon('.', np.asarray([0.]), soc_base, 'sazim', 'sazim', 'degree','Solar azimuthal angle')
        write_latlon('.', np.asarray([stoa]), soc_base, 'stoa', 'stoa', 'W.m-2','Solar Irradiance')
        write_latlon('.', np.asarray([sza]), soc_base, 'szen', 'szen', 'Degrees','Solar zenith angle')
    write_latlon('.', 0., soc_base, 'surf', 'alb', 'None','Albedo weights', basis=np.array([1.])) 
    if co2_mmr > 0: 
        write_latlon('.',np.ones(p.shape)*co2_mmr, soc_base, 'co2','CO2', 'None', 'CO2 MMR', p_lev=p)
    if co_mmr > 0:
        write_latlon('.',np.ones(p.shape)*co_mmr, soc_base, 'co','CO', 'None', 'CO MMR', p_lev=p)
    if sio_mmr > 0:
        write_latlon('.',np.ones(p.shape)*sio_mmr, soc_base, 'hcfc22','HCFC22', 'None', 'HCFC22 MMR', p_lev=p)
    if n2_mmr > 0:
        write_latlon('.',np.ones(p.shape)*n2_mmr, soc_base, 'n2','N2', 'None', 'N2 MMR', p_lev=p)
    if h2o_mmr > 0:
        write_latlon('.',np.ones(p.shape)*h2o_mmr, soc_base, 'cfc113','CFC113', 'None', 'CFC113 MMR', p_lev=p)
    if o2_mmr > 0:
        write_latlon('.',np.ones(p.shape)*o2_mmr, soc_base, 'o2','O2', 'None', 'O2 MMR', p_lev=p)
    if isinstance(real_h2o, np.ndarray):
        write_latlon('.',real_h2o, soc_base, 'q','H2O', 'None', 'H2O MMR', p_lev=p)

def get_band_number(spectral_file_name):
    match = re.search(r'_b(\d+)', spectral_file_name)
    if match:
        band_str = match.group(1) 
        band_num = int(band_str)  
        return band_num
    else:
        print("未匹配到 band 数量")
### Functions to generate neccessary T-p input files.
# 1. artificial T-p profile
def adiabatic_profile(p_surf, p_toa, T_surf, T_strat, dlnT_dlnp=2/7, nlevel=40, option='log'):
    """
    计算大气绝热温度廓线 (包含等温平流层近似)
    
    参数:
        p_surf: 表面气压 (Surface pressure, 例如 100000 Pa)
        p_toa: 大气层顶气压 (Top of atmosphere pressure, 例如 100 Pa)
        T_surf: 表面温度 (Surface temperature, 例如 288 K)
        T_strat: 平流层温度下限 (Stratospheric temperature, 例如 200 K)
        dlnT_dlnp: 绝热指数 R/cp (对于干空气约为 2/7)
        nlevel: 垂直层数 (Number of vertical full levels)
        option: 垂直网格划分方式 ('log' 或 'linear')
    
    返回:
        ps, t_surf, p_full, p_half, t_full, t_half
    """
    
    # 1. 计算边界气压 (p_half) - 共有 nlevel + 1 层
    if option == 'log':
        # 在对数气压坐标下均匀分布
        p_half = np.logspace(np.log10(p_surf), np.log10(p_toa), nlevel + 1)
    elif option == 'linear':
        # 在线性气压坐标下均匀分布
        p_half = np.linspace(p_surf, p_toa, nlevel + 1)
    else:
        raise ValueError("参数 option 必须是 'log' 或 'linear'")
        
    # 2. 计算中心气压 (p_full) - 共有 nlevel 层
    if option == 'log':
        # 对数网格下，使用对数空间的平均（即几何平均）更能代表层中心真实的物理气压值
        p_full = np.exp((np.log(p_half[:-1]) + np.log(p_half[1:])) / 2.0)
    else:
        # 线性网格下，使用算术平均
        p_full = (p_half[:-1] + p_half[1:]) / 2.0

    # 3. 根据泊松方程计算干绝热温度
    # 公式: T = T_surf * (p / p_surf)^(R/cp)
    t_trop_half = T_surf * (p_half / p_surf) ** dlnT_dlnp
    t_trop_full = T_surf * (p_full / p_surf) ** dlnT_dlnp

    # 4. 引入平流层等温假设 (对流层顶修正)
    # 当绝热冷却导致的温度低于设定的平流层温度 T_strat 时，取 T_strat
    t_half = np.maximum(t_trop_half, T_strat)
    t_full = np.maximum(t_trop_full, T_strat)

    return p_surf, T_surf, p_full, p_half, t_full, t_half    

# 2. Read GCM output as 1D input
def count_run_files_and_dirs(path):
    with os.scandir(path) as entries:
        run_items = [entry for entry in entries if entry.name.startswith('run')]
    return len(run_items)
def compute_dayside_mean(var,gw):
    if len(var.shape) == 3:
        l_h,l_1,l_2 = var.shape
        dayside_mean = np.mean(np.dot(gw,var[:,:,:l_2//4]),axis=1)/2+np.mean(np.dot(gw,var[:,:,l_2//4*3:]),axis=1)/2
    elif len(var.shape) == 2:
        l_1,l_2 = var.shape
        dayside_mean = np.mean(np.dot(gw,var[:,:l_2//4]))/2+np.mean(np.dot(gw,var[:,l_2//4*3:]))/2
    else:
        dayside_mean = np.nan
    return dayside_mean
def read_nc(case_name,i,j):
    max_run_id = count_run_files_and_dirs(os.path.join(GCM_root,case_name))
    folder = f'run{str(max_run_id).zfill(4)}' # from previous time step
    directory = os.path.join(GCM_root,case_name,folder)
    nc_file = next((file for file in os.listdir(directory) if file.endswith('.nc')), None)
    if not nc_file:
        print(f"Warning: {nc_file} not found.")
        return
    file_path = os.path.join(directory, nc_file)
    with nc.Dataset(file_path) as file_list:
        lat = file_list.variables['lat'][:]; lon = file_list.variables['lon'][:]
        gw = compute_gaussian(len(lat) // 2)
        if i >= 0 and j >= 0:
            t_full = file_list.variables['soc_temp_lw'][0,:,i,j]
            t_half = file_list.variables['soc_t_half'][0,:,i,j]
            p_full = file_list.variables['soc_p_full'][0,:,i,j]
            p_half = file_list.variables['soc_p_half'][0,:,i,j]
            t_surf = file_list.variables['t_surf_for_soc'][0,i,j]
            soc_olr = file_list.variables['soc_olr'][0,i,j]
            ps = file_list.variables['ps'][0,i,j]
            soc_spectral_olr = file_list.variables['soc_spectral_olr'][0,:,i,j]
            soc_toa_sw_down = file_list.variables['soc_toa_sw_down'][0,i,j]
            soc_surf_flux_sw_down = file_list.variables['soc_surf_flux_sw_down'][0,i,j]
            soc_coszen = file_list.variables['soc_coszen'][0,i,j]
            soc_flux_lw_up = file_list.variables['soc_flux_lw_up'][0,:,i,j]
        elif i == -1 and j == -1:
            t_full = np.mean(np.dot(gw,file_list.variables['soc_temp_lw'][0,:,:,:]),axis=1)
            t_half = np.mean(np.dot(gw,file_list.variables['soc_t_half'][0,:,:,:]),axis=1)
            p_full = np.mean(np.dot(gw,file_list.variables['soc_p_full'][0,:,:,:]),axis=1)
            p_half = np.mean(np.dot(gw,file_list.variables['soc_p_half'][0,:,:,:]),axis=1)
            t_surf = np.mean(np.dot(gw,file_list.variables['t_surf_for_soc'][0,:,:]))
            soc_olr = np.mean(np.dot(gw,file_list.variables['soc_olr'][0,:,:]))
            ps = np.mean(np.dot(gw,file_list.variables['ps'][0,:,:]))
            soc_spectral_olr = np.mean(np.dot(gw,file_list.variables['soc_spectral_olr'][0,:,:,:]),axis=1)
            soc_toa_sw_down = np.mean(np.dot(gw,file_list.variables['soc_toa_sw_down'][0,:,:]))
            soc_surf_flux_sw_down = np.mean(np.dot(gw,file_list.variables['soc_surf_flux_sw_down'][0,:,:]))
            soc_coszen = np.mean(np.dot(gw,file_list.variables['soc_coszen'][0,:,:]))
            soc_flux_lw_up = np.mean(np.dot(gw,file_list.variables['soc_flux_lw_up'][0,:,:,:]),axis=1)
        elif i == -2 and j == -2:
            t_full = compute_dayside_mean(file_list.variables['soc_temp_lw'][0,:,:,:],gw)
            t_half = compute_dayside_mean(file_list.variables['soc_t_half'][0,:,:,:],gw)
            p_full = compute_dayside_mean(file_list.variables['soc_p_full'][0,:,:,:],gw)
            p_half = compute_dayside_mean(file_list.variables['soc_p_half'][0,:,:,:],gw)
            t_surf = compute_dayside_mean(file_list.variables['t_surf_for_soc'][0,:,:],gw)
            soc_olr = compute_dayside_mean(file_list.variables['soc_olr'][0,:,:],gw)
            ps = compute_dayside_mean(file_list.variables['ps'][0,:,:],gw)
            soc_spectral_olr = compute_dayside_mean(file_list.variables['soc_spectral_olr'][0,:,:,:],gw)
            soc_toa_sw_down = compute_dayside_mean(file_list.variables['soc_toa_sw_down'][0,:,:],gw)
            soc_surf_flux_sw_down = compute_dayside_mean(file_list.variables['soc_surf_flux_sw_down'][0,:,:],gw)
            soc_coszen = compute_dayside_mean(file_list.variables['soc_coszen'][0,:,:],gw)
            soc_flux_lw_up = compute_dayside_mean(file_list.variables['soc_flux_lw_up'][0,:,:,:],gw)
    return t_full, t_half, p_full, p_half, t_surf, soc_olr, ps, soc_spectral_olr, soc_toa_sw_down, soc_surf_flux_sw_down, soc_coszen, soc_flux_lw_up


# input T-p from csv file
# import pandas as pd
# csv_path = f"/work/home/ac9b0k6rio/corrk_tools/python/model_1d_test/Ext_Data_Fig4_CO2_CO_TP_samples.csv"
# df = pd.read_csv(csv_path, usecols=[0, 1]) 
# data = df.iloc[1:]
# pres = data.iloc[:, 0].to_numpy()
# temp = data.iloc[:, 1].to_numpy()
# pres = pres[:np.where(~np.isnan(pres))[0][-1] + 1]
# temp = temp[:np.where(~np.isnan(temp))[0][-1] + 1]
# pres = pres[:209]; temp = temp[:209]
# p_half = pres[::2]; t_half = temp[::2]
# p_full = pres[1::2]; t_full = temp[1::2]
# ps = pres[-1]; t_surf = temp[-1]
# print(p_full[-1],t_full[-1],ps,t_surf)


if __name__ == "__main__":
    # set atmos composition here:
    n_co2 = 0; n_co = 0; n_h2o = 0.0; n_n2 = 0.0; n_sio = 0.0; n_o2 = 1.0
    
    M_co2 = 44e-3; M_co = 28e-3; M_h2o = 18e-3; M_n2 = 28e-3; M_sio = 44e-3; M_o2 = 32e-3
    mmw = n_co2*M_co2+n_co*M_co+n_h2o*M_h2o+n_n2*M_n2+n_sio*M_sio+n_o2*M_o2
    co2_mmr = n_co2*M_co2/mmw; co_mmr = n_co*M_co/mmw; h2o_mmr = n_h2o*M_h2o/mmw; n2_mmr = n_n2*M_n2/mmw; sio_mmr = n_sio*M_sio/mmw; o2_mmr = n_o2*M_o2/mmw
    
    # setup the 1D calculation
    soc_base = 'test'
    spectral_root = f'/work/home/ac9b0k6rio/SocSpecGen/spectral_files'
    star_name = 'Trappist-1_sphinx'
    spectral_file_name_lw = f"sp_lw_b96_Trappist-1_sphinx_o2co2h2o"
    spectral_file_name_sw = f"sp_sw_b94_Trappist-1_sphinx_o2co2h2o"
    
    band_n_sw = get_band_number(spectral_file_name_sw)
    spectral_file_sw = os.path.join(spectral_root,f"sp_b{band_n_sw}",spectral_file_name_sw)
    band_n_lw = get_band_number(spectral_file_name_lw)
    spectral_file_lw = os.path.join(spectral_root,f"sp_b{band_n_lw}",spectral_file_name_lw)
    
    # prepare for the T-p profile
    tp_option: Literal['adiabatic', 'GCM'] = 'GCM'
    
    if tp_option == 'adiabatic':
        # not completed yet
        ps,t_surf,p_full,p_half,t_full,t_half = adiabatic_profile(p_surf=1e5, p_toa=1e4, T_surf=300, T_strat=150, dlnT_dlnp=2/7, nlevel=10, option='log')
    elif tp_option == 'GCM':
        case_name = '55cnce_1e5pa_1e0co2'
        t_full, t_half, p_full, p_half, t_surf, soc_olr, ps, soc_spectral_olr, soc_toa_sw_down, soc_surf_flux_sw_down, soc_coszen, soc_flux_lw_up = read_nc(case_name,32,0)
    else:
        pass
    
    write_input(f'{soc_base}','lw',ps,t_surf,p_full,p_half,t_full,t_half,co_mmr=co_mmr,co2_mmr=co2_mmr,sio_mmr=sio_mmr,n2_mmr=n2_mmr,h2o_mmr=h2o_mmr,o2_mmr=o2_mmr)
    
    run_str = f'Cl_run_cdf -B {soc_base} -s {spectral_file_lw} -R 1 {band_n_lw} -I -g 4 -C 5 -t 12 -v 13 -q -z 1 -u'
    os.system(run_str)
    plev, uflux = read_latlon('.',f'{soc_base}','uflx','uflx')
    olr = uflux[0]
    print(f"OLR: {olr:.4e}, {t_full[-1]**4*5.67e-8:.4e}")
    print(uflux)
    os.system(f'rm *{soc_base}.*')
    
    if soc_coszen > 1e-10 and soc_toa_sw_down > 0: # SW calculation only for dayside hemisphere
        write_input(f'{soc_base}','sw',ps,t_surf,p_full,p_half,t_full,t_half,
                    co_mmr=co_mmr,co2_mmr=co2_mmr,sio_mmr=sio_mmr,n2_mmr=n2_mmr,h2o_mmr=h2o_mmr,o2_mmr=o2_mmr,
                    stoa = soc_toa_sw_down/soc_coszen, sza = abs(np.arccos(soc_coszen)*180/np.pi))
        run_str_sw = f'Cl_run_cdf -B {soc_base} -s {spectral_file_sw} -R 1 {band_n_sw} -S -r -g 4 -C 5 -t 16 -v 13 -z 1 -u'
        os.system(run_str_sw)
        plev, dflux = read_latlon('.',f'{soc_base}','vflx','vflx')
        ssw = dflux[-1]
        os.system(f'rm *{soc_base}.*')
    else:
        ssw = 0
    print(ssw, soc_toa_sw_down)
    
    """spectral_olr = []
    for bn in range(1, 111):
        while not os.path.exists(f'{soc_base}.uflx'):
            write_input(f'{soc_base}','lw',ps,t_surf,p_full,p_half,t_full,t_half,co_mmr=co_mmr,co2_mmr=co2_mmr,sio_mmr=sio_mmr,n2_mmr=n2_mmr,h2o_mmr=h2o_mmr)
            run_str = f'Cl_run_cdf -B {soc_base} -s {spectral_file_lw} -R {bn} {bn} -I -g 4 -C 5 -t 12 -v 13 -q -z 1 -u'
            os.system(run_str)
        plev, uflux = read_latlon('.',f'{soc_base}','uflx','uflx')
        olr = uflux[0]
        spectral_olr.append(olr)
        print(f"OLR: {olr:.4e}")
        os.system(f'rm *{soc_base}.*')
    # save spectral_olr to npz
    print(f"total olr: {np.sum(spectral_olr)}")
    np.savez(f'hu_co2-co.npz', spectral_olr=spectral_olr)"""