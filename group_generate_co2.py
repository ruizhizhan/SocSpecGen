import os
import numpy as np
import netCDF4
from typing import Literal
import re
from tools import read_wnedges,generate_LBL_from_xsec_two_colomn,generate_LBL_from_ExoMol_hdf5

def generate_spectral_file(test_name,spec_type: Literal['lw', 'sw'],wnedges_lower, wnedges_upper,star_name,
                           path_in,datasource,num_kterm):
    root       = "/work/home/ac9b0k6rio/SocSpecGen/"
    band_n     = len(wnedges_upper)
    os.chdir(root)
    work_dir   = os.path.join(root, f'{test_name}')
    if not os.path.exists(work_dir):
        os.system('mkdir '+work_dir)
    else:
        os.system('rm -rf '+work_dir)
        os.system('mkdir '+work_dir)
    os.chdir(work_dir)
    Molecule_str = '12C-16O2'
    gas_id = '2'
    include_cia = True       # test only
    include_solar_sed = True # test only
    gas_ab_source: Literal['hitran', 'ExoMol', 'ExoMol0pres'] = 'ExoMol' 

    ncfile_name = f'{Molecule_str}_{datasource}'
    if os.path.exists(os.path.join(root,f'abs_coeff/{ncfile_name}.nc')):
        update_library = False # Set False if abs_coeff file already exists for the right .h5
    else:
        update_library = True
    if gas_ab_source == 'ExoMol': #input should be a hdf5 file including xsec tables
        # !!! make sure wnedges are set within the range of the xsec file
        hdf5_path = os.path.join(root,path_in)
        print("Processing xsec tables ..., update_library = ",update_library)
    output_path,mon_path,LbL_path,T_grid,P_grid = generate_LBL_from_ExoMol_hdf5(hdf5_path,Molecule_str,datasource,update_library,test_name)
    outputfilename = 'sp_'+spec_type+'_b'+str(band_n)+'_'+star_name+'_'+datasource+'_'+'nk'+str(num_kterm)

    # pt file
    pt_path = os.path.join(root,f'block5/pt_file_{test_name}')
    if os.path.exists(pt_path):
        os.system(f"rm {pt_path}")
    with open(pt_path,"a") as file:
        file.write('*PTVAL'+'\n')
        for P in P_grid:
            file.write(str(P*1e+5))
            for T in T_grid:
                file.write(' '+str(T))
            file.write('\n')
        file.write('*END')

    # ref pt_file
    ref_pt_path = os.path.join(root,f'block5/ref_pt_file_{test_name}')
    if os.path.exists(ref_pt_path):
        os.system(f'rm {ref_pt_path}')
    with open(ref_pt_path,"a") as file:
        file.write('*REF 1 '+str(band_n)+' '+gas_id+' 1e+5 300.0')  # *REF first−band last−band gas pressure temperature.
        

    # generate skeleton spectral file using prep_spec utility
    exec_file_name = os.path.join(work_dir,f"sp_exec{test_name}.sh")
    if os.path.exists(exec_file_name):
        os.remove(exec_file_name)
    skeleton_file_name = f'sp_b{band_n}_{test_name}'
    skeleton_file_path = os.path.join(work_dir,skeleton_file_name)
    if os.path.exists(skeleton_file_path):
        os.remove(skeleton_file_path)
    with open(exec_file_name, "w+") as f:
        f.write('prep_spec <<EOF'+'\n')
        f.write(skeleton_file_name+'\n')
        f.write(str(band_n)+'\n')                 # number of bands
        f.write('1'+'\n')                                 # number of absorbing gases
        f.write(gas_id+'\n')                              # absorber ids (1 for water)
        if include_cia:
            f.write('1'+'\n')                                 # number of generalised continua to be included
            f.write(f'{gas_id} {gas_id}'+'\n')                               # gas identifiers for continuum 1 co2-co2
        else:
            f.write('0\n')
        f.write('0'+'\n')                                 # number of aerosols
        f.write('c'+'\n')                                 # band units (c for inverse cm)

        for band in range(band_n):                      # write band edges one by one
            f.write(f'{wnedges_lower[band]} ')                  #
            f.write(str(wnedges_upper[band])+'\n')              #


        for band in range(band_n):                         # absorber ids in each band 
            f.write(gas_id+'\n')                             #

        for band in range(band_n):                         # continua ids in each band
            f.write('0'+'\n')                             #
            
        f.write('n'+'\n')                                 # exclude no bands

        f.write('-1'+'\n')                                # close prep_spec
        f.write('EOF'+'\n')                               #

    os.chmod(exec_file_name,0o777)
    os.system(exec_file_name)                    # run file script
    os.system('rm '+exec_file_name)                   # clean script

    def find_index(lower_bound, upper_bound, lower, upper):
        """
        查找 lower 和 upper 所在的 band 索引。
        
        逻辑规则：
        1. 如果数值 < 整体下限 (lower_bound[0]) -> 返回 0
        2. 如果数值 > 整体上限 (upper_bound[-1]) -> 返回 最后一个 index
        3. 如果在范围内 -> 返回对应的 band index
        
        参数:
        lower_bound, upper_bound: 长度为 band_n 的 numpy 数组，需按从小到大排序
        lower, upper: 待查找的数值
        """
        lb = np.array(lower_bound)
        ub = np.array(upper_bound)
        n = len(lb) # band 的总数量

        # 定义一个内部函数来复用逻辑
        def get_clamped_idx(val):
            # 1. 超出下限：返回 0
            if val < lb[0]:
                return 0
            
            # 2. 超出上限：返回最后一个索引
            if val > ub[-1]:
                return n - 1
            
            # 3. 正常查找：在区间内
            matches = np.where((lb <= val) & (ub >= val))[0]
            
            if matches.size > 0:
                return matches[0]
            else:
                # 4. 特殊情况：值在 min 和 max 之间，但正好落在两个 band 的缝隙里
                # (例如 band1是10-20，band2是30-40，数值是25)
                # 如果你的 band 是连续的，这步永远不会触发。
                # 如果有缝隙，这里返回 None 还是其它值取决于业务需求。
                return None 

        idx_lower = get_clamped_idx(lower)
        idx_upper = get_clamped_idx(upper)
        
        return idx_lower+1, idx_upper+1

 # 5.2 Generate corrk data
    exec_file_corrk_abs = f"corr_k_ExoMol_{test_name}.sh"
    if os.path.exists(exec_file_corrk_abs):
        os.system('rm '+exec_file_corrk_abs)

    with open(exec_file_corrk_abs, "w+") as f:
        f.write('Ccorr_k'+' ')
        f.write(f'-s {skeleton_file_name} ')                 # spectral file
        # find the right band limits
        lower = 1.0; upper  = 20000.0
        idx_lower, idx_upper = find_index(wnedges_lower,wnedges_upper,lower,upper)
        f.write(f'-R {idx_lower} {idx_upper} ')                          # band limits
        f.write(f'-F {pt_path}'+' ')                        # pressures and temperatures at which to calculate coefficients
        f.write(f'-r {ref_pt_path}'+' ')                    # reference conditions for scaling
        f.write('-l '+gas_id+' 1.0e5'+' ')                        # generate line absorption data. gas id then maximum absorptive pathlength for the gas (kg/m2)
        f.write(f'-n {num_kterm}'+' ')                             # Number of k-terms the correlated-k fit should use
        f.write('-lk'+' ')                               # a look-up table will be used for the pressure/temperature scaling
        f.write('+p'+' ')                                # Planckian Weighting
        f.write('-o '+output_path+' ')                   # Pathname of output file
        f.write('-m '+mon_path+' ')                      # Pathname of monitoring file
        f.write('-L '+LbL_path+' ')                      # Pathname of LbL file (.nc)
        f.write('-np 1')                                 # Number of OpenMP threads

    os.chmod(exec_file_corrk_abs,0o777)
    os.system('cat '+exec_file_corrk_abs)
    os.system('./'+exec_file_corrk_abs)
    os.system('rm '+exec_file_corrk_abs)

    # If include CIA from hitran
    if include_cia:
        T_cia_grid = np.arange(200,801,100) # should be consistent with the CIA file, Figure 4.25
        # if T is too high, OLR will become very large...
        pt_cia_path = os.path.join(root,f'block19/pt_cia_{test_name}')
        if os.path.exists(pt_cia_path):
            os.system(f'rm {pt_cia_path}')
        with open(pt_cia_path,"a") as file:
            file.write('*PTVAL'+'\n')
            P_0 = 1 # bar (pick surface pressure? from other code: titan 2e5, mars 1e5...)
            # if P is set to small, OLR will become very large...
            file.write(str(P_0*1e+5))
            for T in T_cia_grid:
                file.write(' '+str(T))
            file.write('\n')
            file.write('*END')
        file_name_list = ['CO2-CO2_2018']
        #for T in [200,280,350,430,529,651,800]:
       #     file_name_list.append(f'CO2-CO2_1_750_{T}.00K')
        #for T in [200,250,350]:# 
        #    file_name_list.append(f'CO2-CO2_1000_1800_{T}.00K')
        #file_name_list.append('CO2-CO2_2510_2850_221.00K')
        #file_name_list.append('CO2-CO2_2510_2850_235.00K')
        #file_name_list.append('CO2-CO2_2510_2850_297.00K')
        #file_name_list.append('CO2-CO2_2851_3250_298.00K')
        
        for file in file_name_list:
            output_cia_path = os.path.join(root,f'block19/output_CIA_{file}_{test_name}')
            if os.path.exists(output_cia_path):
                os.remove(output_cia_path)
            output_cia_nc_path = os.path.join(root,f'block19/output_CIA_{file}_{test_name}.nc')
            if os.path.exists(output_cia_nc_path):
                os.remove(output_cia_nc_path)
            mon_cia_path = os.path.join(root,f'block19/monitoring_CIA_{file}_{test_name}')
            if os.path.exists(mon_cia_path):
                os.remove(mon_cia_path)
            lbl_cia_path = os.path.join(root,f'block19/LBL_CIA_{file}_{test_name}.nc')
            if os.path.exists(lbl_cia_path):
                os.remove(lbl_cia_path)
            
        exec_file_CIA = os.path.join(work_dir,f"corr_k_CIA_{test_name}.sh")
        if os.path.exists(exec_file_CIA):
            os.system('rm '+exec_file_CIA)
        with open(exec_file_CIA, "w+") as f:
            for file in file_name_list:
                #matches = re.search(r'CO2-CO2_(\d+)_(\d+)_', file)
                #lower, upper = matches.groups()
                idx_lower, idx_upper = (1,band_n) #find_index(wnedges,int(lower),int(upper))
                f.write('Ccorr_k'+
                        ' -CIA '+root+f'/hitran/CO2-CO2_2018/'+file+'_.cia'+   # hitran−cia−file
                        f' -R {idx_lower} {idx_upper} '+  # first and last band
                        f' -F {pt_cia_path}'+   # p-t file
                        ' -ct '+gas_id+' '+gas_id+' 1000.0'+ # gas1 gas2 max−path(maximum absorptive pathlength (kg2/m5) for the gas pair.)
                        ' -i '+'1.0'+ # Frequency increment
                        ' -t 1.0e-2'+ # Tolerance for the calculation
                        ' -s '+skeleton_file_name+
                        ' +p -lk'+ # Planckian Weighting, look-up table
                        ' -o '+ root+f'/block19/output_CIA_{file}_{test_name}'+ # output file
                        ' -m '+ root+f'/block19/monitoring_CIA_{file}_{test_name}'+ # monitoring file.
                        ' -L '+ root+f'/block19/LBL_CIA_{file}_{test_name}.nc' + '\n')  # 

        os.chmod(exec_file_CIA,0o777)
        os.system('cat '+exec_file_CIA)
        os.system(exec_file_CIA)
        os.system('rm '+exec_file_CIA)

    # add UV lines to sw files
    if spec_type == 'sw':
        # make hitran format .xsc file
        # run prepare_xuv_new.ipynb get 'CO2_venot.uvxsc'
        uv_pt_file = 'pt_co2_uv'
        UV_CO2_path=os.path.join(root,'ExoMol/12C_16O2/XUV/')
        os.system(f"cp {os.path.join(UV_CO2_path,uv_pt_file)} {os.path.join(root,'block5')}")       # change the T grid will not change the output spectral file
        UV_CO2 = 'CO2_DTU.uvxsc'
        
        exec_file_corrk_xuv = f"corr_k_ExoMol_{test_name}_xuv.sh"
        if os.path.exists(exec_file_corrk_xuv):
            os.system('rm '+exec_file_corrk_xuv)
        output_path_xuv = os.path.join(root,f'block5/output_xuv_{test_name}')
        mon_path = os.path.join(root,f'block5/monitoring_xuv_{test_name}')
        LBL_path = os.path.join(root,f'block5/LBL_xuv_{test_name}.nc')
        if os.path.exists(output_path_xuv):
            os.system('rm '+output_path_xuv)
            os.system(f'rm {output_path_xuv}.nc')
        if os.path.exists(mon_path):
            os.system('rm '+mon_path)
        if os.path.exists(LBL_path):
            os.system('rm '+LBL_path)
        with open(exec_file_corrk_xuv, "w+") as f:
            f.write('Ccorr_k'+' ')
            f.write(f'-s {skeleton_file_name} ')                 # spectral file
            # find the right band limits
            f.write(f'-UVX {os.path.join(root,UV_CO2_path,UV_CO2)} ')
            lower = 1e7/250; upper  = 1e7/100        # from xsec plot; nm -> cm-1
            idx_lower, idx_upper = find_index(wnedges_lower,wnedges_upper,lower,upper)
            f.write(f'-R {idx_lower} {idx_upper} ')                          # band limits
            f.write('-F '+root+f'/block5/{uv_pt_file}'+' ')                        # pressures and temperatures at which to calculate coefficients
            f.write('-i 1.0 ')
            f.write('-l '+gas_id+' 1.0e1'+' ')                        # generate line absorption data. gas id then maximum absorptive pathlength for the gas (kg/m2)
            f.write('-t 1.0e-3 ')
            Solar_spec = root+'/stellar_spectra/soc_in/'+star_name
            f.write(f'+S {Solar_spec} ')
            f.write('-o '+output_path_xuv+' ')                   # Pathname of output file
            f.write('-m '+mon_path+' ')                      # Pathname of monitoring file
            f.write('-L '+LBL_path+' ')                      # Pathname of LbL file (.nc)
            f.write('-np 1')                                 # Number of OpenMP threads
        os.chmod(exec_file_corrk_xuv,0o777)
        os.system('cat '+exec_file_corrk_xuv)
        os.system('./'+exec_file_corrk_xuv)
        os.system('rm '+exec_file_corrk_xuv)
    # 5.5 Adding data to the spectral file
    exec_file_sp  = os.path.join(work_dir,f"fill_in_sp_{test_name}.sh")

    f = open(exec_file_sp, "w+")

    f.write('prep_spec <<EOF'+'\n')
    f.write(skeleton_file_name+'\n') # Enter the name of the spectral file.
    f.write('a'+ '\n') # append

    # add block 3 (Rayleigh scattering coefficient, short wave only)
    if spec_type == 'sw':
        f.write('3'+'\n')             # Rayleigh scattering coefficient
        f.write(root+'/stellar_spectra/soc_in/'+star_name+'\n') # Enter file of solar irradiance data.
        f.write('C'+'\n')             # (A)air. (H)H2-He gas. (C)custom composition (A/H/C)
        f.write('A'+'\n')             # (A)calculation include all gases listed in block 0.
    # add block 5 (k-coeff)
    f.write('5'+'\n')                 # is exists, type 'y' to continue
    f.write(output_path+'\n')
    # add xuv for sw
    if spec_type == 'sw':
        f.write('5'+'\n')
        f.write('y \n')
        f.write(output_path_xuv+'\n')
    # add block 6 (thermal source function) & 2 (star spectrum)
    if spec_type == 'lw':
        f.write('6'+'\n')             # coefficients of a polynomial fit to the Planck function in each band 
        f.write('n'+'\n')             # Is a filter function required (Y/N)?
        f.write('T'+'\n')             # Table or Polynomial fit (T/P)?
        f.write('50 2000'+'\n')       # Temperature range in K
        f.write('195'+'\n')            # Number of points [tested: 10K/point is enough]
    if include_solar_sed:
        f.write('2'+'\n')             # Solar spectrum
        f.write('n'+'\n')             # No filter function
        f.write(root+'/stellar_spectra/soc_in/'+star_name+'\n')
        f.write('y'+'\n')             # agree to overwrite

    # add block 19 (continuum)
    if include_cia:
        f.write('19'+'\n')
        f.write(root+f'/block19/output_CIA_{file_name_list[0]}_{test_name}'+'\n')

    f.write('-1'+ '\n')
    f.write('EOF'+ '\n')

    f.close()
    os.chmod(exec_file_sp,0o777)
    os.system(exec_file_sp) 
    os.system('rm '+exec_file_sp)

    # if path is not exist, create it
    if not os.path.exists(root+f'/spectral_files/sp_b{band_n}'):
        os.makedirs(root+f'/spectral_files/sp_b{band_n}')
    os.system(f'mv {skeleton_file_name} '+root+f'/spectral_files/sp_b{band_n}'+'/'+outputfilename)
    os.system(f'mv {skeleton_file_name}_k '+root+f'/spectral_files/sp_b{band_n}'+'/'+outputfilename+'_k')
    os.system(f'rm {LbL_path}')
    os.remove(root+'/abs_coeff/'+ncfile_name+'.nc')