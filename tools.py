import os
import numpy as np
import netCDF4

def read_wnedges(filename):
    data = []
    with open(filename, 'r') as file:
        for line in file:
            try:
                number = float(line.strip())
                data.append(number)
            except ValueError:
                continue
    return data

def generate_LBL_from_xsec_two_colomn(T_grid, P_grid, Molecule_str,
                                      resolution_xsec_file = '001',
                                      wavnum_min = 0, wavnum_max = 20000):
    '''
        For non-HITRAN data, convert the xsec files to .nc files. 
        input:
            xsec_file: two columns(wave number [cm-1], xsec [cm2/molecule])
            resolution_xsec_file: str, in m⁻¹, format 'nnn' (3 integers); should be consistent with xsec_file
            wavnum_min and wavnum_max: the range of wavnum in .xsec file, in cm-1
        output:
            ncfile
    '''

    root       = "/work/home/ac9b0k6rio/SocSpecGen/"
    path_in    = root+'xsec/xsec_cstep_'+resolution_xsec_file+'_files/'
    # conversion from cm²/molecule to m²/kg : m²/kg = mult_factor * cm²/molecule

    Na          = 6.0221408e+23 # Avogadro number
    M           = 44.0095       # Molar masse of molecule considered [g/mol] (CO2)
    mult_factor = Na/(10*M)

    # Since files are not downloaded from hitran, so we should use -L
    ncfile_name = Molecule_str+"_"+"ExoMol_"+resolution_xsec_file+".nc"
    ncformat = 'NETCDF3_CLASSIC'
    ncfile = netCDF4.Dataset(ncfile_name,"w",format=ncformat,clobber='true', diskless=True) # set clobber to true to overwrite the file successfully
    # create the dimensions for the netCDF file
    ncfile.createDimension('scalar',1)
    ncfile.createDimension('nu',int(wavnum_max/(int(resolution_xsec_file)*10**(-2))+1))
    ncfile.createDimension('pt_pair',len(T_grid)*len(P_grid))
    # create the variables
    nu     = ncfile.createVariable('nu', 'f8', ('nu',))
    kabs   = ncfile.createVariable('kabs','f4',('pt_pair','nu',))
    t_calc = ncfile.createVariable('t_calc','f8',('pt_pair',))
    p_calc = ncfile.createVariable('p_calc','f8',('pt_pair',))
    # assign the data 
    xsec_file=Molecule_str+'_'+str(wavnum_min)+'-'+str(wavnum_max)+'_'+str(T_grid[0]).zfill(4)+'K_'+resolution_xsec_file
    with open(path_in+xsec_file+'.xsec',"r") as f:
        WN = list()
        for line in f:
            data = line.split()
            WN.append(data[0])
    nu[:] = [float(WN[i])*100 + 5 for i in range(len(WN))]              # wavenumber in m⁻¹

    t_calc[:] = (T_grid)*len(P_grid)
    p_calc[:] = sorted([p*1e+5 for p in P_grid]*len(T_grid))     # pressure in Pa


    for i in range(len(P_grid)*len(T_grid)):
        n = len(T_grid)
        p = i//n
        t = i%n
        xsec_file=Molecule_str+'_'+str(wavnum_min)+'-'+str(wavnum_max)+'_'+str(T_grid[t]).zfill(4)+'K_'+resolution_xsec_file
        with open(path_in+xsec_file+'.xsec',"r") as f:
            ACS = list()
            for line in f:
                data = line.split()
                ACS.append(data[1])
            kabs[i,:] = [float(ACS[j])*mult_factor for j in range(len(ACS))]
    # assign attributes
    nu.title         = 'wavenumber'
    nu.long_name     = 'wavenumber'
    nu.units         = 'm-1'
    nu.step           = nu[-1]-nu[-2]

    p_calc.title     = 'pressure'
    p_calc.long_name = 'pressure'
    p_calc.units     = 'Pa'

    t_calc.title     = 'temperature'
    t_calc.long_name = 'temperature'
    t_calc.units     = 'K'

    kabs.title       = 'absorption'
    kabs.long_name   = 'absorption'
    kabs.units       = 'm2 kg-1'
    # close and save
    ncfile.close()
    os.system('mv '+Molecule_str+'_'+'ExoMol_'+resolution_xsec_file+'.nc abs_coeff/')
    
    # set paths:
    #output pathname
    output_path = root+'block5/output_ExoMol_'+resolution_xsec_file
    os.system('rm '+output_path)
    os.system('rm '+output_path+'.nc')
    #monitoring pathname
    mon_path    = root+'block5/monitoring_ExoMol_'+resolution_xsec_file
    os.system('rm '+mon_path)
    #LbL pathname
    LbL_path    = root+'abs_coeff/'+Molecule_str+'_ExoMol_'+resolution_xsec_file+'.nc'  # the one created just before with absorption coefficients
    return output_path,mon_path,LbL_path

def generate_LBL_from_ExoMol_hdf5(hdf5_path,Molecule_str,datasource,update_library=True,test_name=None):
    root       = "/work/home/ac9b0k6rio/SocSpecGen/"
    ncfile_name = f'{Molecule_str}_{datasource}'
    hdf5_file = netCDF4.Dataset(hdf5_path, 'r')
    P_grid = np.array(hdf5_file.variables['p'][:])    # pressure in bar
    T_grid = np.array(hdf5_file.variables['t'][:])   # temperature in K
    if update_library:
        print("Start updating library file ...")
        wns = np.array(hdf5_file.variables['bin_edges'][:])      # wavenumber in cm-1
        xsec_in = np.array(hdf5_file.variables['xsecarr'][:,:,:])       # cross section in cm2/molecule
        # conversion from cm²/molecule to m²/kg : m²/kg = mult_factor * cm²/molecule
        Na          = 6.0221408e+23 # Avogadro number
        M           = hdf5_file.variables['mol_mass'][:]
        mult_factor = Na/(10*M)

        # generate the netCDF file
        ncfile_name = f'{Molecule_str}_{datasource}'
        ncformat = 'NETCDF4'
        ncfile = netCDF4.Dataset(ncfile_name+'.nc',"w",format=ncformat,clobber='true') # set clobber to true to overwrite the file successfully
        # create the dimensions for the netCDF file
        print("Creating the dimensions ...")
        ncfile.createDimension('scalar',1)
        ncfile.createDimension('nu',len(wns))
        ncfile.createDimension('pt_pair',len(T_grid)*len(P_grid))
        # create the variables
        print ("Creating the variables ...")
        nu     = ncfile.createVariable('nu', 'f8', ('nu',))
        kabs   = ncfile.createVariable('kabs','f4',('pt_pair','nu',))
        t_calc = ncfile.createVariable('t_calc','f8',('pt_pair',))
        p_calc = ncfile.createVariable('p_calc','f8',('pt_pair',))
        
        # assign attributes
        print("Assigning the attributes ...")
        nu.title         = 'wavenumber'
        nu.long_name     = 'wavenumber'
        nu.units         = 'm-1'
        nu.step           = (wns[-1]-wns[-2]) * 100

        p_calc.title     = 'pressure'
        p_calc.long_name = 'pressure'
        p_calc.units     = 'Pa'

        t_calc.title     = 'temperature'
        t_calc.long_name = 'temperature'
        t_calc.units     = 'K'

        kabs.title       = 'absorption'
        kabs.long_name   = 'absorption'
        kabs.units       = 'm2 kg-1'
        
        # assign the data
        print("Assigning the data ...")
        print(wns.dtype)
        print(f"shape of wns {wns.shape}")
        nu[:] = wns*100           # wavenumber in m⁻¹
        t_calc[:] = np.tile(T_grid, len(P_grid))
        p_calc[:] = np.repeat(P_grid*1e5, len(T_grid))     # pressure in Pa
        
        kabas = xsec_in.reshape(len(P_grid) * len(T_grid), -1) * mult_factor
        print("Assigning the absorption coefficients ...")
        kabs[:] =  kabas # !!! from xsec to abs coeff

        # close and save
        print("Library file updated successfully!")
        ncfile.close()
        # print(f'mv {ncfile_name}.nc {root}abs_coeff/{ncfile_name}.nc')
        os.system(f"mv {ncfile_name}.nc {os.path.join(root,f'abs_coeff/{ncfile_name}.nc')}")

    # set paths:
    #output pathname
    output_path = root+'block5/output_'+ncfile_name +'_'+test_name if test_name else ''
    if os.path.exists(output_path):
        os.system('rm '+output_path)
    if os.path.exists(output_path+'.nc'):
        os.system('rm '+output_path+'.nc')
    #monitoring pathname
    mon_path    = root+'block5/monitoring_'+ncfile_name+'_'+test_name if test_name else ''
    if os.path.exists(mon_path):
        os.system('rm '+mon_path)
    #LbL pathname
    ref_LBL = os.path.join(root,f'abs_coeff/{ncfile_name}.nc')
    LbL_path    = os.path.join(root,f'abs_coeff/{ncfile_name}_{test_name}.nc')  # the one created just before with absorption coefficients
    if os.path.exists(LbL_path):
        os.remove(LbL_path) 
    os.system(f"cp {ref_LBL} {LbL_path}")
    return output_path,mon_path,LbL_path,T_grid,P_grid