import numpy as np
import os
import netCDF4


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
    path_in    = root+'/xsec/xsec_cstep_'+resolution_xsec_file+'_files/'
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
    output_path = root+'/block5/output_ExoMol_'+resolution_xsec_file
    os.system('rm '+output_path)
    os.system('rm '+output_path+'.nc')
    #monitoring pathname
    mon_path    = root+'/block5/monitoring_ExoMol_'+resolution_xsec_file
    os.system('rm '+mon_path)
    #LbL pathname
    LbL_path    = root+'/abs_coeff/'+Molecule_str+'_ExoMol_'+resolution_xsec_file+'.nc'  # the one created just before with absorption coefficients
    return output_path,mon_path,LbL_path

def find_index(lower_bound, upper_bound, lower, upper, strict_band_edges=False):
    """
    Finds the band indices corresponding to the 'lower' and 'upper' values.

    Args:
        lower_bound: Array of band start wavenumbers.
        upper_bound: Array of band end wavenumbers.
        lower: The lower limit of the data range.
        upper: The upper limit of the data range.
        strict_band_edges: 
            If True, returns indices of bands that are FULLY contained within [lower, upper].
            If False, uses original logic (clamped to global bounds, handles partial overlaps).

    Returns:
        Tuple (idx_start, idx_end) using 1-based indexing.
    """
    lb = np.array(lower_bound)
    ub = np.array(upper_bound)
    n = len(lb) 

    # --- Strict Mode: Only include bands fully covered by the data range ---
    if strict_band_edges:
        # 1. Find the first band where the band's lower bound is >= data's lower limit
        valid_starts = np.where(lb >= lower)[0]
        
        # 2. Find the last band where the band's upper bound is <= data's upper limit
        valid_ends = np.where(ub <= upper)[0]

        if valid_starts.size == 0 or valid_ends.size == 0:
            raise ValueError(f"find_index (Strict): No bands are fully contained within the range {lower}-{upper}.")

        idx_lower = valid_starts[0] # First valid band (0-based index)
        idx_upper = valid_ends[-1]  # Last valid band (0-based index)

        if idx_lower > idx_upper:
            raise ValueError(f"find_index (Strict): Range {lower}-{upper} is too narrow to fully contain any complete band.")

        # --- Warning Logic: Check for discarded data ---
        discarded_ranges = []
        # Check if data exists before the first selected band
        if lower < lb[idx_lower]:
            discarded_ranges.append(f"Start: {lower} - {lb[idx_lower]} cm-1")
        
        # Check if data exists after the last selected band
        if upper > ub[idx_upper]:
            discarded_ranges.append(f"End:   {ub[idx_upper]} - {upper} cm-1")
            
        if discarded_ranges:
            print("\n[Warning] Strict Band Mode: Data segments were discarded because they do not fill a complete band:")
            for msg in discarded_ranges:
                print(f"  - {msg}")
            print("  Suggestion: Adjust 'wnedges' in config_data.py to align with your data boundaries to include this data.\n")

        return idx_lower + 1, idx_upper + 1

    # --- Legacy/Loose Mode: Original Logic ---
    def get_clamped_idx(val, is_lower_bound_val):
        # 1. Below global lower bound: return 0
        if val < lb[0]:
            return 0
        
        # 2. Above global upper bound: return the last index
        if val > ub[-1]:
            return n - 1
        
        # 3. Find which bands contain the value
        matches = np.where((lb <= val) & (ub >= val))[0]
        
        if matches.size > 0:
            if is_lower_bound_val:
                # Logic A: If finding 'lower', take the "higher" band (last match).
                return matches[-1]
            else:
                # Logic B: If finding 'upper', take the "lower" band (first match).
                return matches[0]
        else:
            return None 

    idx_lower = get_clamped_idx(lower, is_lower_bound_val=True)
    idx_upper = get_clamped_idx(upper, is_lower_bound_val=False)
    
    if idx_lower is not None and idx_upper is not None:
        return idx_lower + 1, idx_upper + 1
    else:
        raise ValueError("find_index: Lower or upper value falls into a gap between bands.")

def generate_LBL_from_ExoMol_hdf5(root, hdf5_path,Molecule_str,datasource,update_library=True,test_name=None):
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
    output_path = root+'/block5/output_'+ncfile_name +'_'+test_name if test_name else ''
    if os.path.exists(output_path):
        os.system('rm '+output_path)
    if os.path.exists(output_path+'.nc'):
        os.system('rm '+output_path+'.nc')
    #monitoring pathname
    mon_path    = root+'/block5/monitoring_'+ncfile_name+'_'+test_name if test_name else ''
    if os.path.exists(mon_path):
        os.system('rm '+mon_path)
    #LbL pathname
    ref_LBL = os.path.join(root,f'abs_coeff/{ncfile_name}.nc')
    LbL_path    = os.path.join(root,f'abs_coeff/{ncfile_name}_{test_name}.nc')  # the one created just before with absorption coefficients
    if os.path.exists(LbL_path):
        os.remove(LbL_path) 
    os.system(f"cp {ref_LBL} {LbL_path}")
    return output_path,mon_path,LbL_path,T_grid,P_grid