""" 
    Add block 6 to sw spectral file
    or add 
"""
import os
from typing import Literal, Tuple, List

origin_type:Literal['sw','lw'] = 'lw'
root = os.path.dirname(os.path.abspath(__file__))

for star in ['GJ3929', 'LTT1445A', 'LHS1140', 'TOI198', 'TOI406', 'TOI771', 'HD260655', 'TOI244']: #
    spec_file = os.path.join(root,f"spectral_files/sp_b186/sp_{origin_type}_b186_{star}_sphinx_CO2_T62xP22_001_nk20")
    
    skeleton_file_root = os.path.join(root,f"spectral_files/agni_version/sp_b186/")
    if not os.path.exists(skeleton_file_root):
        os.makedirs(skeleton_file_root)
    skeleton_file_name = os.path.join(skeleton_file_root,f"sp_b186_{star}_sphinx_CO2_T62xP22_001_nk20")
    os.system(f"cp {spec_file} {skeleton_file_name}")
    os.system(f"cp {spec_file}_k {skeleton_file_name}_k")
    
    exec_file_sp  = os.path.join(root,f"cali_spec.sh")
    f = open(exec_file_sp, "w+")

    f.write('prep_spec <<EOF'+'\n')
    f.write(skeleton_file_name+'\n') # Enter the name of the spectral file.
    f.write('a'+ '\n') # append
    if origin_type == 'sw':
        # add block 6 (thermal source function) 
        f.write('6'+'\n')             # coefficients of a polynomial fit to the Planck function in each band 
        f.write('n'+'\n')             # Is a filter function required (Y/N)?
        f.write('T'+'\n')             # Table or Polynomial fit (T/P)?
        f.write('50 2000'+'\n')       # Temperature range in K
        f.write('195'+'\n')            # Number of points [tested: 10K/point is enough]
    else:
        f.write('3'+'\n')             # Rayleigh scattering coefficient
        star_name = f"{star}_sphinx"
        f.write(root+'/stellar_spectra/soc_in/'+star_name+'\n') # Enter file of solar irradiance data.
        f.write('C'+'\n')             # (A)air. (H)H2-He gas. (C)custom composition (A/H/C)
        f.write('A'+'\n')             # (A)calculation include all gases listed in block 0.

    f.write('-1'+ '\n')
    f.write('EOF'+ '\n')

    f.close()
    os.chmod(exec_file_sp,0o777)
    os.system(exec_file_sp) 
    os.system('rm '+exec_file_sp)
