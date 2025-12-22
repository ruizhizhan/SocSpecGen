"""
    write and submit a list of slurm scripts to generate correlated-k spectral files
"""


import numpy as np
import os
from group_generate_co2 import generate_spectral_file

root = os.path.dirname(os.path.abspath(__file__))
slurms_root = os.path.join(root,"slurms")
def write_py(test_name,spec_type:str,wnedges_lower,wnedges_upper,star_name,path_in,datasource,num_kterm):
    python_path = os.path.join(slurms_root,f'{test_name}.py')
    if os.path.exists(python_path):
        os.remove(python_path)
    with open(python_path,'w') as f:
        f.write(f'import numpy as np\n')
        f.write(f'import os\n')
        f.write(f'from group_generate_co2 import generate_spectral_file\n')
        f.write(f'\n')
        # wnedges is numpy array, write it as list, comma separated
        f.write(f'wnedges_lower = np.array({wnedges_lower.tolist()})\n')
        f.write(f'wnedges_upper = np.array({wnedges_upper.tolist()})\n')
        f.write(f'star_name = \'{star_name}\'\n')
        f.write(f'path_in = \'{path_in}\'\n')
        f.write(f'datasource = \'{datasource}\'\n')
        f.write(f'num_kterm = {num_kterm}\n')
        f.write(f'\n')
        f.write(f'generate_spectral_file(\'{test_name}\',\'{spec_type}\',wnedges_lower,wnedges_upper,star_name,path_in,datasource,num_kterm)\n')

def get_file_size_in_gb(file_path):
    try:
        size_bytes = os.path.getsize(file_path)
        size_gb = size_bytes / (1024 * 1024 * 1024)
        return size_gb
    except OSError as e:
        print(f"Error: {e}")
        return None
def ask_for_yes_no():
    user_input = input("Continue to next T-P grid case? (y/n): ").strip().lower()
    if user_input == 'y':
        print("continue...")
        return 1
    elif user_input == 'n':
        print("Stopped!")
        return 0
    else:
        print("Invalid input.")
        return 0

def write_slurm(case_name,hdf5_path_in):
    hdf5_size = get_file_size_in_gb(os.path.join(root,hdf5_path_in))
    ncores = int(np.ceil(hdf5_size/2)+1)  # 4 GB a core and 2 files
    if ncores >= 6:
        ncores = ncores + 1
    slurm_path = os.path.join(slurms_root,f'{case_name}.sh')
    if os.path.exists(slurm_path):
        os.remove(slurm_path)
    with open(slurm_path,'w') as f:
        f.write(f'#!/bin/bash\n')
        f.write(f'#SBATCH --job-name={case_name}\n')
        f.write(f'#SBATCH --output=output_%j.out \n')
        f.write(f'#SBATCH --error=error_%j.err\n')
        f.write(f'#SBATCH -N 1 \n')
        f.write(f'#SBATCH -c {ncores} \n')
        f.write(f'#SBATCH --partition=wzhcnormal\n')
        f.write(f'\n')
        f.write(f'cd {slurms_root}\n')
        f.write(f'source /work/home/ac9b0k6rio/miniconda3/etc/profile.d/conda.sh\n')
        f.write(f'source /work/home/ac9b0k6rio/miniconda3/bin/activate SocSpecGen\n')
        f.write(f'python {case_name}_lw.py\n')
        f.write(f'rm -r {root}/{case_name}_lw\n')
        f.write(f'python {case_name}_sw.py\n')
        f.write(f'rm -r {root}/{case_name}_sw\n')

def write_slurm_group(case_name_list,slurm_name,hdf5_path_in):
    hdf5_size = get_file_size_in_gb(os.path.join(root,hdf5_path_in))
    ncores = int(np.ceil(hdf5_size/2)+1)  # 4 GB a core and 2 files
    if ncores >= 6:
        ncores = 32 #ncores + 3
    slurm_path = os.path.join(slurms_root,f'{slurm_name}.sh')
    if os.path.exists(slurm_path):
        os.remove(slurm_path)
    with open(slurm_path,'w') as f:
        f.write(f'#!/bin/bash\n')
        f.write(f'#SBATCH --job-name={slurm_name}\n')
        f.write(f'#SBATCH --output=output_%j.out \n')
        f.write(f'#SBATCH --error=error_%j.err\n')
        f.write(f'#SBATCH -N 1 \n')
        f.write(f'#SBATCH -c {ncores} \n')
        f.write(f'#SBATCH --partition=wzhcnormal\n')
        f.write(f'\n')
        f.write(f'ulimit -c unlimited\nulimit -s unlimited\nulimit -l unlimited\n')
        f.write(f'cd {slurms_root}\n')
        f.write(f'source /work/home/ac9b0k6rio/miniconda3/etc/profile.d/conda.sh\n')
        f.write(f'source /work/home/ac9b0k6rio/miniconda3/bin/activate SocSpecGen\n')
        for case_name in case_name_list:
            f.write(f'python {case_name}_lw.py\n')
            f.write(f'rm -r ../{case_name}_lw\n')
            f.write(f'python {case_name}_sw.py\n')
            f.write(f'rm -r ../{case_name}_sw\n')


#wnedges_list.append(np.concatenate([np.arange(0,20000,400), np.arange(20000,90001,2000)])) # sp85, default

# designed for CO2: default, empty band between 20000-40000 cm-1
# wnedges_lower = np.concatenate([np.arange(0,20001,400), np.arange(40000,100000-5000,5000)])
# wnedges_upper = np.concatenate([np.arange(400,20001,400), np.arange(40000,100000,5000)])

# designed for CO2: high resolution around 15 um 
# wavlen_upper = np.arange(13.52e-6,16.64e-6,0.01e-6) # MIRI F1500W: 13.52 - 16.64
# wavlen_lower = np.arange(13.53e-6,16.65e-6,0.01e-6) # MIRI F1500W: 13.52 - 16.64
# wavnum_upper = [1e-2/wavlen for wavlen in wavlen_upper]
# wavnum_lower = [1e-2/wavlen for wavlen in wavlen_lower]
# wnedges_lower = np.concatenate([np.arange(0,20001,400), np.arange(40000,100000-5000,5000)])
# wnedges_upper = np.concatenate([np.arange(400,20001,400), np.arange(40000,100000,5000)])
# wnedges_lower = np.sort(np.concatenate([wnedges_lower, wavnum_lower])); wnedges_lower = np.unique(np.round(wnedges_lower, 10))
# wnedges_upper = np.sort(np.concatenate([wnedges_upper, wavnum_upper])); wnedges_upper = np.unique(np.round(wnedges_upper, 10))

# design for DDT plot, 0.2 um from 1 to 25 um
wavlen = np.arange(26.2e-6,1.0e-6,-0.2e-6)
wavnum = [1e-2/wavlen_i for wavlen_i in wavlen]
wnedges = np.concatenate([np.arange(0,20001,400), np.arange(40000,100000,5000), wavnum])
wnedges = np.unique(np.round(wnedges, 3))
wnedges_lower = wnedges[:-1]
wnedges_upper = wnedges[1:]


wnedges_lower[0] = 1.0
band_n = len(wnedges_lower)

res_list = ['001'] # '100','010','001'; run '010' and '100' for optimal configuration only. 
star_sed_list = ['GJ3929','LTT1445A', 'LHS1140', 'TOI198', 'TOI406', 'TOI771', 'HD260655', 'TOI244'] # , 
NT_list = [62] # 13,25,42,49,57,62(default->4500K)
NP_list = [22] # 8,15,22,43
nk_list = [20] # 5,10,20,30

hdf5_root = os.path.join(root,'ExoMol/hdf5/CO2/')
# data source should be consistent with hdf5 files.

for res in res_list: # 
    for NT in NT_list: # 
        for NP in NP_list: # 
            datasource = f'CO2_T{NT}xP{NP}_{res}'
            hdf5_path_in = hdf5_root+f'CO2_UCL-4000_1.0-20000.0_T{NT}xP{NP}_{res}.h5'
            # generate files with no h5 file same time
            case_name_list = []
            for nk in nk_list: # 
                for star_sed in star_sed_list:
                    print(f"Writing files for nk={nk}, NT={NT}, NP={NP}, res={res}, bands={band_n},star={star_sed}")
                    case_name = f'CO2_b{band_n}_{star_sed}_T{NT}xP{NP}_{res}_{nk}'
                    if os.path.exists(os.path.join(f"{root}/spectral_files",f"sp_b{band_n}",f"sp_lw_b{band_n}_{star_sed}_CO2_T{NT}xP{NP}_{res}_{nk}")):
                        print(f"{case_name} exists, skipping ...")
                        continue
                    write_py(f'{case_name}_lw','lw',wnedges_lower,wnedges_upper,f"{star_sed}_sphinx",hdf5_path_in,datasource,nk)
                    write_py(f'{case_name}_sw','sw',wnedges_lower,wnedges_upper,f"{star_sed}_sphinx",hdf5_path_in,datasource,nk)
                    os.system(f"cp tools.py slurms")
                    os.system(f"cp group_generate_co2.py slurms")
                    
                    case_name_list.append(case_name)
            slurm_name = f'{datasource}_{star_sed}'
            write_slurm_group(case_name_list,slurm_name,hdf5_path_in)
            #if not ask_for_yes_no():
            #    exit()
            os.chmod(os.path.join(slurms_root,f'{slurm_name}.sh'),0o755)
            os.chdir(slurms_root)
            os.system(f'sbatch {slurm_name}.sh')