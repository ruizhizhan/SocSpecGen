#!/bin/bash 
#SBATCH -J codex_no_mtckd_L
#SBATCH -p wzhcnormal
#SBATCH -n 1
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --time 00:05:00
#SBATCH --comment=BASE
###SBATCH -o std.out.%j
###SBATCH -e std.err.%j
#SBATCH -o /work/home/ac9b0k6rio/SocSpecGen/out/std.out.%j
#SBATCH -e /work/home/ac9b0k6rio/SocSpecGen/out/std.err.%j


#########################################################
### Slurm scripts for Sugon Portal_5.0 of BASE
### Version 1.0    |  2019-09-09  |  created by Zhang Guoliang
### Version 1.0.1  |  2020-11-24  |  modified by Zhang Guoliang
#########################################################

### Get parameters from GUI

MIDFILE_DIR=/work/home/ac9b0k6rio/SocSpecGen/.portal
source $MIDFILE_DIR/job_portal.var
source $MIDFILE_DIR/job_interface.var

### Set basic var   ### MARK_slurm2pbs

JOBID=$SLURM_JOB_ID                                  ### slurm2pbs
NP=$SLURM_NPROCS                                     ### slurm2pbs
NNODE=`srun hostname | sort -V | uniq | wc -l`          ### slurm2pbs

LOG_FILE=$WORK_DIR/job_${JOB_NAME}_${JOBID}.log
HOST_FILE=$WORK_DIR/job_${JOB_NAME}_${JOBID}_${NP}c_${NNODE}n.ma 

srun hostname | sort -V | uniq -c |awk '{print $2":"$1}' > $HOST_FILE  ### slurm2pbs

### Write basic job infomations

echo -e "The start time is: `date +"%Y-%m-%d %H:%M:%S"` \n" | tee -a $LOG_FILE 
echo -e "My job ID is: $JOBID \n" | tee -a $LOG_FILE  
echo -e "The total cores is: $NP \n" | tee -a $LOG_FILE 
echo -e "The hosts is: \n" | tee -a $LOG_FILE
cat $HOST_FILE | tee -a $LOG_FILE
echo -e "\n"  | tee -a $LOG_FILE 

### Run APP

  # MARK_CMD  #Don't delete this line!!!
#!/bin/bash
## job script created by Gridview Jobmanager.
bash /work/home/ac9b0k6rio/SocSpecGen/apply_no_mtckd_L_20260529_122413.sh

  # MARK_BASH #Don't delete this line!!!

if [ $? -ne 0 ]; then
  exit 1
fi
echo The end time is: `date +"%Y-%m-%d %H:%M:%S"` | tee -a $LOG_FILE   

