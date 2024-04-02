#!/bin/bash

# Common shell function for the IMI
# Functions available in this file include:
#   - submit_job
#       - submit_slurm_job
#       - submit_pbs_job
#   - print_stats
#   - imi_failed 
#   - ncmax 
#   - ncmin 

# Description: 
#   Submit a job with default ICI settings using either SBATCH or PBS
# Usage:
#   submit_job $SchedulerType $JobArguments
submit_job() {
    if [[ $1 = "slurm" ]]; then
        submit_slurm_job "${@:2}"
    elif [[ $1 = "PBS" ]]; then
        submit_pbs_job "${@:2}"
    else
        echo "Scheduler type $1 not recognized."
    fi
}

# Description: 
#   Submit a job with default ICI settings using SBATCH
# Usage:
#   submit_slurm_job $JobArguments
submit_slurm_job() {
    sbatch --mem $SimulationMemory \
        -c $SimulationCPUs \
        -t $RequestedTime \
        -p $SchedulerPartition \
        -W ${@}; wait;
}

# Description: 
#   Submit a job with default ICI settings using PBS
# Usage:
#   submit_pbs_job $JobArguments
submit_pbs_job() {
    qsub -l nodes=1 \
        -l mem="$SimulationMemory" \
        -l ncpus=$SimulationCPUs \
        -l walltime=$RequestedTime \
        -l site=needed=$SitesNeeded \
        -l model=ivy \
        -sync y ${@}; wait;
}

# Description: 
#   Print runtime stats based on existing variables
# Usage:
#   print_stats
print_stats() {
    printf "\nRuntime statistics (s):"
    printf "\n Setup     : $( [[ ! -z $setup_end ]] && echo $(( $setup_end - $setup_start )) || echo 0 )"
    printf "\n Spinup     : $( [[ ! -z $spinup_end ]] && echo $(( $spinup_end - $spinup_start )) || echo 0 )"
    printf "\n Jacobian     : $( [[ ! -z $jacobian_end ]] && echo $(( $jacobian_end - $jacobian_start )) || echo 0 )"
    printf "\n Inversion     : $( [[ ! -z $inversion_end ]] && echo $(( $inversion_end - $inversion_start )) || echo 0 )"
    printf "\n Posterior     : $( [[ ! -z $posterior_end ]] && echo $(( $posterior_end - $posterior_start )) || echo 0 )\n\n"
}

# Description: Print error message for if the IMI fails
#   Copy output file to output directory if it exists
# Usage:
#   imi_failed
imi_failed() {
    file=`basename "$0"`
    printf "\nFATAL ERROR on line number ${1} of ${file}: IMI exiting."
    if [ -d "${OutputPath}/${RunName}" ]; then
        cp "${InversionPath}/imi_output.log" "${OutputPath}/${RunName}/imi_output.log"
    fi
    exit 1
}

# Description: Print max value of given variable in netCDF file
#   Returns int if only trailing zeros, float otherwise
# Usage:
#   ncmax <variable> <netCDF file path>
ncmax() {
    python -c "import sys; import xarray;\
    print('%g' % xarray.open_dataset(sys.argv[2])[sys.argv[1]].max())" $1 $2
}

# Description: Print min value of given variable in netCDF file
#   Returns int if only trailing zeros, float otherwise
# Usage:
#   ncmax <variable> <netCDF file path>
ncmin() {
    python -c "import sys; import xarray; \
    print('%g' % xarray.open_dataset(sys.argv[2])[sys.argv[1]].min())" $1 $2
}
