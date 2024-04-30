#!/bin/bash
echo "running {END} jacobian simulations" >> {InversionPath}/imi_output.log

# remove error status file if present
rm -f .error_status_file.txt

sbatch --array={START}-{END}{JOBS} --mem $RequestedMemory \
-c $RequestedCPUs \
-t $RequestedTime \
-p $SchedulerPartition \
-W run_jacobian_simulations.sh
