#!/bin/bash
#SBATCH --time=01-00:00
#SBATCH --account=def-bengioy
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=0
#SBATCH --nodes=1

## ----------------------------------------------
## Variables
export COMMIT_HASH="7b694e9d2522e8f2e60bf1861b4c1e416e7351d9"
export EXPERIMENT_COMMAND="python -u diplomacy_research/scripts/build_dataset.py --filter order_based/no_press_all"
export WALL_TIME_HRS=24.0
export AUTO_KILL=0

## ----------------------------------------------
## Running experiment
printf -v ARRAY_TASK_ID "%05d" ${SLURM_ARRAY_TASK_ID:-0}
export EXPERIMENT_FOLDER=/scratch/paquphil/
export MAX_DURATION=`bc <<< "$WALL_TIME_HRS * 3600 - 240"`
export KILL_DURATION=`bc <<< "$WALL_TIME_HRS * 3600 - 120"`
export LAUNCH_FILE="$EXPERIMENT_FOLDER/.launch.$ARRAY_TASK_ID.sh"
export COMMIT_FILE="$EXPERIMENT_FOLDER/.commit_hash"

# Detecting commit hash and storing it for future use
if [ -f $COMMIT_FILE ]; then
    read -r COMMIT_HASH < "$COMMIT_FILE"
fi
echo "Detected: Commit Hash - ${COMMIT_HASH::7}"

# Creating exit script
exit_script() {
    trap - SIGINT SIGTERM
    echo "Received Ctrl-C / SIGTERM. Exiting..."
    kill -- -"$(ps -o pgid= -p "$$")"
    if [ $AUTO_KILL -eq 1 ]; then
        sleep 30
        echo "Killing all remaining processes..."
        pkill -SIGKILL -U $USER
    fi
    exit 0
}
trap exit_script SIGINT SIGTERM

# Executing experiment
echo "Launching experiment on ${HOSTNAME%%.*}"
cp /scratch/paquphil/singularity/research/*${COMMIT_HASH::7}.sif* $SLURM_TMPDIR
cd $SLURM_TMPDIR
for i in *.gz; do
    [ -f "$i" ] || break
    gunzip "$i"
done
export CLEAN_PATH=/opt/software/singularity-3.2/bin:/opt/software/bin:/opt/software/slurm/bin:/usr/local/bin:/usr/bin:/usr/local/sbin:/usr/sbin
(PATH=$CLEAN_PATH; timeout -s SIGTERM -k $KILL_DURATION $MAX_DURATION singularity run --bind $EXPERIMENT_FOLDER:/work_dir --bind /scratch --bind /cvmfs *${COMMIT_HASH::7}.sif.cpu $EXPERIMENT_COMMAND)
