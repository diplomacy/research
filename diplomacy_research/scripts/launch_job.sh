#!/bin/bash
#SBATCH --time=00-03:00
#SBATCH --account=rpp-bengioy
#SBATCH --nodes=1
#SBATCH --mincpus=24
#SBATCH --mem=0
#SBATCH --gres=gpu:4
#SBATCH --array=0-7%1
#SBATCH --job-name=launch_job.sh
#SBATCH --exclude=cdr245,cdr876
#SBATCH --wait-all-nodes=1
#SBATCH --mail-user=pcpaquette@gmail.com
#SBATCH --mail-type=ALL

## To debug use the following salloc
## salloc --time=00-00:59 --account=rpp-bengioy --nodes=1 --mincpus=24 --mem=0 --gres=gpu:4

## ----------------------------------------------
## Variables
export COMMIT_HASH=""
export EXPERIMENT_COMMAND="python -u diplomacy_research/models/policy/order_based/train.py"
export WALL_TIME_HRS=3.0
export AUTO_KILL=0

## ----------------------------------------------
## Running experiment
printf -v ARRAY_TASK_ID "%05d" ${SLURM_ARRAY_TASK_ID:-0}
export EXPERIMENT_FOLDER=$SLURM_SUBMIT_DIR
export LOCK_FILE="$EXPERIMENT_FOLDER/.$SLURM_JOB_ID.$ARRAY_TASK_ID.${HOSTNAME%%.*}.lock"
export MAX_DURATION=`bc <<< "$WALL_TIME_HRS * 3600 - 240"`
export KILL_DURATION=`bc <<< "$WALL_TIME_HRS * 3600 - 120"`
export LAUNCH_FILE="$EXPERIMENT_FOLDER/.launch.$ARRAY_TASK_ID.sh"
export KILL_FILE="$EXPERIMENT_FOLDER/.kill.$ARRAY_TASK_ID.sh"
export KILL_COMMAND="ssh $SLURM_SUBMIT_HOST $KILL_FILE &"
export COMMIT_FILE="$EXPERIMENT_FOLDER/.commit_hash"

# Detecting commit hash and storing it for future use
if [ -f $COMMIT_FILE ]; then
    read -r COMMIT_HASH < "$COMMIT_FILE"
else
    echo "$COMMIT_HASH" > "$COMMIT_FILE"
fi
echo "Detected: Commit Hash - ${COMMIT_HASH::7}"

# Acquiring lock
if [ -f "$LOCK_FILE" ]; then
   echo "Lock file already exists. - Exiting"
   exit 0
else
   echo "Acquired lock $LOCK_FILE"
   touch $LOCK_FILE
fi

# Creating exit script
exit_script() {
    trap - SIGINT SIGTERM
    echo "Received Ctrl-C / SIGTERM. Exiting..."
    if [ -z "$LAUNCHED_BY_SSH" ]; then
        bash -c "$KILL_COMMAND" &
    fi
    kill -- -"$(ps -o pgid= -p "$$")"
    if [ $AUTO_KILL -eq 1 ]; then
        sleep 30
        echo "Killing all remaining processes..."
        pkill -SIGKILL -U $USER
    fi
    exit 0
}
trap exit_script SIGINT SIGTERM

# Killing all running python processes first
if [ $AUTO_KILL -eq 1 ]; then
    pkill --signal SIGKILL python -U $USER
fi

# Making sure other nodes are started
if [ -z "$LAUNCHED_BY_SSH" ]; then
    echo "Waiting for other nodes..."
    sleep 30
    echo "Checking if all nodes have started..."

    # Building a launch script
    if [ ! -f "$LAUNCH_FILE" ]; then
        echo "Building launch script..."
        touch $LAUNCH_FILE
        set | grep "SLURM" | while read VAR; do echo "export $VAR"; done >> $LAUNCH_FILE
        set | grep "CUDA_VISIBLE" | while read VAR; do echo "export $VAR"; done >> $LAUNCH_FILE
        set | grep "GPU_DEVICE" | while read VAR; do echo "export $VAR"; done >> $LAUNCH_FILE
        echo 'export SLURM_STEP_NODELIST=${HOSTNAME%%.*}' >> $LAUNCH_FILE
        echo 'export SLURM_TOPOLOGY_ADDR=${HOSTNAME%%.*}' >> $LAUNCH_FILE
        echo 'export SLURMD_NODENAME=${HOSTNAME%%.*}' >> $LAUNCH_FILE
        echo 'export LAUNCHED_BY_SSH=1' >> $LAUNCH_FILE
        echo '' >> $LAUNCH_FILE
        cat /var/spool/slurmd/job$SLURM_JOB_ID/slurm_script >> $LAUNCH_FILE
        chmod +x $LAUNCH_FILE
    fi

    # Displaying nodes detected
    echo "Found the following hosts: ...."
    scontrol show hostnames $SLURM_JOB_NODELIST | while read SSH_HOST
    do
        if ping -c 1 "$SSH_HOST" &> /dev/null; then
            echo "Host: $SSH_HOST - Status: Online"
        else
            echo "Host: $SSH_HOST - Status: OFFLINE?"
        fi
    done

    # Building a kill script
    # To cancel all nodes properly when we use scancel.
    if [ ! -f "$KILL_FILE" ] && [ $AUTO_KILL -eq 1 ]; then
        echo "Building kill script..."
        touch $KILL_FILE
        echo "#!/bin/bash" >> $KILL_FILE

        # Killing hosts started by SSH first, then hosts started by Slurm
        scontrol show hostnames $SLURM_JOB_NODELIST | while read SSH_HOST
        do
            if [ "${HOSTNAME%%.*}" != "$SSH_HOST" ]; then
                echo "ssh $SSH_HOST pkill -SIGTERM -U $USER &" >> $KILL_FILE
            fi
        done
        echo "ssh ${HOSTNAME%%.*} pkill -SIGTERM -U $USER &" >> $KILL_FILE

        # Waiting 60 secs and sending SIGKILL
        echo "sleep 60" >> $KILL_FILE

        # Sending SIGKILL
        scontrol show hostnames $SLURM_JOB_NODELIST | while read SSH_HOST
        do
            if [ "${HOSTNAME%%.*}" != "$SSH_HOST" ]; then
                echo "ssh $SSH_HOST pkill -SIGKILL -U $USER &" >> $KILL_FILE
            fi
        done
        echo "ssh ${HOSTNAME%%.*} pkill -SIGKILL -U $USER &" >> $KILL_FILE
        chmod +x $KILL_FILE
    fi

    # Starting other nodes if they have not acquired their lock
    scontrol show hostnames $SLURM_JOB_NODELIST | while read SSH_HOST
    do
        if [ ! -f "$EXPERIMENT_FOLDER/.$SLURM_JOB_ID.$ARRAY_TASK_ID.$SSH_HOST.lock" ]; then
            echo "Executing launch script via SSH on $SSH_HOST"
            ssh $SSH_HOST timeout -s SIGTERM -k $KILL_DURATION $MAX_DURATION $LAUNCH_FILE &
        fi
    done
fi

# Executing experiment
echo "Launching experiment on ${HOSTNAME%%.*}"
cp /scratch/paquphil/singularity/research/*${COMMIT_HASH::7}.sif* $SLURM_TMPDIR
cd $SLURM_TMPDIR
for i in *.gz; do
    [ -f "$i" ] || break
    gunzip "$i"
done
mkdir -p $EXPERIMENT_FOLDER/diplomacy
cp -rus /scratch/paquphil/diplomacy/*.hdf5 $EXPERIMENT_FOLDER/diplomacy 2> /dev/null
cp -rus /scratch/paquphil/diplomacy/*.pb $EXPERIMENT_FOLDER/diplomacy 2> /dev/null
cp -rus /scratch/paquphil/diplomacy/*.pbz $EXPERIMENT_FOLDER/diplomacy 2> /dev/null
cp -rus /scratch/paquphil/diplomacy/*.pkl $EXPERIMENT_FOLDER/diplomacy 2> /dev/null
cp -rus /scratch/paquphil/diplomacy/*.rdb $EXPERIMENT_FOLDER/diplomacy 2> /dev/null
export CLEAN_PATH=/opt/software/singularity-3.2/bin:/opt/software/bin:/opt/software/slurm/bin:/usr/local/bin:/usr/bin:/usr/local/sbin:/usr/sbin
(PATH=$CLEAN_PATH; timeout -s SIGTERM -k $KILL_DURATION $MAX_DURATION singularity run --nv --bind $EXPERIMENT_FOLDER:/work_dir --bind /scratch --bind /cvmfs *${COMMIT_HASH::7}.sif $EXPERIMENT_COMMAND | tee -a $EXPERIMENT_FOLDER/output.$ARRAY_TASK_ID.${HOSTNAME%%.*}.${COMMIT_HASH::7}.log)
