#!/bin/bash
export JOB_ID="$1"
export HOSTNAMES=$(scontrol show job "$JOB_ID" | grep ' NodeList=' | grep -v '(null)' | tail -n 1 | awk -F= '{print $2}')

echo "Sending SIGTERM"
scontrol show hostnames $HOSTNAMES | while read SSH_HOST
do
     echo "Sending SIGTERM to $SSH_HOST..."
     bash -c "ssh $SSH_HOST pkill -SIGTERM -U $USER" &
done

echo "Waiting 35 secs..."
sleep 35

echo "Sending SIGKILL"
scontrol show hostnames $HOSTNAMES | while read SSH_HOST
do
     echo "Sending SIGKILL to $SSH_HOST..."
     bash -c "ssh $SSH_HOST pkill -SIGKILL -U $USER" &
done

echo "Killing job $JOB_ID"
/opt/software/slurm-17.11.3-2/bin/scancel "$JOB_ID"
