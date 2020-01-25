#!/bin/bash

NUM_CLIENTS=25
NUM_JOBS="\j"  # The prompt escape for number of jobs currently running

trap ctrl_c INT

function ctrl_c() {
  pkill -P $$
}

for ((i=0; i<NUM_CLIENTS; i++)); do
  while (( ${NUM_JOBS@P} >= NUM_CLIENTS )); do
    wait -n
  done
  sleep .05
  python3 app/client.py &
done
wait
exit
