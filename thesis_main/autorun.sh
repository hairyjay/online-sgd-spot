#!/bin/bash
for i in {1..12}
do
  echo "RUN $i START"
  python ./main/preempt.py --size=64 --K=20 --bs=256 -e
  echo "RUN $i END"
done
