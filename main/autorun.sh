#!/bin/bash
#for i in {1..3}
#do
#  echo "RUN $i START"
#  python ./main/preempt.py --size=64 --K=20 --bs=256 --d=6720 --a=0.90 -e
#  echo "RUN $i END"
#done
i=0

while [ $(ls -l ./runs/ | grep -c ^d) -lt 3 ]
do
  echo "RUN $i START"
  python ./main/preempt.py --size=64 --K=20 --bs=256 --d=6720 --a=0.90 -f -d -e
  echo "RUN $i END"
  i=$[$i+1]
done
