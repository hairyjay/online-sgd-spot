#!/bin/bash
#for i in {1..3}
#do
#  echo "RUN $i START"
#  python -m main.preempt --size=64 --K=20 --bs=256 --d=6720 --a=0.90 -e
#  echo "RUN $i END"
#done
i=0

while [ $(ls -l ./runs/ | grep -c ^d) -lt 12 ]
do
  echo "RUN $i START"
  python -m main.preempt --size=64 --K=10 --bs=256 --d=6264 --a=1 -e
  echo "RUN $i END"
  i=$[$i+1]
done

#OLD LINES
  #python -m main.preempt --size=64 --K=20 --bs=256 --d=6264 --a=1 -e -f --optimizer=adam #EMNIST ADAM ONDEMAND
  #python -m main.preempt --size=64 --K=20 --bs=256 --d=6577 --a=0.9 -e -f --optimizer=adam #EMNIST ADAM
  #python -m main.preempt --size=64 --K=10 --bs=256 --d=1536 --a=1 -e -f --J=48000 --target=0.97 --dataset=imnist #InfiMNIST ONDEMAND
  #python -m main.preempt --size=64 --K=10 --bs=256 --d=1613 --a=0.9 -e -d --J=48000 --target=0.97 --dataset=imnist #InfiMNIST
  #python -m main.preempt --size=64 --K=4 --bs=256 --d=2920 --a=1 -e --J=91250 --target=0.6 --dataset=cifar #CIFAR ONDEMAND
  #python -m main.preempt --size=64 --K=4 --bs=256 --d=3066 --a=0.9 -e -d --J=91250 --target=0.6 --dataset=cifar #CIFAR