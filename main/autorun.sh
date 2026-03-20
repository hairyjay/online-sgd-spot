#!/bin/bash
#for i in {1..3}
#do
#  echo "RUN $i START"
#  python -m main.preempt --size=64 --K=20 --bs=256 --d=6720 --a=0.90 -e
#  echo "RUN $i END"
#done
set -e
# i=0

# while [ $(ls -l ./runs/ | grep -c ^d) -lt 4 ]
# do
#   echo "RUN $i START"
  #python -m main.preempt --size=64 --K=20 --bs=256 --t=0.008 --d=8400 --J=250000 --a=0.9 --distr=uniform -e -d
#   python -m main.preempt --size=64 --K=20 --bs=256 --t=0.008 --d=4200 --J=125000 --a=0.9 -e --target=0.9 -f=6000 d --drift-start 1000 1000 1000 1000 --drift-time 10 100 1000 10000 --drift-cats 5 5 5 5
#   echo "RUN $i END"
#   i=$[$i+1]
# done

echo "RUN 1 START"
python -m main.preempt --size=64 --K=20 --bs=256 --t=0.008 --d=4200 --J=125000 --a=0.9 -e --target=0.9 -f=4500 d --drift-start 1000 1000 1000 1000 --drift-time 10 100 1000 10000 --drift-cats 5 5 5 5
echo "RUN 1 END"

echo "RUN 2 START"
python -m main.preempt --size=64 --K=20 --bs=256 --t=0.008 --d=4200 --J=125000 --a=0.9 -e --target=0.9 -f=4500 d --drift-start 1000 1110 1310 2410 --drift-time 10 100 1000 10000 --drift-cats 5 5 5 5
echo "RUN 2 END"

echo "RUN 3 START"
python -m main.preempt --size=64 --K=20 --bs=256 --t=0.008 --d=4200 --J=125000 --a=0.9 -e --target=0.9 -f=4500 d --drift-start 1450 1400 1200 1000 --drift-time 10 100 1000 10000 --drift-cats 5 5 5 5
echo "RUN 3 END"

# while [ $(ls -l ./runs/ | grep -c ^d) -lt 5 ]
# do
#   echo "RUN $i START"
#   #python -m main.preempt --size=64 --K=20 --bs=256 --t=0.008 --d=8400 --J=250000 --a=0.9 --distr=uniform -e -d
#   python -m main.preempt --size=64 --K=20 --bs=256 --t=0.008 --d=5000 --J=150000 --a=0.9 --distr=uniform -e d --drift-start=1800 --drift-time=0
#   echo "RUN $i END"
#   i=$[$i+1]
# done

#python -m main.preempt --size=64 --K=40 --bs=128 --t=0.004 --d=4400 --J=250000 --a=0.8 --distr=dirichlet -e -d

#OLD LINES
  #python -m main.preempt --size=64 --K=20 --bs=256 --d=6264 --a=1 -e -f --optimizer=adam #EMNIST ADAM ONDEMAND
  #python -m main.preempt --size=64 --K=20 --bs=256 --d=6577 --a=0.9 -e -f --optimizer=adam #EMNIST ADAM
  #python -m main.preempt --size=64 --K=10 --bs=256 --d=1536 --a=1 -e -f --J=48000 --target=0.97 --dataset=imnist #InfiMNIST ONDEMAND
  #python -m main.preempt --size=64 --K=10 --bs=256 --d=1613 --a=0.9 -e -d --J=48000 --target=0.97 --dataset=imnist #InfiMNIST
  #python -m main.preempt --size=64 --K=4 --bs=256 --d=2920 --a=1 -e --J=91250 --target=0.6 --dataset=cifar #CIFAR ONDEMAND
  #python -m main.preempt --size=64 --K=4 --bs=256 --d=3066 --a=0.9 -e -d --J=91250 --target=0.6 --dataset=cifar #CIFAR