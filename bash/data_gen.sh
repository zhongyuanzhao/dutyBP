#!/bin/bash


gtype='poisson'
datapath="./data/data_poisson_train"
echo "submit task ${datapath}";
python3.9 src/data_generation_poisson.py --datapath=${datapath} --gtype=${gtype} --size=100 --seed=100 --m=8;

size=10
seed=500

datapath="./data/data_${gtype}_${size}"
echo "submit task ${datapath}";
python3.9 src/data_generation_poisson.py --datapath=${datapath} --gtype=${gtype} --size=${size} --seed=${seed} --m=8;

datapath="./data/data_${gtype}_d12_${size}"
echo "submit task ${datapath}";
python3.9 src/data_generation_poisson.py --datapath=${datapath} --gtype=${gtype} --size=${size} --seed=${seed} --m=12;

echo "Done"