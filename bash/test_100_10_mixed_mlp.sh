#!/bin/bash

source .venv/bin/activate ;

cd ./src;

size=10
opts='0,4,60,50,57,47,67,37,25,45,5'
# opts='35,55,5'
T=1000
max_jobs=40


for gtype in 'poisson' ; do
    datapath="../data/data_${gtype}_${size}"
    for radius in 0.0 1.0; do
        for opts in '5' '6' '46' '35' '45' ; do
            echo "submit task opts ${opts}, radius ${radius}";
            python3.9 bp_test_sch_mixed_traffic_mlp.py --training_set=PPm8 --datapath=${datapath} --opts=${opts} --out=../out --learning_rate=0.0001 --weight_decay=0.01 --radius=${radius} --gtype=${gtype} --T=${T} &

            # Keep track of the number of running jobs
            running_jobs=$(jobs -p | wc -l)

            # If the number of running jobs is greater than or equal to 7, wait for one of them to finish
            if [[ $running_jobs -ge ${max_jobs} ]]; then
                wait -n
            fi
            
        done
    done
done

deactivate;