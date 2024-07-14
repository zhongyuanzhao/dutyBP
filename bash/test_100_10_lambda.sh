#!/bin/bash

source .venv/bin/activate ;

cd ./src;

size=10
opts='1,4,61,71,57,47,67,77,35,55,5'
# opts='35,55,5'
T=1000
max_jobs=40


for gtype in 'poisson' ; do
	datapath="../data/data_${gtype}_${size}"
    # for radius in 0.0 1.0; do
    for radius in 1.0; do
        # for opts in '1,5' '61,71' '47,57' '67,77' '35' ; do
        # for opts in '0' '7' '37' '6' '26' '5' '25' '49' '39' '60' '67' ; do
        # for opts in '0' '7' '37' '6' '26' '5' '25'  ; do
        # for opts in '49' '39' '60' '67' ; do
        for opts in '0' '50' '60' '7' '27' '67' '57' '5' '25' '6' '26' '9' '39' '49' '35' '45' '4' '44'; do
            echo "submit task opts ${opts}, radius ${radius}";
            python3.9 bp_test_sch_lambda.py --arrival_max=12.1 --arrival_min=3.0 --arrival_step=1.0 --training_set=PPm8 --datapath=${datapath} --opts=${opts} --out=../out --learning_rate=0.0001 --weight_decay=0.01 --radius=${radius} --gtype=${gtype} --T=${T} &

            python3.9 bp_test_sch_lambda.py --arrival_max=3.0 --arrival_min=0.1 --arrival_step=0.3 --training_set=PPm8 --datapath=${datapath} --opts=${opts} --out=../out --learning_rate=0.0001 --weight_decay=0.01 --radius=${radius} --gtype=${gtype} --T=${T} &

            # Keep track of the number of running jobs
            running_jobs=$(jobs -p | wc -l)

            # If the number of running jobs is greater than or equal to 7, wait for one of them to finish
            if [[ $running_jobs -ge ${max_jobs} ]]; then
                wait -n
            fi

        done
    done
done
