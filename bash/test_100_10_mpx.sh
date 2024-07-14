#!/bin/bash

source .venv/bin/activate ;

cd ./src;

today=$(date +"%Y-%m-%d") ;
size=10
opts='45,46,49'
T=500
radius=1.0
max_jobs=48

for gtype in 'poisson' ; do
	datapath="../data/data_${gtype}_${size}"
    for mpx in 0.5 0.7 0.8 0.9 1.0 1.1 1.2 1.3 1.5; do
        echo "submit task opts ${opts}, radius ${radius}, mpx ${mpx}";
        python3 bp_test_sch_burst_const.py --mpx=${mpx} --training_set=PPm8 --datapath=${datapath} --opts=${opts} --out=../out --learning_rate=0.0001 --weight_decay=0.01 --radius=${radius} --gtype=${gtype} --T=${T} > ../out/output_r_${radius}_opts_${opts}_${today}_burst_mpx.txt &
        python3 bp_test_sch_const.py --mpx=${mpx} --training_set=PPm8 --datapath=${datapath} --opts=${opts} --out=../out --learning_rate=0.0001 --weight_decay=0.01 --radius=${radius} --gtype=${gtype} --T=${T} > ../out/output_r_${radius}_opts_${opts}_${today}_regular_mpx.txt &

        # Keep track of the number of running jobs
        running_jobs=$(jobs -p | wc -l)

        # If the number of running jobs is greater than or equal to 7, wait for one of them to finish
        if [[ $running_jobs -ge ${max_jobs} ]]; then
            wait -n
        fi

    done
done

deactivate;