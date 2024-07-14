# Shortest Path Biased Backpressure Routing Using Link Features and Graph Neural Networks

This repository is for the source code for paper "Biased Backpressure Routing Using Link Features and Graph Neural Networks" submitted to IEEE TMLCN (preprint will appear soon), and its preliminary conference publications
- Zhongyuan Zhao, Gunjan Verma, Ananthram Swami, Santiago Segarra, "Enhanced Backpressure with Wireless Link Features," IEEE CAMSAP 2023, Herradura, Costa Rica, 2023, pp. 271-275, doi: 10.1109/CAMSAP58249.2023.10403470. [Preprint](https://arxiv.org/pdf/2310.04364), [IEEEXplore](https://ieeexplore.ieee.org/document/10403470), [slides](https://zhongyuanzhao.com/files/BiasBP-CAMSAP2023-15min.pdf), [intro](https://zhongyuanzhao.com/publications/2023-09-26-enhanced-sp-backpressure.html)
- Zhongyuan Zhao, Bojan Radojičić, Gunjan Verma, Ananthram Swami, Santiago Segarra, "Delay-aware Backpressure Routing Using Graph Neural Networks," IEEE ICASSP 2023, Rhodes Island, Greece, 2023, pp. 1-5, doi: 10.1109/ICASSP49357.2023.10095267. [Preprint](https://arxiv.org/pdf/2211.10748.pdf), [IEEEXplore](https://ieeexplore.ieee.org/document/10095267), [slides](https://zhongyuanzhao.com/files/BiasBP-ICASSP2023-8min-upload.pdf), [poster](https://zhongyuanzhao.com/files/BiasBP-ICASSP2023-8min-poster.pdf), [video](https://youtu.be/dRTVzpKXrBs), [Intro](https://zhongyuanzhao.com/publications/2022-11-19-link-duty-cycle-backpressure.html)

Additional test results supplemental to the journal paper can be found [here](/doc/supplement.pdf).

## Abstract

To reduce the latency of Backpressure (BP) routing in wireless multi-hop networks, we propose to enhance the existing shortest path-biased BP (SP-BP) and sojourn time-based backlog metrics, since they introduce no additional time step-wise signaling overhead to the basic BP.
Rather than relying on hop-distance, we introduce a new edge-weighted shortest path bias built on the scheduling duty cycle of wireless links, which can be predicted by a graph convolutional neural network based on the topology and traffic of wireless networks.
Additionally, we tackle three long-standing challenges associated with SP-BP: optimal bias scaling, efficient bias maintenance, and integration of delay awareness. 
Our proposed solutions inherit the throughput optimality of the basic BP, as well as its practical advantages of low complexity and fully distributed implementation. 
Our approaches rely on common link features and introduces only a one-time constant overhead to previous SP-BP schemes, or a one-time overhead linear in the network size to the basic BP.
Numerical experiments show that our solutions can effectively address the major drawbacks of slow startup, random walk, and the last packet problem in basic BP, improving the end-to-end delay of existing low-overhead BP algorithms under various settings of network traffic, interference, and mobility.

## Environment setup

It is recommended to use docker to setup the software environment for convenience. We use Ubuntu Linux 20.04 LTS for demonstration, you may need to use different commands for other OS.

Step 1: install and setup docker, as well as your account, if you haven't done so. 

Step 2: Pull tensorflow 2 with jupyter docker image

`docker pull tensorflow/tensorflow:2.9.1-gpu-jupyter` 

Step 3: Launch your docker container from the image

`docker run --user $(id -u):$(id -g) -it --rm -v ~/dutyBP:/tf/biasBP -w /tf/biasBP --gpus all tensorflow/tensorflow:2.9.1-gpu-jupyter`

Step 4: Install additional packages from the jupyter command terminal

`pip3 install -r requirements_exact.txt`

Step 5: In your command line, commit the changes made to the docker container

`docker container ls`

Find the name of your docker container, for example, `festive_einstein`, then commit the changes 

`docker commit festive_einstein tensorflow/tensorflow:2.9.1-gpu-jupyter`

You don't have to re-launch the container to run the code, it is for the convenience of the next time you launch the container.

## How to run the code

The following steps demonstrate how to run the code referring to the journal paper "Biased Backpressure Routing Using Link Features and Graph Neural Networks".

The scripts are ran from the root of this repository.

### Step 1: data generation (optional)

You can skip this step if using the given data sets

```bash 
bash bash/data_gen.sh
```

### Step 2: GNN training
You can skip this step if you are only to test the trained model in `./model/`

```bash 
python3.9 src/gnn_training_sch.py --training_set=PPm8 --learning_rate=0.0001 --weight_decay=0.001 --T=200 --datapath=./data/data_poisson_train --num_layer=5 --opt=5
```

To train the MLP-based model, comment out line 7 and uncomment line 8 in `gnn_training_sch.py`

The training converges in 5 epochs (about 5 hours on a workstation with a specification of 32GB memory, 8 cores, and Geforce GTX 1070 GPU.)

### Step 3: testing

The test scripts here is designed for parallel execution a workstation with 48 CPU cores and 256GB RAM, if your computer has fewer CPU cores or RAM, it is advised to set the `max_jobs` accordingly in the bash scripts to avoid deadlock. 

You can reduce or enlarge the test scope by changing `for radius in 0.0 1.0 ; do` in the following bash scripts, the manuscript only show results for `radius=1.0`. 

Experiment 1: optimal bias scaling in Fig. 4 

```bash
bash bash/test_100_10_mpx.sh 
``` 


Experiment 2: Section VII-B-1) test on all streaming traffic across different network sizes in Fig. 5(a)

```bash
bash bash/test_100_10_regular.sh 
``` 


Experiment 3: Section VII-B-2) test on all bursty traffic across different network sizes in Figs. 5(b) & 5(c)

```bash
bash bash/test_100_10_burst.sh 
``` 


Experiment 4: Section VII-B-3) test on 50/50 mixed streaming and bursty traffic across different network sizes in Figs. 6

```bash
bash bash/test_100_10_mixed.sh  
``` 

For additional curves for MLP-based routing schemes

```bash
bash bash/test_100_10_mixed_mlp.sh  
``` 

Experiment 5: Section VII-C network throughput across various traffic loads with all streaming flows in Fig. 7

```bash
bash bash/test_100_10_lambda.sh 
``` 

Notice that different routing schemes are coded as two digits numbers, some are experimental schemes not appear in the paper. 
The definition of the routing scheme code is as follows:

```python
lgd_basic = {
    0: 'BP',
    1: 'SP-Hop',
    4: 'VBR',
    5: r'SP-$\bar{r}/(xr)$',
    6: r'SP-$1/x$',
    7: r'EDR-$\bar{r}$',
    8: 'MBP-4',
    9: r'SP-$1/r$',
}
lgd_add = {
    0: '',
    1: '-ph',
    2: '-expQ',
    3: '-min-expQ',
    4: '-min',
    5: '-SJB',
    6: '-HOL',
}
lgds = {}
for i in lgd_basic.keys():
    root = lgd_basic[i]
    for j in lgd_add.keys():
        app = lgd_add[j]
        key = j*10 + i
        lgds[key] = root + app
```

Results are visualized in `results_plot.ipynb` and addtional results for MLP test in `results_plot-MLP.ipynb`
