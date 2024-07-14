import argparse
import tensorflow as tf
import numpy as np
import scipy.io as sio

from backpressure import *
from gnn_routing_agent_sch import GDPGAgent, FLAGS, flags
from qlearning import QLearning

flags.DEFINE_string('opts', '5,6,1,7,0', 'Routing scheme.')
flags.DEFINE_string('root', '..', 'Root dir of project.')
flags.DEFINE_float('radius', 1.0, 'Interference radius.')
flags.DEFINE_string('gtype', 'ba', 'Root dir of project.')
flags.DEFINE_float('mpx', 1.0, 'Multiplier to per-hop distance')

agent = GDPGAgent(FLAGS, 5000)

std = 0.01
batch_size = FLAGS.batch
T = FLAGS.T
cf_radius = FLAGS.radius
datapath = FLAGS.datapath
val_mat_names = sorted(os.listdir(datapath))

opts = FLAGS.opts.split(',')
opts = [int(i) for i in opts]
opts = sorted(opts)
opts_txt = [str(i) for i in opts]
opts_txt = '-'.join(opts_txt)
link_rate_max = 42  # 42
link_rate_min = 10  # 10
link_rate_avg = (link_rate_max + link_rate_min)/2
arrival_max = 10.0  # 1.0
arrival_min = 2.0  # 0.2
arrival_avg = (arrival_min + arrival_max)/2
burst_cutoff = 30
mpx = FLAGS.mpx

output_dir = FLAGS.out
output_csv = os.path.join(output_dir, "test_{}_T_{}_ir_{:.1f}_opts_{}_link-{}_mpx-{:.1f}_burst.csv".format(
    datapath.split("/")[-1], T, cf_radius, opts_txt, link_rate_avg, mpx))
df_res = pd.DataFrame(
    columns=["filename", "seed", "num_nodes", "m", "T", "cf_radius", "cf_degree",
             "opt", "Algo", "num_flows",
             "z",
             "src_delay_raw",
             "src_delay_mean", "src_delay_max", "src_delay_std",
             "est_delay_mean", "est_delay_max", "est_delay_std",
             "delivery_mean", "delivery_max", "delivery_std",
             "active_links",
             "runtime"]
)

lgds = {
    0: 'BP',
    1: 'SP-Hop',
    4: 'VBR',
    5: r'SP-$\bar{r}/(xr)$',
    6: r'SP-$1/x$',
    7: 'EDR-10',
    8: 'MBP-4',
    9: r'SP-$10\bar{r}/r$',
    10: 'BP-ph',
    11: 'SP-Hop-ph',
    14: 'VBR-ph',
    15: r'SP-$\bar{r}/(xr)$-ph',
    16: r'SP-$1/x$-ph',
    17: 'EDR-10-ph',
    18: 'MBP-4-ph',
    19: r'SP-$10\bar{r}/r$-ph',
    20: 'BP-ph-e',
    21: 'SP-Hop-ph-e',
    24: 'VBR-ph-e',
    25: r'SP-$\bar{r}/(xr)$-ph-e',
    26: r'SP-$1/x$-ph-e',
    27: 'EDR-10-ph-e',
    28: 'MBP-4-ph-e',
    29: r'SP-$10\bar{r}/r$-ph-e',
    35: 'exp'
}

# Create fig folder if not exist
modeldir = os.path.join(FLAGS.root, "model")
if not os.path.isdir(modeldir):
    os.mkdir(modeldir)

log_dir = os.path.join(FLAGS.root, "logs")
if not os.path.isdir(log_dir):
    os.mkdir(log_dir)

# Get a list of available GPUs
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)
# Set the number of GPUs to use
num_gpus = len(gpus)
# Set up a MirroredStrategy to use all available GPUs
if num_gpus > 1:
    strategy = tf.distribute.MirroredStrategy(devices=["/gpu:%d" % i for i in range(num_gpus)])
else:
    strategy = tf.distribute.get_strategy() # default strategy
# load models
actor_model = os.path.join(modeldir, 'model_ChebConv_{}_a{}_actor'.format(FLAGS.training_set, 5, 5))

try:
    # Define and compile your model within the strategy scope
    with strategy.scope():
        agent.load(actor_model)
except:
    print("unable to load {}".format(actor_model))


def get_directional_links(adj, link_list, rates):
    assert len(link_list) == rates.size
    rows, cols = adj.nonzero()
    edges = adj.indices
    rates_direct = np.zeros((edges.shape[0], 1))
    for i in range(edges.shape[0]):
        src, dst = rows[i], cols[i]
        if (src, dst) in link_list:
            rates_direct[i, 0] = rates[link_list.index((src, dst))]
        elif (dst, src) in link_list:
            rates_direct[i, 0] = rates[link_list.index((dst, src))]
        else:
            pass
    return rates_direct


# Define tensorboard
# agent.log_init()
gidx = 0

for id in range(len(val_mat_names)):
    filepath = os.path.join(datapath, val_mat_names[id])
    mat_contents = sio.loadmat(filepath)
    net_cfg = mat_contents['network'][0,0]
    # link_rates = mat_contents["link_rate"][0]
    flows_cfg = mat_contents["flows"][0]
    pos_c = mat_contents["pos_c"]

    seed = int(net_cfg['seed'].flatten()[0])
    NUM_NODES = int(net_cfg['num_nodes'].flatten()[0])
    m = net_cfg['m'].flatten()[0]

    if NUM_NODES != 100:
        continue

    # Configuration
    if FLAGS.gtype.lower() == 'poisson':
        bp_env = Backpressure(NUM_NODES, T, seed, m, pos_c, cf_radius=cf_radius, gtype=filepath, trace=True)
    else:
        bp_env = Backpressure(NUM_NODES, T, seed, m, pos_c, cf_radius=cf_radius, gtype=FLAGS.gtype, trace=True)
    if not bp_env.connected:
        print("Unconnected {}".format(val_mat_names[id]))
        continue
    bp_env.queues_init()

    for f_case in range(10):
        # Generate random flows
        np.random.seed(seed*10 + f_case)
        # flows_perc = np.random.randint(15, 30)
        flows_perc = np.random.randint(30, 50)
        num_flows = round(flows_perc / 100 * bp_env.num_nodes)
        nodes = bp_env.graph_c.nodes()
        num_arr = np.random.permutation(nodes)
        arrival_rates = np.random.uniform(arrival_min, arrival_max, (num_flows,))
        link_rates = np.random.uniform(link_rate_min, link_rate_max, size=(bp_env.num_links,))
        bp_env.links_init(link_rates)

        bp_env.clear_all_flows()
        flows = []
        for fidx in range(num_flows):
            src = num_arr[2 * fidx]
            dst = num_arr[2 * fidx + 1]
            bp_env.add_flow(src, dst, rate=arrival_rates[fidx])
            flow = {'src': src, 'dst': dst, 'rate': arrival_rates[fidx]}
            flows.append(flow)

        bp_env.flows_init(cutoff=burst_cutoff)
        # print(filepath)

        # shortest_paths = np.zeros((NUM_NODES, NUM_NODES))
        # bias_matrix = np.zeros_like(bp_env.queue_matrix)
        state, link_shares = agent.foo_train(bp_env.adj_i, link_rates, np.zeros_like(link_rates), train=False)
        # cali_const = np.nanmean(link_shares)
        # cali_const = 0.1
        cali_const = 1.0/link_rate_avg

        for opt in opts:
            starttime = time.time()
            bp_env.queues_init()
            bp_env.flows_reset()
            bp_env.pheromone_init(decay=0.99, unit=0.01)

            if opt % 10 == 5:
                delay_est = np.divide(link_rate_avg, np.multiply(link_rates, link_shares.flatten()))
                for link, delay in zip(bp_env.link_list, delay_est):
                    src, dst = link
                    bp_env.graph_c[src][dst]["delay"] = delay
                shortest_paths = all_pairs_shortest_paths(bp_env.graph_c, weight='delay')
                bias_matrix = shortest_paths
            elif opt % 10 == 6:
                delay_est = np.divide(1.0, link_shares.flatten())
                # delay_est = delay_est * (link_rate_avg/np.mean(delay_est))
                for link, delay in zip(bp_env.link_list, delay_est):
                    src, dst = link
                    bp_env.graph_c[src][dst]["delay"] = delay
                shortest_paths = all_pairs_shortest_paths(bp_env.graph_c, weight='delay')
                bias_matrix = shortest_paths
            elif opt % 10 == 9:
                # delay_est = np.divide(link_rate_avg/cali_const, link_rates)
                delay_est = np.divide(link_rate_avg**2, link_rates)
                for link, delay in zip(bp_env.link_list, delay_est):
                    src, dst = link
                    bp_env.graph_c[src][dst]["delay"] = delay
                shortest_paths = all_pairs_shortest_paths(bp_env.graph_c, weight='delay')
                bias_matrix = shortest_paths
            elif opt % 10 == 2:
                delay_est = np.divide(1.0, link_rates)
                for link, delay in zip(bp_env.link_list, delay_est):
                    src, dst = link
                    bp_env.graph_c[src][dst]["delay"] = delay
                shortest_paths = all_pairs_shortest_paths(bp_env.graph_c, weight='delay')
                bias_matrix = shortest_paths
            elif opt % 10 == 1:
                shortest_paths = all_pairs_shortest_paths(bp_env.graph_c, weight=None)
                bias_matrix = shortest_paths
            elif opt % 10 == 7:
                # EDR: Enhanced Dynamic Backpressure Routing
                shortest_paths = all_pairs_shortest_paths(bp_env.graph_c, weight=None)
                bias_matrix = (1.0/cali_const) * shortest_paths
            elif opt % 10 == 3:  # Q learning bias + SP
                shortest_paths = all_pairs_shortest_paths(bp_env.graph_c, weight=None)
            elif opt % 10 in [0, 8]:
                bias_matrix = np.zeros_like(bp_env.queue_matrix)
            elif opt % 10 == 4:
                # VBR (virtual queue based backpressure routing)
                shortest_paths = all_pairs_shortest_paths(bp_env.graph_c, weight=None)
                bias_matrix = np.zeros_like(shortest_paths)
                for fidx in range(bp_env.num_flows):
                    src = bp_env.src_nodes[fidx]
                    dst = bp_env.dst_nodes[fidx]
                    bias_matrix[:, dst] = 1.0 * 1.2**(bp_env.flows[fidx].arrival_rate/(shortest_paths[:, dst]+0.2))
                    bias_matrix[:, dst] = np.multiply(bias_matrix[:, dst], 1.6**(shortest_paths[:, dst]))
                    # bias_matrix[:, dst] = np.multiply(bias_matrix[:, dst], 26.0)
                bias_matrix = bias_matrix * link_rate_avg
            else:
                raise ValueError("unsupported opt")
            # bias_matrix = shortest_paths
            if opt % 10 == 8:
                mbp = 4.0
            else:
                mbp = 0.0

            bias_vector = bp_env.bias_diff(bias_matrix)
            # routing simulation
            active_links = np.zeros((T,))
            for t in range(T):
                bp_env.pkt_arrival(t)

                if opt % 10 == 3:  # Q learning bias + SP
                    bias_QL, Q_mtx = QLearning(bp_env, 5, 0.05, 0.9)
                    bias_matrix = bias_QL + shortest_paths
                    bias_vector = bp_env.bias_diff(bias_matrix)

                # Commodity and W computation
                if opt < 10:
                    W_amp, W_sign, C_opt = bp_env.commodity_selection(bp_env.queue_matrix, mbp, bias_vector * mpx)
                elif opt < 20:
                    W_amp, W_sign, C_opt = bp_env.commodity_selection(bp_env.queue_matrix, mbp, (1.0/cali_const) * bp_env.pheromones+bias_vector * mpx)
                elif opt < 30:
                    W_amp, W_sign, C_opt = bp_env.commodity_selection(bp_env.queue_matrix_exp, mbp, bias_vector * mpx)
                elif opt < 40:
                    link_bias_vec = bias_vector * (link_rate_avg/np.min(delay_est)) * mpx
                    W_amp, W_sign, C_opt = bp_env.commodity_selection(bp_env.queue_matrix_exp, mbp, link_bias_vec)
                else:
                    # xor_mask = (bp_env.pheromones < 0) ^ (bias_vector < 0)
                    # sign_mask = np.sign(bias_vector)
                    # sign_mask[xor_mask] = -sign_mask[xor_mask]
                    # bias_vector_wts = np.abs(np.multiply(bp_env.pheromones, bias_vector))
                    # bias_vector_wts = np.multiply(bias_vector_wts, sign_mask)
                    # bias_vector_wts = np.multiply(np.abs(bp_env.pheromones), bias_vector)
                    # link_bias_vec = bias_vector * (link_rate_avg / np.mean(delay_est))
                    # link_bias_vec = bias_vector + (1.0/0.03) * bp_env.pheromones
                    # link_bias_vec = np.multiply(bias_vector, np.expand_dims(link_rates, 1))
                    link_bias_vec = bias_vector * (link_rate_avg/np.min(delay_est)) * mpx
                    W_amp, W_sign, C_opt = bp_env.commodity_selection(bp_env.queue_matrix, mbp, link_bias_vec)

                W_amp[C_opt==-1] = 0.0
                bp_env.W[:, t] = W_amp
                bp_env.WSign[:, t] = W_sign
                bp_env.opt_comd_mtx[:, t] = C_opt
                active_links[t] = np.count_nonzero(W_amp)

                # Greedy Maximal Scheduling & Transmission
                mwis = bp_env.scheduling(bp_env.W[:, t] * bp_env.link_rates[:, t])
                bp_env.transmission(t, mwis)
                # ind_vec = np.zeros_like(link_rates)
                # ind_vec[mwis] = 1.0
                # state, link_shares = agent.foo_train(bp_env.adj_i, link_rates, ind_vec, train=True)

                # Collect number of packets in networks for each flow
                for fidx in range(bp_env.num_flows):
                    bp_env.flow_pkts_in_network[fidx, t] = np.sum(bp_env.queue_matrix[:, bp_env.flows[fidx].dest_node])
            # print("Average packets in network:")
            # print(round(bp_env.flow_pkts_in_network.mean(), 2))
            cnt_in, cnt_out, delay_e2e, delay_e2e_raw, delay_est, undeliver = bp_env.collect_delay()
            src_delay_mean = np.nanmean(delay_e2e)
            src_delay_max = np.nanmax(delay_e2e)
            src_delay_std = np.nanstd(delay_e2e)
            est_delay_mean = np.nanmean(delay_est)
            est_delay_max = np.nanmax(delay_est)
            est_delay_std = np.nanstd(delay_est)
            delivery_raw = np.divide(cnt_out.astype(float), cnt_in.astype(float))
            delivery_mean = np.nanmean(delivery_raw)
            delivery_max = np.nanmax(delivery_raw)
            delivery_std = np.nanstd(delivery_raw)
            runtime = time.time() - starttime
            # batch_size = T # len(agent.memory)
            # loss = agent.replay(batch_size)

            print("{}: n {}, f {}, s {}, cf_deg {:.3f}, c {}, ".format(val_mat_names[id], NUM_NODES, num_flows, seed, bp_env.mean_conflict_degree, f_case),
                  "opt {}, runtime {:.2f}, links {:.1f}".format(opt, runtime, np.nanmean(active_links)),
                  "Delay: mean {:.3f}, max {:.3f}, std {:.3f}".format(src_delay_mean, src_delay_max, src_delay_std),
                  "All: mean {:.3f}, max {:.3f}, std {:.3f}".format(est_delay_mean, est_delay_max, est_delay_std),
                  "Delivery: mean {:.3f}, max {:.3f}, std {:.3f}".format(delivery_mean, delivery_max, delivery_std),
                  "cali: {:.3f}".format(np.nanmean(cali_const)),
                  )

            # if not np.isnan(loss):
            #     checkpoint_path = os.path.join(actor_model, 'cp-{epoch:04d}.ckpt'.format(epoch=epoch))
            #     agent.save(checkpoint_path)
            # agent.log_scalar("loss", np.nanmean(losses), step=gidx)
            # agent.log_scalar("delay_hop", src_d_hop_mean, step=gidx)
            # agent.log_scalar("bias_mean", np.mean(bias_matrix[:, bp_env.dst_nodes]), step=gidx)
            # bias_flows = bias_matrix[:, bp_env.dst_nodes]
            # bp_env.plot_routes(bias_flows, opt, with_labels=False)

            gidx += 1
            if opt < 30:
                algo_name = lgds[opt]
            else:
                algo_name = 'exp'

            result = {
                "filename": val_mat_names[id],
                "seed": seed,
                "num_nodes": NUM_NODES,
                "m": m,
                "T": T,
                "cf_radius": cf_radius,
                "cf_degree": bp_env.mean_conflict_degree,
                "opt": opt,
                "Algo": algo_name,
                "f_case": f_case,
                "num_flows": bp_env.num_flows,
                "src_delay_mean": src_delay_mean,
                "src_delay_max": src_delay_max,
                "src_delay_std": src_delay_std,
                "est_delay_mean": est_delay_mean,
                "est_delay_max": est_delay_max,
                "est_delay_std": est_delay_std,
                "src_delay_raw": delay_e2e,
                "delivery_mean": delivery_mean,
                "delivery_max": delivery_max,
                "delivery_std": delivery_std,
                "runtime": runtime,
                "active_links": np.nanmean(active_links),
                "z": np.nanmean(cali_const),
                }
            # df_res = df_res.append(result, ignore_index=True)
            new_row = pd.DataFrame(result)
            df_res = pd.concat([df_res, new_row], ignore_index=True)
            df_res.to_csv(output_csv, index=False)

