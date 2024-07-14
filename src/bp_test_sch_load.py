import argparse

import numpy as np
import scipy.io as sio

from backpressure import *
from gnn_routing_agent_sch import GDPGAgent, FLAGS, flags

flags.DEFINE_string('opts', '5,6,1,7,0', 'Routing scheme.')
flags.DEFINE_string('root', '..', 'Root dir of project.')
flags.DEFINE_float('radius', 1.0, 'Interference radius.')
flags.DEFINE_string('gtype', 'ba', 'Root dir of project.')

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

output_dir = FLAGS.out
output_csv = os.path.join(output_dir, "test_{}_T_{}_ir_{:.1f}_load_opts_{}.csv".format(
    datapath.split("/")[-1], T, cf_radius, opts_txt))
df_res = pd.DataFrame(
    columns=["filename", "seed", "num_nodes", "m", "T", "cf_radius", "cf_degree",
             "opt", "Algo", "num_flows", "arrival_rate",
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
    9: r'SP-$\bar{r}/r$',
}

# Create fig folder if not exist
modeldir = os.path.join(FLAGS.root, "model")
if not os.path.isdir(modeldir):
    os.mkdir(modeldir)

log_dir = os.path.join(FLAGS.root, "logs")
if not os.path.isdir(log_dir):
    os.mkdir(log_dir)

# load models
actor_model = os.path.join(modeldir, 'model_ChebConv_{}_a{}_actor'.format(FLAGS.training_set, FLAGS.num_layer))

try:
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

    link_rates = np.random.uniform(10, 42, size=(bp_env.num_links,))
    link_shares = agent.topology_encode(bp_env.adj_i, link_rates)
    cali_const = 0.1 #1.0/bp_env.mean_conflict_degree
    # link_shares *= cali_const/np.nanmean(link_shares)

    for f_case in range(10):
        # Generate random flows
        np.random.seed(seed*10 + f_case)
        flows_perc = np.random.randint(15, 30)
        num_flows = round(flows_perc / 100 * bp_env.num_nodes)
        nodes = bp_env.graph_c.nodes()
        num_arr = np.random.permutation(nodes)
        arrival_rates = 1.0 * np.random.uniform(0.2, 1.0, (num_flows,))
        link_rates = np.random.uniform(10, 42, size=(bp_env.num_links,))
        bp_env.links_init(link_rates)

        for arrival_rate in np.flip(np.arange(0.05, 1.66, 0.2)):
            arrival_rates = arrival_rate * np.ones((num_flows,))
            bp_env.clear_all_flows()
            flows = []
            for fidx in range(num_flows):
                src = num_arr[2 * fidx]
                dst = num_arr[2 * fidx + 1]
                bp_env.add_flow(src, dst, rate=arrival_rates[fidx])
                flow = {'src': src, 'dst': dst, 'rate': arrival_rates[fidx]}
                flows.append(flow)

            np.random.seed(seed * 10 + f_case)
            bp_env.flows_init()
            # print(filepath)

            # shortest_paths = np.zeros((NUM_NODES, NUM_NODES))
            # bias_matrix = np.zeros_like(bp_env.queue_matrix)
            # cali_const = 0.04

            for opt in opts:
                starttime = time.time()
                bp_env.queues_init()
                bp_env.flows_reset()

                if opt == 5:
                    delay_est = np.divide(26.0, np.multiply(link_rates, link_shares.flatten()))
                    for link, delay in zip(bp_env.link_list, delay_est):
                        src, dst = link
                        bp_env.graph_c[src][dst]["delay"] = delay
                    shortest_paths = all_pairs_shortest_paths(bp_env.graph_c, weight='delay')
                    bias_matrix = shortest_paths
                elif opt == 6:
                    delay_est = np.divide(1.0, link_shares.flatten())
                    for link, delay in zip(bp_env.link_list, delay_est):
                        src, dst = link
                        bp_env.graph_c[src][dst]["delay"] = delay
                    shortest_paths = all_pairs_shortest_paths(bp_env.graph_c, weight='delay')
                    bias_matrix = shortest_paths
                elif opt == 9:
                    delay_est = np.divide(26.0/cali_const, link_rates)
                    for link, delay in zip(bp_env.link_list, delay_est):
                        src, dst = link
                        bp_env.graph_c[src][dst]["delay"] = delay
                    shortest_paths = all_pairs_shortest_paths(bp_env.graph_c, weight='delay')
                    bias_matrix = shortest_paths
                elif opt == 1:
                    shortest_paths = all_pairs_shortest_paths(bp_env.graph_c, weight=None)
                    bias_matrix = shortest_paths
                elif opt == 7:
                    # EDR: Enhanced Dynamic Backpressure Routing
                    shortest_paths = all_pairs_shortest_paths(bp_env.graph_c, weight=None)
                    bias_matrix = (1.0/cali_const) * shortest_paths
                elif opt in [0, 8]:
                    bias_matrix = np.zeros_like(bp_env.queue_matrix)
                elif opt == 4:
                    # VBR (virtual queue based backpressure routing)
                    shortest_paths = all_pairs_shortest_paths(bp_env.graph_c, weight=None)
                    bias_matrix = np.zeros_like(shortest_paths)
                    for fidx in range(bp_env.num_flows):
                        src = bp_env.src_nodes[fidx]
                        dst = bp_env.dst_nodes[fidx]
                        bias_matrix[:, dst] = 1.0 * 1.2**(bp_env.flows[fidx].arrival_rate/(shortest_paths[:, dst]+0.2))
                        bias_matrix[:, dst] = np.multiply(bias_matrix[:, dst], 1.6**(shortest_paths[:, dst]))
                        # bias_matrix[:, dst] = np.multiply(bias_matrix[:, dst], 26.0)
                    bias_matrix = bias_matrix * 26.0
                else:
                    raise ValueError("unsupported opt")
                # bias_matrix = shortest_paths
                if opt == 8:
                    mbp = 4.0
                else:
                    mbp = 0.0
                # routing simulation
                active_links = np.zeros((T,))
                for t in range(T):
                    bp_env.pkt_arrival(t)

                    # Commodity and W computation
                    # if opt in [4, 5, 6]:
                    #     W_amp, W_sign, C_opt = bp_env.commodity_selection(bp_env.queue_matrix + bias_matrix)
                    # else:
                    #     W_amp, W_sign, C_opt = bp_env.commodity_selection_old(bp_env.queue_matrix + bias_matrix, mbp)
                    W_amp, W_sign, C_opt = bp_env.commodity_selection(bp_env.queue_matrix + bias_matrix, mbp)
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

                print("{}: n {}, f {}, s {}, cf {:.1f}, c {}, a {:.2f} ".format(
                    val_mat_names[id], NUM_NODES, num_flows, seed, bp_env.mean_conflict_degree,
                    f_case, arrival_rate),
                      "opt {}, runtime {:.2f}, links {:.1f}".format(opt, runtime, np.nanmean(active_links)),
                      "Delay: mean {:.1f}, max {:.1f}, std {:.1f}".format(src_delay_mean, src_delay_max, src_delay_std),
                      "All: mean {:.1f}, max {:.1f}, std {:.1f}".format(est_delay_mean, est_delay_max, est_delay_std),
                      "Delivery: mean {:.2f}, max {:.2f}, std {:.2f}".format(delivery_mean, delivery_max, delivery_std),
                      "bias: {:.3f}".format(np.nanmean(bias_matrix)),
                      )

                # if not np.isnan(loss):
                #     checkpoint_path = os.path.join(actor_model, 'cp-{epoch:04d}.ckpt'.format(epoch=epoch))
                #     agent.save(checkpoint_path)
                # agent.log_scalar("loss", np.nanmean(losses), step=gidx)
                # agent.log_scalar("delay_hop", src_d_hop_mean, step=gidx)
                # agent.log_scalar("bias_mean", np.mean(bias_matrix[:, bp_env.dst_nodes]), step=gidx)
                # bp_env.plot_routes(delay_n2c, opt)

                gidx += 1

                result = {
                    "filename": val_mat_names[id],
                    "seed": seed,
                    "num_nodes": NUM_NODES,
                    "m": m,
                    "T": T,
                    "cf_radius": cf_radius,
                    "cf_degree": bp_env.mean_conflict_degree,
                    "arrival_rate": arrival_rate,
                    "opt": opt,
                    "Algo": lgds[opt],
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
                     "active_links": np.nanmean(active_links),
                    "runtime": runtime
                    }
                df_res = df_res.append(result, ignore_index=True)
                df_res.to_csv(output_csv, index=False)

    # break