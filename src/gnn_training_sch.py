import argparse

import numpy as np
import scipy.io as sio

from backpressure import *
from gnn_routing_agent_sch import GDPGAgent, FLAGS, flags
# from mlp_routing_agent_sch import GDPGAgent, FLAGS, flags

flags.DEFINE_integer('opt', '5', 'Routing scheme.')

agent = GDPGAgent(FLAGS, 5000)

std = 0.01
batch_size = FLAGS.batch
T = FLAGS.T
datapath = FLAGS.datapath
val_mat_names = sorted(os.listdir(datapath))
output_dir = FLAGS.out
output_csv = os.path.join(output_dir, "train_trace_{}_opt_{}_T_{}.csv".format(datapath.split("/")[-1], FLAGS.opt, T))
df_res = pd.DataFrame(
    columns=["filename", "seed", "num_nodes", "m", "T", "num_flows",
             "src_delay_raw",
             "src_delay_mean", "src_delay_max", "src_delay_std",
             "est_delay_mean", "est_delay_max", "est_delay_std",
             "delivery_mean", "delivery_max", "delivery_std",
             "runtime"]
)

# Create fig folder if not exist
modeldir = os.path.join("..", "model")
if not os.path.isdir(modeldir):
    os.mkdir(modeldir)

log_dir = os.path.join("..", "logs")
if not os.path.isdir(log_dir):
    os.mkdir(log_dir)

# load models
if FLAGS.opt == 5:
    actor_model = os.path.join(modeldir, 'model_{}_{}_a{}_o{}_actor'.format(agent.name, FLAGS.training_set, FLAGS.num_layer, FLAGS.opt))
else:
    actor_model = os.path.join(modeldir, 'model_{}_{}_a{}_actor'.format(agent.name, FLAGS.training_set, FLAGS.num_layer))

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

for epoch in range(FLAGS.epochs):
    for id in np.random.permutation(len(val_mat_names)):
        filepath = os.path.join(datapath, val_mat_names[id])
        mat_contents = sio.loadmat(filepath)
        net_cfg = mat_contents['network'][0,0]
        # link_rates = mat_contents["link_rate"][0]
        pos_c = mat_contents["pos_c"]

        seed = int(net_cfg['seed'].flatten()[0])
        NUM_NODES = int(net_cfg['num_nodes'].flatten()[0])
        m = net_cfg['m'].flatten()[0]
        radius = np.round(np.random.uniform(0.0, 0.8), 1)

        starttime = time.time()
        # Configuration
        bp_env = Backpressure(NUM_NODES, T, seed, m, pos_c, cf_radius=radius, trace=True)
        link_rates = np.random.uniform(10, 42, size=(bp_env.num_links,))
        bp_env.links_init(link_rates)
        bp_env.queues_init()

        # flows_cfg = mat_contents["flows"][0]
        # num_flows = len(flows_cfg)
        # for fidx in range(num_flows):
        #     flow = flows_cfg[fidx]
        #     src = flow['src'][0,0].flatten()[0]
        #     dst = flow['dst'][0,0].flatten()[0]
        #     rate = flow['rate'][0,0].flatten()[0]
        #     bp_env.add_flow(src, dst, rate)

        # Generate random flows
        flows_perc = np.random.randint(15, 30)
        num_flows = round(flows_perc / 100 * bp_env.num_nodes)
        nodes = bp_env.graph_c.nodes()
        num_arr = np.random.permutation(nodes)
        arrival_rates = np.random.uniform(0.2, 1.0, (num_flows,))

        flows = []
        for fidx in range(num_flows):
            src = num_arr[2 * fidx]
            dst = num_arr[2 * fidx + 1]
            bp_env.add_flow(src, dst, rate=arrival_rates[fidx])
            flow = {'src': src, 'dst': dst, 'rate': arrival_rates[fidx]}
            flows.append(flow)

        bp_env.flows_init()
        bp_env.pheromone_init()
        # print(filepath)

        # shortest_paths = np.zeros((NUM_NODES, NUM_NODES))
        # bias_matrix = np.zeros_like(bp_env.queue_matrix)

        # generating bias per graph
        if FLAGS.opt == 5:
            link_shares = agent.topology_encode(bp_env.adj_i, link_rates)
            delay_link = np.divide(26.0, np.multiply(link_rates, link_shares.flatten()))
        else:
            link_shares = agent.topology_encode(bp_env.adj_i, np.ones_like(link_rates))
            delay_link = np.divide(1.0, link_shares.flatten())

        for link, delay in zip(bp_env.link_list, delay_link):
            src, dst = link
            bp_env.graph_c[src][dst]["delay"] = delay
        shortest_paths = all_pairs_shortest_paths(bp_env.graph_c, weight='delay')
        bias_matrix = shortest_paths
        # routing simulation
        for t in range(T):
            bp_env.pkt_arrival(t)
            # Commodity and W computation
            W_amp, W_sign, C_opt = bp_env.commodity_selection(bp_env.queue_matrix + bias_matrix)
            # W_amp, W_sign, C_opt = bp_env.commodity_selection(bp_env.queue_matrix)
            W_amp[C_opt == -1] = 0.0
            bp_env.W[:, t] = W_amp
            bp_env.WSign[:, t] = W_sign
            bp_env.opt_comd_mtx[:, t] = C_opt

            # Greedy Maximal Scheduling & Transmission
            mwis = bp_env.scheduling(bp_env.W[:, t] * bp_env.link_rates[:, t])
            bp_env.transmission(t, mwis)
            ind_vec = np.zeros_like(link_rates)
            ind_vec[mwis] = 1.0
            if FLAGS.opt == 5:
                state, link_shares = agent.foo_train(bp_env.adj_i, link_rates, ind_vec, train=True)
            else:
                state, link_shares = agent.foo_train(bp_env.adj_i, np.ones_like(link_rates), ind_vec, train=True)

            # Collect number of packets in networks for each flow
            for fidx in range(bp_env.num_flows):
                bp_env.flow_pkts_in_network[fidx, t] = np.sum(bp_env.queue_matrix[:, bp_env.flows[fidx].dest_node])
        # print("Average packets in network:")
        # print(round(bp_env.flow_pkts_in_network.mean(), 2))
        losses = []
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
        batch_size = 1 # len(agent.memory)
        loss = agent.replay(batch_size)

        print("{}: n {}, f {}, s {}, cf_deg {:.3f}, ri {:.2f} ".format(val_mat_names[id], NUM_NODES, num_flows, seed,
                                                                   bp_env.mean_conflict_degree, radius),
              "Loss {:.3f}, runtime {:.3f}".format(loss, runtime),
              "Delay: mean {:.3f}, max {:.3f}, std {:.3f}".format(src_delay_mean, src_delay_max, src_delay_std),
              "All: mean {:.3f}, max {:.3f}, std {:.3f}".format(est_delay_mean, est_delay_max, est_delay_std),
              "Delivery: mean {:.3f}, max {:.3f}, std {:.3f}".format(delivery_mean, delivery_max, delivery_std),
              "z: {:.6f}, ind: {:.3f}".format(np.nanmean(link_shares), np.nanmean(ind_vec)),
              )

        if not np.isnan(loss):
            checkpoint_path = os.path.join(actor_model, 'cp-{epoch:04d}.ckpt'.format(epoch=epoch))
            agent.save(checkpoint_path)
        # agent.log_scalar("loss", np.nanmean(losses), step=gidx)
        # agent.log_scalar("delay_hop", src_d_hop_mean, step=gidx)
        # agent.log_scalar("bias_mean", np.mean(bias_matrix[:, bp_env.dst_nodes]), step=gidx)
        gidx += 1

        result = {
            "epoch": epoch,
            "filename": val_mat_names[id],
            "seed": seed,
            "num_nodes": NUM_NODES,
            "m": m,
            "T": T,
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
            "runtime": runtime
        }
        df_res = df_res.append(result, ignore_index=True)
    df_res.to_csv(output_csv, index=False)

