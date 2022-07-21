HYPERPARAM_SEARCH = False
HYPERPARAM_SEARCH_N_TRIALS = None   # how many grid search trials to run
                                    #    (set to None for exhaustive search)

import argparse
from itertools import permutations
import pickle
from queue import PriorityQueue
import os
import random
import time

import networkx as nx
import numpy as np
from sklearn.manifold import TSNE
import torch
import torch.nn as nn
import torch.multiprocessing as mp
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import roc_auc_score
import pickle

from common import data
from common import models
from common import utils

if HYPERPARAM_SEARCH:
    from test_tube import HyperOptArgumentParser
    from hyp_search import parse_encoder
else:
    from config import parse_encoder
from test import validation

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def build_model(args):
    # build model
    if args.method_type == "order":
        model = models.OrderEmbedder(2, args.hidden_dim, args)
    elif args.method_type == "mlp":
        model = models.BaselineMLP(2, args.hidden_dim, args)
    model.to(utils.get_device())
    if args.test and args.model_path:
        model.load_state_dict(torch.load(args.model_path,
            map_location=utils.get_device()))
    return model

def make_data_source(args):
    toks = args.dataset.split("-")
    if toks[0] == "syn":
        if len(toks) == 1 or toks[1] == "balanced":
            data_source = data.OTFSynDataSource(
                node_anchored=args.node_anchored)
        elif toks[1] == "imbalanced":
            data_source = data.OTFSynImbalancedDataSource(
                node_anchored=args.node_anchored)
        else:
            raise Exception("Error: unrecognized dataset")
    else:
        if len(toks) == 1 or toks[1] == "balanced":
            data_source = data.DiskDataSource(toks[0],
                node_anchored=args.node_anchored)
        elif toks[1] == "imbalanced":
            data_source = data.DiskImbalancedDataSource(toks[0],
                node_anchored=args.node_anchored)
        else:
            raise Exception("Error: unrecognized dataset")
    return data_source

def train(args, model, logger, in_queue, out_queue):
    """Train the order embedding model.

    args: Commandline arguments
    logger: logger for logging progress
    in_queue: input queue to an intersection computation worker
    out_queue: output queue to an intersection computation worker
    """
    # print("Running train")
    scheduler, opt = utils.build_optimizer(args, model.parameters())
    if args.method_type == "order":
        clf_opt = optim.Adam(model.clf_model.parameters(), lr=args.lr)
        # clf_opt_nodes = optim.Adam(model.clf_model_nodes.parameters(), lr=args.lr)
        mlp_model_nodes_opt = optim.Adam(model.mlp_model_nodes.parameters(), lr=args.lr)

    done = False
    k = 0
    print(k)
    while not done:
        data_source = make_data_source(args)
        loaders = data_source.gen_data_loaders(args.eval_interval *
            args.batch_size, args.batch_size, train=True)
        for batch_target, batch_neg_target, batch_neg_query in zip(*loaders):
            msg, _ = in_queue.get()
            if msg == "done":
                done = True
                break
            # train
            model.train()
            model.zero_grad()
            pos_a, pos_b, neg_a, neg_b = data_source.gen_batch(batch_target,
                batch_neg_target, batch_neg_query, True)
            # emb_pos_a, emb_pos_b = model.emb_model(pos_a), model.emb_model(pos_b)
            # emb_neg_a, emb_neg_b = model.emb_model(neg_a), model.emb_model(neg_b)

            # Added by TC
            emb_pos_a, emb_pos_a_nodes = model.emb_model(pos_a)
            emb_pos_b, emb_pos_b_nodes = model.emb_model(pos_b)
            emb_neg_a, emb_neg_a_nodes = model.emb_model(neg_a)
            emb_neg_b, emb_neg_b_nodes = model.emb_model(neg_b)
            # print(emb_pos_a.shape, emb_neg_a.shape, emb_neg_b.shape)
            emb_as = torch.cat((emb_pos_a, emb_neg_a), dim=0)
            emb_bs = torch.cat((emb_pos_b, emb_neg_b), dim=0)
            labels = torch.tensor([1]*pos_a.num_graphs + [0]*neg_a.num_graphs).to(
                utils.get_device())

            # Added by TC
            # Alignment matrrix label
            align_mat_batch = []
            for sample_idx in range(len(pos_b.node_label)):
                align_mat = torch.zeros(len(pos_a.node_label[sample_idx]), len(pos_b.node_label[sample_idx]))
                for i, a_n in enumerate(pos_a.node_label[sample_idx]):
                    if a_n in pos_b.alignment[sample_idx]:
                        align_mat[i][pos_b.alignment[sample_idx].index(a_n)] = 1
                align_mat_batch.append(align_mat)

            align_mat_batch_all = []
            for i, align_mat in enumerate(align_mat_batch):
                align_mat_batch_all.append(align_mat.flatten())
            labels_nodes = torch.cat(align_mat_batch_all, dim=-1)

            intersect_embs = None
            # pred = model(emb_as, emb_bs)
            # loss = model.criterion(pred, intersect_embs, labels)
            pred = model(emb_as, emb_bs, emb_pos_a_nodes, emb_pos_b_nodes)
            loss = model.criterion(pred, intersect_embs, labels, align_mat_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            if scheduler:
                scheduler.step()

            if args.method_type == "order":
                with torch.no_grad():
                    pred, pred_nodes = model.predict(pred)
                model.clf_model.zero_grad()
                pred = model.clf_model(pred.unsqueeze(1))
                criterion = nn.NLLLoss()
                clf_loss = criterion(pred, labels)
                clf_loss.backward()
                clf_opt.step()

                # model.clf_model_nodes.zero_grad()
                # pred_nodes = model.clf_model(pred_nodes.unsqueeze(1))

#                 model.mlp_model_nodes.zero_grad()
#                 pred_scores = model.mlp_model_nodes(emb_pos_a_nodes, emb_pos_b_nodes, align_mat_batch)
#                 # pred_scores = torch.cat(pred_scores_batch_all, dim=-1)
#                 node_alignment_loss = F.binary_cross_entropy_with_logits(pred_scores, labels_nodes)
#                 node_alignment_loss.backward()
#                 mlp_model_nodes_opt.step()
                # criterion_nodes = nn.BCEWithLogitsLoss()
                # clf_loss_nodes = criterion_nodes(pred_scores, labels_nodes)


                # pred_scores_batch_all = []
                # for i in range(len(emb_pos_a_nodes)):
                #     align_mat_score = model.mlp_model_nodes(emb_pos_a_nodes[i], emb_pos_b_nodes[i])
                #     pred_scores_batch_all.append(align_mat_score.flatten())
                # pred_scores = torch.cat(pred_scores_batch_all, dim=-1)



                # clf_loss_nodes.backward()
                # mlp_model_nodes_opt.step()

                # print("Node Alignment Loss {}; AUC {}".format(node_alignment_loss, auc))
                # criterion_nodes = nn.NLLLoss()
                # clf_loss_nodes = criterion_nodes(pred_nodes, labels_nodes.long())
                # clf_loss_nodes.backward()
                # clf_opt_nodes.step()




            pred = pred.argmax(dim=-1)
            # pred_nodes = pred_nodes.argmax(dim=-1)
            acc = torch.mean((pred == labels).type(torch.float))
            # acc_nodes = torch.mean((pred_nodes == labels_nodes).type(torch.float))
            acc_nodes = roc_auc_score(labels_nodes, pred_scores.detach().numpy())
            train_loss = loss.item()
            train_acc = (acc.item() + acc_nodes.item())/2

            # out_queue.put(("step", (loss.item(), acc, acc_nodes)))
            print('Epoch: {}, loss: {}, Accuracy: {}'.format(k+1, loss.item(), acc))
            k += 1

def train_loop(args):

    if not os.path.exists(os.path.dirname(args.model_path)):
        os.makedirs(os.path.dirname(args.model_path))
    if not os.path.exists("plots/"):
        os.makedirs("plots/")

    print("Starting {} workers".format(args.n_workers))
    in_queue, out_queue = mp.Queue(), mp.Queue()

    print("Using dataset {}".format(args.dataset))

    record_keys = ["conv_type", "n_layers", "hidden_dim",
        "margin", "dataset", "max_graph_size", "skip"]
    args_str = ".".join(["{}={}".format(k, v)
        for k, v in sorted(vars(args).items()) if k in record_keys])
    # logger = SummaryWriter(comment=args_str)
    logger = SummaryWriter()


    model = build_model(args).to(device)
    model.share_memory()

    if args.method_type == "order":
        clf_opt = optim.Adam(model.clf_model.parameters(), lr=args.lr)
    else:
        clf_opt = None

    data_source = make_data_source(args)
    # print(type(data_source))
    # loaders = data_source.gen_data_loaders(args.val_size, args.batch_size,
    #     train=False, use_distributed_sampling=False)
    if args.test:
        print('Test Data')
        loaders = data_source.gen_data_loaders(args.val_size, args.batch_size,
                                               train=False, use_distributed_sampling=False)
    else:
        print('Train Data')
        loaders = data_source.gen_data_loaders(args.val_size, args.batch_size,
                                               train=True, use_distributed_sampling=False)
    test_pts = []
    # count = 0
    for batch_target, batch_neg_target, batch_neg_query in zip(*loaders):
        # print(count)
        # count += 1

        # print(batch_target, batch_neg_target, batch_neg_query)

        if args.test:
            pos_a, pos_b, neg_a, neg_b = data_source.gen_batch(batch_target,
                batch_neg_target, batch_neg_query, train = False)
        else:
            pos_a, pos_b, neg_a, neg_b = data_source.gen_batch(batch_target,
                batch_neg_target, batch_neg_query, train = True)
        # print(type(pos_a))
        if pos_a:
            pos_a = pos_a.to(device)
            pos_b = pos_b.to(device)
        neg_a = neg_a.to(device)
        neg_b = neg_b.to(device)
        test_pts.append((pos_a, pos_b, neg_a, neg_b))
        # print(len(test_pts[len(test_pts)-1][0])) # 12
        # file = open('test_pts.obj','wb')
        # pickle.dump(test_pts,file)
        # print('done!')
    print('Data Load Finished!')
    
    train(args, model, logger, in_queue, out_queue)

#     workers = []
#     for i in range(args.n_workers):
#         worker = mp.Process(target=train, args=(args, model, data_source,
#             in_queue, out_queue))
#         worker.start()
#         workers.append(worker)

    if args.test:
        # pos_a, pos_b, neg_a, neg_b = data_source.gen_batch(batch_target,
        #     batch_neg_target, batch_neg_query, train = False)
        # if pos_a:
        #     pos_a = pos_a.to(torch.device("cpu"))
        #     pos_b = pos_b.to(torch.device("cpu"))
        # neg_a = neg_a.to(torch.device("cpu"))
        # neg_b = neg_b.to(torch.device("cpu"))
        # test_pts.append((pos_a, pos_b, neg_a, neg_b))
        validation(args, model, test_pts, logger, 0, 0, verbose=True)
    else:
        batch_n = 0
        # for epoch in range(args.n_batches // args.eval_interval):
        for epoch in range(54):
            for i in range(args.eval_interval):
                in_queue.put(("step", None))
            for i in range(args.eval_interval):
                # print(args.eval_interval)
                msg, params = out_queue.get()
                train_loss, train_acc, train_acc_nodes = params
                print("Batch {}. Loss: {:.4f}. Subgraph acc: {:.4f} Node Alignment acc: {:.4f}".format(
                    batch_n, train_loss, train_acc, train_acc_nodes), end="               \r")
                logger.add_scalar("Loss/train", train_loss, batch_n)
                logger.add_scalar("Accuracy/train", train_acc, batch_n)
                batch_n += 1
            validation(args, model, test_pts, logger, batch_n, epoch)

    for i in range(args.n_workers):
        in_queue.put(("done", None))
    for worker in workers:
        worker.join()

def main(force_test=False):
    mp.set_start_method("spawn", force=True)
    parser = (argparse.ArgumentParser(description='Order embedding arguments')
        if not HYPERPARAM_SEARCH else
        HyperOptArgumentParser(strategy='grid_search'))

    utils.parse_optimizer(parser)
    parse_encoder(parser)
    args = parser.parse_args([])
    # train(args)

    if force_test:
        args.test = True

    # Currently due to parallelism in multi-gpu training, this code performs
    # sequential hyperparameter tuning.
    # All gpus are used for every run of training in hyperparameter search.
    if HYPERPARAM_SEARCH:
        for i, hparam_trial in enumerate(args.trials(HYPERPARAM_SEARCH_N_TRIALS)):
            print("Running hyperparameter search trial", i)
            print(hparam_trial)
            train_loop(hparam_trial)
    else:
        train_loop(args)

mp.set_start_method("spawn", force=True)
parser = (argparse.ArgumentParser(description='Order embedding arguments')
    if not HYPERPARAM_SEARCH else
    HyperOptArgumentParser(strategy='grid_search'))

utils.parse_optimizer(parser)
parse_encoder(parser)
args = parser.parse_args([])

if not os.path.exists(os.path.dirname(args.model_path)):
    os.makedirs(os.path.dirname(args.model_path))
if not os.path.exists("plots/"):
    os.makedirs("plots/")

print("Starting {} workers".format(args.n_workers))
in_queue, out_queue = mp.Queue(), mp.Queue()

print("Using dataset {}".format(args.dataset))

record_keys = ["conv_type", "n_layers", "hidden_dim",
    "margin", "dataset", "max_graph_size", "skip"]
args_str = ".".join(["{}={}".format(k, v)
    for k, v in sorted(vars(args).items()) if k in record_keys])
# logger = SummaryWriter(comment=args_str)
logger = SummaryWriter()


model = build_model(args).to(device)
model.share_memory()

if args.method_type == "order":
    clf_opt = optim.Adam(model.clf_model.parameters(), lr=args.lr)
else:
    clf_opt = None

data_source = make_data_source(args)
print(data_source)
# print(type(data_source))
# loaders = data_source.gen_data_loaders(args.val_size, args.batch_size,
#     train=False, use_distributed_sampling=False)
if args.test:
    print('Test Data')
    loaders = data_source.gen_data_loaders(args.val_size, args.batch_size,
                                           train=False, use_distributed_sampling=False)
else:
    print('Train Data')
    loaders = data_source.gen_data_loaders(args.val_size, args.batch_size,
                                           train=True, use_distributed_sampling=False)
test_pts = []
# count = 0
for batch_target, batch_neg_target, batch_neg_query in zip(*loaders):
    # print(count)
    # count += 1

    # print(batch_target, batch_neg_target, batch_neg_query)

    if args.test:
        pos_a, pos_b, neg_a, neg_b = data_source.gen_batch(batch_target,
            batch_neg_target, batch_neg_query, train = False)
    else:
        pos_a, pos_b, neg_a, neg_b = data_source.gen_batch(batch_target,
            batch_neg_target, batch_neg_query, train = True)
    # print(type(pos_a))
    if pos_a:
        pos_a = pos_a.to(device)
        pos_b = pos_b.to(device)
    neg_a = neg_a.to(device)
    neg_b = neg_b.to(device)
    test_pts.append((pos_a, pos_b, neg_a, neg_b))
    # print(len(test_pts[len(test_pts)-1][0])) # 12
    # file = open('test_pts.obj','wb')
    # pickle.dump(test_pts,file)
    # print('done!')
print('Data Load Finished!')


model = build_model(args).to(device)
model.share_memory()

scheduler, opt = utils.build_optimizer(args, model.parameters())
if args.method_type == "order":
    clf_opt = optim.Adam(model.clf_model.parameters(), lr=1e-3)
    # clf_opt_nodes = optim.Adam(model.clf_model_nodes.parameters(), lr=args.lr)
    mlp_model_nodes_opt = optim.Adam(model.mlp_model_nodes.parameters(), lr=args.lr)
print('Setting Finished!')

done = False
k = 0
for k in range(30):
    data_source = make_data_source(args)
    loaders = data_source.gen_data_loaders(args.eval_interval *
        args.batch_size, args.batch_size, train=True)
    step = 0
    for batch_target, batch_neg_target, batch_neg_query in zip(*loaders):
#             msg, _ = in_queue.get()
#             if msg == "done":
#                 done = True
#                 break
        # train
        model.train()
        model.zero_grad()
        pos_a, pos_b, neg_a, neg_b = data_source.gen_batch(batch_target,
            batch_neg_target, batch_neg_query, True)
        # emb_pos_a, emb_pos_b = model.emb_model(pos_a), model.emb_model(pos_b)
        # emb_neg_a, emb_neg_b = model.emb_model(neg_a), model.emb_model(neg_b)

        # Added by TC
        emb_pos_a, emb_pos_a_nodes = model.emb_model(pos_a.to(device))
        emb_pos_b, emb_pos_b_nodes = model.emb_model(pos_b.to(device))
        emb_neg_a, emb_neg_a_nodes = model.emb_model(neg_a.to(device))
        emb_neg_b, emb_neg_b_nodes = model.emb_model(neg_b.to(device))
        # print(emb_pos_a.shape, emb_neg_a.shape, emb_neg_b.shape)
        emb_as = torch.cat((emb_pos_a, emb_neg_a), dim=0)
        emb_bs = torch.cat((emb_pos_b, emb_neg_b), dim=0)
        labels = torch.tensor([1]*pos_a.num_graphs + [0]*neg_a.num_graphs).to(
            utils.get_device())

        # Added by TC
        # Alignment matrrix label
        align_mat_batch = []
        for sample_idx in range(len(pos_b.node_label)):
            align_mat = torch.zeros(len(pos_a.node_label[sample_idx]), len(pos_b.node_label[sample_idx]))
            for i, a_n in enumerate(pos_a.node_label[sample_idx]):
                if a_n in pos_b.alignment[sample_idx]:
                    align_mat[i][pos_b.alignment[sample_idx].index(a_n)] = 1
            align_mat_batch.append(align_mat)

        align_mat_batch_all = []
        for i, align_mat in enumerate(align_mat_batch):
            align_mat_batch_all.append(align_mat.flatten())
        labels_nodes = torch.cat(align_mat_batch_all, dim=-1)

        intersect_embs = None
        # pred = model(emb_as, emb_bs)
        # loss = model.criterion(pred, intersect_embs, labels)
        pred = model(emb_as, emb_bs, emb_pos_a_nodes, emb_pos_b_nodes)
        loss = model.criterion(pred, intersect_embs, labels, align_mat_batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if scheduler:
            scheduler.step()

        if args.method_type == "order":
            if loss.item() > 10:
                acc = 0
            else:
                with torch.no_grad():
                    pred, pred_nodes = model.predict(pred)
                model.clf_model.zero_grad()
                pred = model.clf_model(pred.unsqueeze(1))
                criterion = nn.NLLLoss()
                clf_loss = criterion(pred, labels)
                clf_loss.backward()
                clf_opt.step()
                pred = pred.argmax(dim=-1)
                acc = torch.mean((pred == labels).type(torch.float))
    
            # model.clf_model_nodes.zero_grad()
            # pred_nodes = model.clf_model(pred_nodes.unsqueeze(1))

#             model.mlp_model_nodes.zero_grad()
#             pred_scores = model.mlp_model_nodes(emb_pos_a_nodes, emb_pos_b_nodes, align_mat_batch)
#             # pred_scores = torch.cat(pred_scores_batch_all, dim=-1)
#             node_alignment_loss = F.binary_cross_entropy_with_logits(pred_scores, labels_nodes)
#             node_alignment_loss.backward()
#             mlp_model_nodes_opt.step()
            # criterion_nodes = nn.BCEWithLogitsLoss()
            # clf_loss_nodes = criterion_nodes(pred_scores, labels_nodes)


            # pred_scores_batch_all = []
            # for i in range(len(emb_pos_a_nodes)):
            #     align_mat_score = model.mlp_model_nodes(emb_pos_a_nodes[i], emb_pos_b_nodes[i])
            #     pred_scores_batch_all.append(align_mat_score.flatten())
            # pred_scores = torch.cat(pred_scores_batch_all, dim=-1)



            # clf_loss_nodes.backward()
            # mlp_model_nodes_opt.step()

            # print("Node Alignment Loss {}; AUC {}".format(node_alignment_loss, auc))
            # criterion_nodes = nn.NLLLoss()
            # clf_loss_nodes = criterion_nodes(pred_nodes, labels_nodes.long())
            # clf_loss_nodes.backward()
            # clf_opt_nodes.step()
     # pred_nodes = pred_nodes.argmax(dim=-1)
#         acc = torch.mean((pred == labels).type(torch.float))
        # acc_nodes = torch.mean((pred_nodes == labels_nodes).type(torch.float))
        # acc_nodes = roc_auc_score(labels_nodes, pred_scores.detach().numpy())
        train_loss = loss.item()
        # train_acc = (acc.item() + acc_nodes.item())/2

        # out_queue.put(("step", (loss.item(), acc, acc_nodes)))
        print('Epoch: {}, loss: {}, Accuracy: {}'.format(k, loss.item(), acc))
    k += 1

model.eval()


def admm_opt(emb_a, emb_b, adj_pair, true_matrix, initial_align, epochs=50, p=1):
    # Initialize X0 (q, t), Y0 (q, t), Z0 (q, t), P0 (q, t) 
#     initial_X = true_matrix.T
#     initial_X = torch.zeros(true_matrix.shape).T
#     initial_X = torch.empty((true_matrix.shape)).T
#     torch.nn.init.orthogonal_(initial_X)
#     align_ind = torch.argmax(initial_X, dim=1)
#     for i in range(initial_X.shape[0]):
#         initial_X[i, align_ind[i]] = 1

    initial_X = initial_align.detach().cpu().T
    
    initial_Y = torch.mm(initial_X, adj_pair[0].float())
    
    H = torch.zeros((initial_X.shape[0], initial_X.shape[1]))
#     for i in range(initial_X.shape[0]):
#         for j in range(initial_X.shape[1]):
#             H[i, j] = torch.max(emb_b[i]-emb_a[j])
#     H = torch.max(torch.zeros(H.shape), torch.min(torch.ones(H.shape), H))
    initial_Z = initial_X*H
    
    initial_P = initial_X.clone()
    
    # Initialize U1, U2, U3
    initial_U1, initial_U2, initial_U3 = torch.zeros(initial_X.shape), torch.zeros(initial_X.shape), torch.zeros(initial_X.shape)
    
    # ADMM Algorithm
    for epoch in range(epochs):
        # Update P
        u, d, v = torch.linalg.svd(initial_X - initial_U3/p)
        initial_P = torch.mm(torch.mm(u, torch.eye(initial_X.shape[0], initial_X.shape[1])), v)
        
        
        # Update X
        X_1 = torch.mm(adj_pair[1].float().T, initial_Y) + p*torch.mm(initial_Y, adj_pair[0].float().T)\
        + torch.mm(initial_U1, adj_pair[0].float().T) + (p*initial_Z+initial_U2)*H\
        + p*initial_P + initial_U3
        
        X_2 = torch.mm(initial_Y.T, initial_Y) + torch.mm(adj_pair[0].float(), adj_pair[0].float().T)\
        + p*torch.mm(H.T, H) + p*torch.eye(initial_X.shape[1])
        X_2_inv = torch.linalg.inv(X_2)
                
        initial_X = torch.mm(X_1, X_2_inv)
        

        # Update Y
        initial_Y1 = initial_Y.clone()
        Y_1 = torch.mm(adj_pair[1].float(), initial_X) + p*torch.mm(initial_X, adj_pair[0].float())\
        - initial_U1
        
        Y_2 = 2*torch.mm(initial_X.T, initial_X) + p*torch.eye(initial_X.shape[1])
        Y_2_inv = torch.linalg.pinv(Y_2)
        
        initial_Y = torch.mm(Y_1, Y_2_inv)
        
        
        # Update Z
        Z_1 = (p*initial_X*H - initial_U2) / (p+2)
        
        Z_2 = torch.min(torch.zeros(initial_U2.shape), initial_X*H-initial_U2/p)
        
        initial_Z = torch.max(Z_1, Z_2)
        
                
        # Update R1, R2, R3
        R1 = initial_Y - torch.mm(initial_X, adj_pair[0].float())
        R2 = initial_Z - initial_X*H
        R3 = initial_P - initial_X
        
        # Update U1, U2, U3
        initial_U1 += p*R1
        initial_U2 += p*R2
        initial_U3 += p*R3
        
                                
        align_matrix = ortho_align(initial_X)
        auc = roc_auc_score(true_matrix.T.flatten(), initial_X.flatten())
        print('Epoch: {}, AUC: {}'.format(epoch+1, auc))
    return initial_P, auc

def depth_count(adj_t):
    G_t = nx.from_numpy_matrix(adj_t, create_using=nx.DiGraph)
    in_dict = dict(G_t.in_degree)
    source_ind = [ind for ind in in_dict.keys() if in_dict[ind] == 0]
    avg_depth = np.zeros(len(G_t.nodes))
    for ind in source_ind:
        single_dict = nx.shortest_path_length(G_t,ind)
        single_depth = np.zeros(len(G_t.nodes))
        for key, value in single_dict.items():
            single_depth[key] = value
        for i, value in enumerate(avg_depth):
            if value < single_depth[i]:
                avg_depth[i] = single_depth[i]
#         avg_depth += single_depth
#     avg_depth /= len(source_ind)
    return avg_depth

def ortho_align(align_mat):
    align_matrix = torch.zeros(align_mat.shape)
    align_sort = torch.sort(align_mat.flatten(), descending=True).values
    align_sort1 = torch.zeros(align_mat.shape)
    for value in align_sort:
        align_sort1[align_mat==value] = torch.where(align_sort==value)[0].float()
    align_row = []
    align_column = []
    for i in range(align_mat.flatten().shape[0]):
        if torch.sum(align_matrix) < align_matrix.shape[0]:
            ind = torch.where(align_sort1==i)
            if ind[0] not in align_row and ind[1] not in align_column:
                align_matrix[ind[0], ind[1]] = 1
                align_row.append(ind[0])
                align_column.append(ind[1])
    return align_matrix


def alignment(emb_a, emb_b, adj_pair, true_matrix, epoch=200):
    tr = 50
    initial_align = torch.autograd.Variable(-1e-3 * torch.ones(true_matrix.shape).float()).to(emb_a.device)
    initial_align = torch.autograd.Variable(initial_align).to(emb_a.device)
    hardsigmoid = nn.Hardsigmoid()

    avg_t = torch.Tensor(depth_count(np.array(adj_pair[0]))).to(emb_a.device)


    initial_align.requires_grad = True
    align_opt = optim.Adam([initial_align], lr=5e-3)
    for i in range(epoch):
        align_opt.zero_grad()


        align_loss_1 = torch.sum((torch.mm(torch.mm(torch.sigmoid(tr * initial_align)), adj_pair[0].float().to(emb_a.device)), torch.sigmoid(tr * initial_align))) - adj_pair[1].float().to(emb_a.device)**2)
        
        align_loss_2 = 0
        for j in range(initial_align.shape[0]):
            for k in range(initial_align.shape[1]):
                align_loss_2 += torch.sum((torch.max(torch.zeros(emb_a[j].shape).to(emb_a.device), initial_align[j][k] * (emb_b[k]-emb_a[j])))**2)
        
        align_loss_3 = torch.sum((torch.mm(torch.sigmoid(tr * initial_align.T), torch.sigmoid(tr * initial_align)) - torch.eye(initial_align.shape[1]).to(emb_a.device))**2)

        align_loss_4 = 0
        for j in range(initial_align.shape[0]):
            for k in range(initial_align.shape[1]):
                align_loss_4 += (torch.sigmoid(tr * initial_align[j][k]) * avg_t[j])**2
                

        align_loss = align_loss_1 + align_loss_2 + 1e-5 * align_loss_3 + 1e-3 * align_loss_4
        align_loss.backward()
        align_opt.step()    
        
        align_matrix = ortho_align(initial_align.detach().cpu())
        auc = roc_auc_score(true_matrix.flatten(), align_matrix.flatten())
        print('Epoch: {}, Loss: {}, AUC: {}'.format(i, align_loss.item(), auc))
    return align_matrix

USE_ORCA_FEATS = False

model.eval()
all_raw_preds, all_preds, all_labels = [], [], []
all_raw_preds_nodes, all_preds_nodes, all_labels_nodes = [], [], []
all_test_pairs = []
for pos_a, pos_b, neg_a, neg_b in test_pts:
    if pos_a:
        pos_a = pos_a.to(utils.get_device())
        pos_b = pos_b.to(utils.get_device())
    neg_a = neg_a.to(utils.get_device())
    neg_b = neg_b.to(utils.get_device())
    labels = torch.tensor([1]*(pos_a.num_graphs if pos_a else 0) +
        [0]*neg_a.num_graphs).to(utils.get_device())
    # Alignment matrrix label
    align_mat_batch = []
    adj_pair = []
    for sample_idx in range(len(pos_b.node_label)):
        align_mat = torch.zeros(len(pos_a.node_label[sample_idx]), len(pos_b.node_label[sample_idx]))
        adj_pair.append([torch.tensor(nx.adjacency_matrix(pos_a.G[sample_idx]).todense()), torch.tensor(nx.adjacency_matrix(pos_b.G[sample_idx]).todense())])
        for i, a_n in enumerate(pos_a.node_label[sample_idx]):
            if a_n in pos_b.alignment[sample_idx]:
                align_mat[i][pos_b.alignment[sample_idx].index(a_n)] = 1
        align_mat_batch.append(align_mat)
        
    align_mat_batch_all = []
    for i, align_mat in enumerate(align_mat_batch):
        align_mat_batch_all.append(align_mat.flatten())
    labels_nodes = torch.cat(align_mat_batch_all, dim=-1)

    with torch.no_grad():
        (emb_neg_a, emb_neg_a_nodes), (emb_neg_b,  emb_neg_b_nodes) = (model.emb_model(neg_a),
            model.emb_model(neg_b))
        if pos_a:
            (emb_pos_a, emb_pos_a_nodes), (emb_pos_b,  emb_pos_b_nodes)  = (model.emb_model(pos_a),
                model.emb_model(pos_b))
            emb_as = torch.cat((emb_pos_a, emb_neg_a), dim=0)
            emb_bs = torch.cat((emb_pos_b, emb_neg_b), dim=0)
        else:
            emb_as, emb_bs = emb_neg_a, emb_neg_b
        pred = model(emb_as, emb_bs, emb_pos_a_nodes, emb_pos_b_nodes)
        raw_pred, raw_pred_nodes = model.predict(pred)
        if USE_ORCA_FEATS:
            import orca
            import matplotlib.pyplot as plt
            def make_feats(g):
                counts5 = np.array(orca.orbit_counts("node", 5, g))
                for v, n in zip(counts5, g.nodes):
                    if g.nodes[n]["node_feature"][0] > 0:
                        anchor_v = v
                        break
                v5 = np.sum(counts5, axis=0)
                return v5, anchor_v
            for i, (ga, gb) in enumerate(zip(neg_a.G, neg_b.G)):
                (va, na), (vb, nb) = make_feats(ga), make_feats(gb)
                if (va < vb).any() or (na < nb).any():
                    raw_pred[pos_a.num_graphs + i] = MAX_MARGIN_SCORE

        if args.method_type == "order":
            pred = model.clf_model(raw_pred.unsqueeze(1)).argmax(dim=-1)
            pred_nodes = model.clf_model_nodes(raw_pred_nodes.unsqueeze(1)).argmax(dim=-1)
            raw_pred *= -1
            raw_pred_nodes *=-1
        elif args.method_type == "ensemble":
            pred = torch.stack([m.clf_model(
                raw_pred.unsqueeze(1)).argmax(dim=-1) for m in model.models])
            for i in range(pred.shape[1]):
                print(pred[:,i])
            pred = torch.min(pred, dim=0)[0]
            raw_pred *= -1
        elif args.method_type == "mlp":
            raw_pred = raw_pred[:,1]
            pred = pred.argmax(dim=-1)
    all_raw_preds.append(raw_pred)
    all_preds.append(pred)
    all_labels.append(labels)
    # Nodes
    pred_nodes = []
    
    avg_auc = 0

    for idx in range(len(pos_b.node_label)):
        print('Graph pair {} start!'.format(idx))

        initial_alignment = alignment(emb_pos_a_nodes[idx], 
                                          emb_pos_b_nodes[idx],
                                          adj_pair[idx],
                                          align_mat_batch[idx],
                                          epoch=200)

        node_align_matrix, auc = admm_opt(emb_pos_a_nodes[idx], 
                                          emb_pos_b_nodes[idx],
                                          adj_pair[idx],
                                          align_mat_batch[idx],
                                          initial_alignment,
                                          epochs=100,
                                          p=0.5)

    avg_auc += auc

print('Average Test AUC {}'.format*(auc/len(pos_b.node_label)))
