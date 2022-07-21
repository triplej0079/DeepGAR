from common import utils
from collections import defaultdict
from datetime import datetime
from sklearn.metrics import roc_auc_score, confusion_matrix
from sklearn.metrics import precision_recall_curve, average_precision_score
import torch

USE_ORCA_FEATS = False # whether to use orca motif counts along with embeddings
MAX_MARGIN_SCORE = 1e9 # a very large margin score to given orca constraints

def validation(args, model, test_pts, logger, batch_n, epoch, verbose=False):
    # test on new motifs
    model.eval()
    all_raw_preds, all_preds, all_labels = [], [], []
    all_raw_preds_nodes, all_preds_nodes, all_labels_nodes = [], [], []
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
        all_raw_preds_nodes.append(raw_pred_nodes)
        all_preds_nodes.append(pred_nodes)
        all_labels_nodes.append(labels_nodes)
    pred = torch.cat(all_preds, dim=-1)
    labels = torch.cat(all_labels, dim=-1)
    raw_pred = torch.cat(all_raw_preds, dim=-1)
    acc = torch.mean((pred == labels).type(torch.float))
    prec = (torch.sum(pred * labels).item() / torch.sum(pred).item() if
            torch.sum(pred) > 0 else float("NaN"))
    recall = (torch.sum(pred * labels).item() /
              torch.sum(labels).item() if torch.sum(labels) > 0 else
              float("NaN"))
    labels = labels.detach().cpu().numpy()
    raw_pred = raw_pred.detach().cpu().numpy()
    pred = pred.detach().cpu().numpy()
    pred_nodes = pred_nodes.detach().cpu().numpy()
    auroc = roc_auc_score(labels, raw_pred)
    avg_prec = average_precision_score(labels, raw_pred)
    tn, fp, fn, tp = confusion_matrix(labels, pred).ravel()
    # Node Level
    pred_nodes = torch.cat(all_preds_nodes, dim=-1)
    labels_nodes = torch.cat(all_labels_nodes, dim=-1)
    raw_pred_nodes = torch.cat(all_raw_preds_nodes, dim=-1)
    acc_nodes = torch.mean((pred_nodes == labels_nodes).type(torch.float))
    prec_nodes = (torch.sum(pred_nodes * labels_nodes).item() / torch.sum(pred_nodes).item() if
        torch.sum(pred_nodes) > 0 else float("NaN"))
    recall_nodes = (torch.sum(pred_nodes * labels_nodes).item() /
        torch.sum(labels_nodes).item() if torch.sum(labels_nodes) > 0 else
        float("NaN"))

    labels_nodes = labels_nodes.detach().cpu().numpy()
    raw_pred_nodes = raw_pred_nodes.detach().cpu().numpy()
    pred_nodes = pred_nodes.detach().cpu().numpy()
    auroc_nodes = roc_auc_score(labels_nodes, raw_pred_nodes)
    avg_prec_nodes = average_precision_score(labels_nodes, raw_pred_nodes)
    tn_nd, fp_nd, fn_nd, tp_nd = confusion_matrix(labels_nodes, pred_nodes).ravel()
    if verbose:
        import matplotlib.pyplot as plt
        precs, recalls, threshs = precision_recall_curve(labels, raw_pred)
        plt.plot(recalls, precs)
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.savefig("plots/precision-recall-curve.png")
        print("Saved PR curve plot in plots/precision-recall-curve.png")

    print("\n{}".format(str(datetime.now())))
    print("Validation. Epoch {}. Acc: {:.4f}. "
        "P: {:.4f}. R: {:.4f}. AUROC: {:.4f}. AP: {:.4f}.\n     "
        "TN: {}. FP: {}. FN: {}. TP: {}".format(epoch,
            acc, prec, recall, auroc, avg_prec,
            tn, fp, fn, tp))
    print("Validation Node Level. Epoch {}. Acc: {:.4f}. "
          "P: {:.4f}. R: {:.4f}. AUROC: {:.4f}. AP: {:.4f}.\n     "
          "TN: {}. FP: {}. FN: {}. TP: {}".format(epoch,
                                                  acc_nodes, prec_nodes, recall_nodes, auroc_nodes, avg_prec_nodes,
                                                  tn_nd, fp_nd, fn_nd, tp_nd))

    if not args.test:
        logger.add_scalar("Accuracy/test", acc, batch_n)
        logger.add_scalar("Precision/test", prec, batch_n)
        logger.add_scalar("Recall/test", recall, batch_n)
        logger.add_scalar("AUROC/test", auroc, batch_n)
        logger.add_scalar("AvgPrec/test", avg_prec, batch_n)
        logger.add_scalar("TP/test", tp, batch_n)
        logger.add_scalar("TN/test", tn, batch_n)
        logger.add_scalar("FP/test", fp, batch_n)
        logger.add_scalar("FN/test", fn, batch_n)
        print("Saving {}".format(args.model_path))
        torch.save(model.state_dict(), args.model_path)

    if verbose:
        conf_mat_examples = defaultdict(list)
        idx = 0
        for pos_a, pos_b, neg_a, neg_b in test_pts:
            if pos_a:
                pos_a = pos_a.to(utils.get_device())
                pos_b = pos_b.to(utils.get_device())
            neg_a = neg_a.to(utils.get_device())
            neg_b = neg_b.to(utils.get_device())
            for list_a, list_b in [(pos_a, pos_b), (neg_a, neg_b)]:
                if not list_a: continue
                for a, b in zip(list_a.G, list_b.G):
                    correct = pred[idx] == labels[idx]
                    conf_mat_examples[correct, pred[idx]].append((a, b))
                    idx += 1
                
    

if __name__ == "__main__":
    from subgraph_matching.train import main
    main(force_test=True)
