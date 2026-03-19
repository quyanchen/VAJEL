from __future__ import division
from __future__ import print_function

import argparse
import csv
import os
import time

import numpy as np
import scipy.sparse as sp
import tensorflow as tf
from sklearn.metrics import average_precision_score, roc_auc_score

from input_data import load_AN
from model import VAJEL
from optimizer import OptimizerVAJEL
from preprocessing import construct_feed_dict, mask_test_edges, mask_test_feas, preprocess_graph, sparse_to_tuple


def sigmoid(x):
    x = np.clip(x, -500, 500)
    return 1.0 / (1 + np.exp(-x))


def get_roc_score_edge(sess, model, feed_dict, num_nodes, edges_pos, edges_neg):
    adj_rec = sess.run(model.reconstructions[0], feed_dict=feed_dict).reshape([num_nodes, num_nodes])
    preds = [sigmoid(adj_rec[e[0], e[1]]) for e in edges_pos]
    preds_neg = [sigmoid(adj_rec[e[0], e[1]]) for e in edges_neg]
    preds_all = np.hstack([preds, preds_neg])
    labels_all = np.hstack([np.ones(len(preds)), np.zeros(len(preds_neg))])
    return roc_auc_score(labels_all, preds_all), average_precision_score(labels_all, preds_all)


def get_roc_score_attr(sess, model, feed_dict, num_nodes, num_features, feas_pos, feas_neg):
    fea_rec = sess.run(model.reconstructions[1], feed_dict=feed_dict).reshape([num_nodes, num_features])
    preds = [sigmoid(fea_rec[e[0], e[1]]) for e in feas_pos]
    preds_neg = [sigmoid(fea_rec[e[0], e[1]]) for e in feas_neg]
    preds_all = np.hstack([preds, preds_neg])
    labels_all = np.hstack([np.ones(len(preds)), np.zeros(len(preds_neg))])
    return roc_auc_score(labels_all, preds_all), average_precision_score(labels_all, preds_all)


def parse_args():
    parser = argparse.ArgumentParser(description="Train VAJEL on one STKE-converted dataset.")
    parser.add_argument("--dataset", default="D:/#postgraduate/PCI/PCI_Code/VAJEL-master/homo", help="Dataset prefix for .edge/.node/.attidx")
    parser.add_argument("--output-prefix", default="result/homo_stke", help="Output prefix for embeddings")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--hidden1", type=int, default=64)
    parser.add_argument("--hidden2", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--cpu-only", action="store_true")
    return parser.parse_args()


def ensure_flags(args):
    flags = tf.flags
    FLAGS = flags.FLAGS
    if not hasattr(FLAGS, "learning_rate"):
        flags.DEFINE_float("learning_rate", args.learning_rate, "Initial learning rate.")
    if not hasattr(FLAGS, "epochs"):
        flags.DEFINE_integer("epochs", args.epochs, "Number of epochs to train.")
    if not hasattr(FLAGS, "hidden1"):
        flags.DEFINE_integer("hidden1", args.hidden1, "Number of units in hidden layer 1.")
    if not hasattr(FLAGS, "hidden2"):
        flags.DEFINE_integer("hidden2", args.hidden2, "Number of units in hidden layer 2.")
    if not hasattr(FLAGS, "weight_decay"):
        flags.DEFINE_float("weight_decay", args.weight_decay, "Weight for L2 loss.")
    if not hasattr(FLAGS, "dropout"):
        flags.DEFINE_float("dropout", args.dropout, "Dropout rate.")
    if not hasattr(FLAGS, "dataset"):
        flags.DEFINE_string("dataset", args.dataset, "Dataset prefix.")

    FLAGS.learning_rate = args.learning_rate
    FLAGS.epochs = args.epochs
    FLAGS.hidden1 = args.hidden1
    FLAGS.hidden2 = args.hidden2
    FLAGS.weight_decay = args.weight_decay
    FLAGS.dropout = args.dropout
    FLAGS.dataset = args.dataset
    return FLAGS


def main():
    args = parse_args()
    if args.cpu_only:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    out_dir = os.path.dirname(args.output_prefix) or "."
    os.makedirs(out_dir, exist_ok=True)

    FLAGS = ensure_flags(args)

    tf.reset_default_graph()
    sess = None

    adj, features = load_AN(args.dataset)
    if adj is None or features is None:
        raise RuntimeError("Failed to load dataset: {}".format(args.dataset))

    try:
        adj_orig = adj - sp.dia_matrix((adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape)
        adj_orig.eliminate_zeros()

        adj_train, _, val_edges, val_edges_false, test_edges, test_edges_false = mask_test_edges(adj)
        _, _, val_feas, val_feas_false, test_feas, test_feas_false = mask_test_feas(features)

        features_orig = features
        features_lil = sp.lil_matrix(features)
        adj_norm = preprocess_graph(adj_train)

        placeholders = {
            "features": tf.sparse_placeholder(tf.float32),
            "adj": tf.sparse_placeholder(tf.float32),
            "adj_orig": tf.sparse_placeholder(tf.float32),
            "features_orig": tf.sparse_placeholder(tf.float32),
            "dropout": tf.placeholder_with_default(0.0, shape=()),
        }

        num_nodes = adj_train.shape[0]
        features_tuple = sparse_to_tuple(features_lil.tocoo())
        num_features = features_tuple[2][1]
        features_nonzero = features_tuple[1].shape[0]

        model = VAJEL(placeholders, num_features, num_nodes, features_nonzero)

        pos_weight_u = float(adj_train.shape[0] * adj_train.shape[0] - adj_train.sum()) / adj_train.sum()
        norm_u = adj_train.shape[0] * adj_train.shape[0] / float((adj_train.shape[0] * adj_train.shape[0] - adj_train.sum()) * 2)
        pos_weight_a = float(features_tuple[2][0] * features_tuple[2][1] - len(features_tuple[1])) / len(features_tuple[1])
        norm_a = features_tuple[2][0] * features_tuple[2][1] / float((features_tuple[2][0] * features_tuple[2][1] - len(features_tuple[1])) * 2)

        with tf.name_scope("optimizer"):
            opt = OptimizerVAJEL(
                preds=model.reconstructions,
                labels=(
                    tf.reshape(tf.sparse_tensor_to_dense(placeholders["adj_orig"], validate_indices=False), [-1]),
                    tf.reshape(tf.sparse_tensor_to_dense(placeholders["features_orig"], validate_indices=False), [-1]),
                ),
                model=model,
                num_nodes=num_nodes,
                num_features=num_features,
                pos_weight_u=pos_weight_u,
                norm_u=norm_u,
                pos_weight_a=pos_weight_a,
                norm_a=norm_a,
            )

        sess = tf.Session()
        sess.run(tf.global_variables_initializer())

        adj_label = sparse_to_tuple(adj_train + sp.eye(adj_train.shape[0]))
        features_label = sparse_to_tuple(features_orig)

        for epoch in range(FLAGS.epochs):
            t = time.time()
            feed_dict = construct_feed_dict(adj_norm, adj_label, features_tuple, features_label, placeholders)
            feed_dict.update({placeholders["dropout"]: FLAGS.dropout})
            outs = sess.run([opt.opt_op, opt.cost, opt.accuracy, opt.log_lik, opt.kl], feed_dict=feed_dict)

            roc_curr, ap_curr = get_roc_score_edge(sess, model, feed_dict, num_nodes, val_edges, val_edges_false)
            roc_curr_a, ap_curr_a = get_roc_score_attr(sess, model, feed_dict, num_nodes, num_features, val_feas, val_feas_false)

            print(
                "Epoch {:04d} loss={:.5f} log_lik={:.5f} kl={:.5f} acc={:.5f} "
                "val_edge_roc={:.5f} val_edge_ap={:.5f} val_attr_roc={:.5f} val_attr_ap={:.5f} time={:.5f}".format(
                    epoch + 1,
                    outs[1],
                    outs[3],
                    outs[4],
                    outs[2],
                    roc_curr,
                    ap_curr,
                    roc_curr_a,
                    ap_curr_a,
                    time.time() - t,
                )
            )

        test_edge_roc, test_edge_ap = get_roc_score_edge(sess, model, feed_dict, num_nodes, test_edges, test_edges_false)
        test_attr_roc, test_attr_ap = get_roc_score_attr(sess, model, feed_dict, num_nodes, num_features, test_feas, test_feas_false)

        z_u = sess.run(model.z_u, feed_dict=feed_dict)
        z_a = sess.run(model.z_a, feed_dict=feed_dict)
        z_u_mean = sess.run(model.z_u_mean, feed_dict=feed_dict)
        z_a_mean = sess.run(model.z_a_mean, feed_dict=feed_dict)
        z_u_log_std = sess.run(model.z_u_log_std, feed_dict=feed_dict)
        z_a_log_std = sess.run(model.z_a_log_std, feed_dict=feed_dict)

        np.save(args.output_prefix + "_n.emb", z_u)
        np.save(args.output_prefix + "_a.emb", z_a)
        np.save(args.output_prefix + "_n_mu.emb", z_u_mean)
        np.save(args.output_prefix + "_a_mu.emb", z_a_mean)
        np.save(args.output_prefix + "_n_sig.emb", z_u_log_std)
        np.save(args.output_prefix + "_a_sig.emb", z_a_log_std)

        metrics_path = args.output_prefix + "_metrics.csv"
        with open(metrics_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["metric", "value"])
            w.writerow(["test_edge_roc", test_edge_roc])
            w.writerow(["test_edge_ap", test_edge_ap])
            w.writerow(["test_attr_roc", test_attr_roc])
            w.writerow(["test_attr_ap", test_attr_ap])
            w.writerow(["num_nodes", num_nodes])
            w.writerow(["num_attrs", num_features])

        print("Test edge ROC:", test_edge_roc)
        print("Test edge AP:", test_edge_ap)
        print("Test attr ROC:", test_attr_roc)
        print("Test attr AP:", test_attr_ap)
        print("Saved embeddings prefix:", args.output_prefix)
        print("Saved metrics:", metrics_path)

    finally:
        if sess is not None:
            sess.close()


if __name__ == "__main__":
    main()
