import argparse
import time

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, average_precision_score

from model.FactorHNE import FactorHNE_lp
from utils.data import load_BioNet_data
from utils.utils import index_generator, parse_minibatch, EarlyStopping

# set random seed
np.random.seed(123456)

# Params
num_ntype = 4
attr_drop = 0
feat_drop = 0
lr = 0.005
weight_decay = 0.001

num_gene = 21584
num_dis = 15030

expected_metapaths = [
    [(0, 2, 0), (0, 0)],
    [(1, 0, 1), (1, 3, 1)]
]


def run_model_LastFM(feats_type, hidden_dim, out_dim, num_latent, num_heads, num_epochs, patience, batch_size,
                     neighbor_samples, repeat, save_postfix, omiga):
    adjlists_gd, _, type_mask, train_val_test_pos_gene_dis, train_val_test_neg_gene_dis = load_BioNet_data()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    features_list = []
    in_dims = []
    if feats_type == 0:
        # one-hot vector used to node features
        for i in range(num_ntype):
            dim = (type_mask == i).sum()
            in_dims.append(dim)
            indices = np.vstack((np.arange(dim), np.arange(dim)))
            indices = torch.LongTensor(indices)
            values = torch.FloatTensor(np.ones(dim))
            features_list.append(torch.sparse.FloatTensor(indices, values, torch.Size([dim, dim])).to(device))
    elif feats_type == 1:
        # all node feature inited by 10-dim zero vector
        for i in range(num_ntype):
            dim = 10
            num_nodes = (type_mask == i).sum()
            in_dims.append(dim)
            features_list.append(torch.zeros((num_nodes, 10)).to(device))

    train_pos_gene_dis = train_val_test_pos_gene_dis['train_pos_gene_dis']
    val_pos_gene_dis = train_val_test_pos_gene_dis['val_pos_gene_dis']
    test_pos_gene_dis = train_val_test_pos_gene_dis['test_pos_gene_dis']
    train_neg_gene_dis = train_val_test_neg_gene_dis['train_neg_gene_dis']
    val_neg_gene_dis = train_val_test_neg_gene_dis['val_neg_gene_dis']
    test_neg_gene_dis = train_val_test_neg_gene_dis['test_neg_gene_dis']
    y_true_test = np.array([1] * len(test_pos_gene_dis) + [0] * len(test_neg_gene_dis))

    auc_list = []
    ap_list = []
    for _ in range(repeat):
        net = FactorHNE_lp([2, 2], in_dims, num_latent, hidden_dim, hidden_dim, out_dim, num_heads, attr_drop,
                           feat_drop)
        net.to(device)
        optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)

        # training loop
        net.train()
        early_stopping = EarlyStopping(patience=patience, verbose=True,
                                       save_path='checkpoint/checkpoint-{}-{}-{}-{}-{}-{}.pt'.format(save_postfix,
                                                                                                     num_latent,
                                                                                                     out_dim,
                                                                                                     hidden_dim,
                                                                                                     int(omiga * 10),
                                                                                                     num_heads))
        dur1 = []
        dur2 = []
        dur3 = []
        train_pos_idx_generator = index_generator(batch_size=batch_size, num_data=len(train_pos_gene_dis))
        val_idx_generator = index_generator(batch_size=batch_size, num_data=len(val_pos_gene_dis), shuffle=False)
        for epoch in range(num_epochs):
            t_start = time.time()
            # training
            net.train()
            for iteration in range(train_pos_idx_generator.num_iterations()):
                # forward
                t0 = time.time()

                train_pos_idx_batch = train_pos_idx_generator.next()
                train_pos_idx_batch.sort()
                train_pos_gene_dis_batch = train_pos_gene_dis[train_pos_idx_batch].tolist()
                train_neg_idx_batch = np.random.choice(len(train_neg_gene_dis), len(train_pos_idx_batch))
                train_neg_idx_batch.sort()
                train_neg_user_artist_batch = train_neg_gene_dis[train_neg_idx_batch].tolist()

                # shuffle
                num_pos = train_pos_idx_batch.shape[0]
                train_batch = np.concatenate([train_pos_gene_dis_batch, train_neg_user_artist_batch], axis=0)
                y_label = np.zeros((train_batch.shape[0], 1), dtype=int)
                y_label[:num_pos] = 1
                train_data = np.concatenate([train_batch, y_label], axis=1)
                np.random.shuffle(train_data)
                train_batch = train_data[:, :-1]
                y_label = train_data[:, -1]

                # train_pos_g_lists, train_pos_indices_lists, train_pos_idx_batch_mapped_lists = parse_minibatch_LastFM(
                #     adjlists_gd, edge_metapath_indices_list_ua, train_pos_gene_dis_batch, device, neighbor_samples,
                #     use_masks, num_user)
                # train_neg_g_lists, train_neg_indices_lists, train_neg_idx_batch_mapped_lists = parse_minibatch_LastFM(
                #     adjlists_gd, edge_metapath_indices_list_ua, train_neg_user_artist_batch, device, neighbor_samples,
                #     no_masks, num_user)
                train_g_lists, train_node_indices_lists, train_idx_batch_mapped_lists = parse_minibatch(
                    adjlists_gd, train_batch, num_gene, neighbor_samples)

                t1 = time.time()
                dur1.append(t1 - t0)

                # [pos_embedding_gene, pos_embedding_dis], _ = net(
                #     (train_pos_g_lists, features_list, type_mask, train_pos_indices_lists,
                #      train_pos_idx_batch_mapped_lists))
                # [neg_embedding_user, neg_embedding_artist], _ = net(
                #     (train_neg_g_lists, features_list, type_mask, train_neg_indices_lists,
                #      train_neg_idx_batch_mapped_lists))
                # pos_embedding_gene = pos_embedding_gene.view(-1, 1, pos_embedding_gene.shape[1])
                # pos_embedding_dis = pos_embedding_dis.view(-1, pos_embedding_dis.shape[1], 1)
                # neg_embedding_user = neg_embedding_user.view(-1, 1, neg_embedding_user.shape[1])
                # neg_embedding_artist = neg_embedding_artist.view(-1, neg_embedding_artist.shape[1], 1)
                # pos_out = torch.bmm(pos_embedding_gene, pos_embedding_dis)
                # neg_out = -torch.bmm(neg_embedding_user, neg_embedding_artist)
                # train_loss = -torch.mean(F.logsigmoid(pos_out) + F.logsigmoid(neg_out))
                # print(train_g_lists)
                embedding_gene, embedding_dis = net(
                    train_g_lists, features_list, type_mask, train_node_indices_lists, train_idx_batch_mapped_lists)
                embedding_gene = embedding_gene.view(-1, 1, embedding_gene.shape[1])
                embedding_dis = embedding_dis.view(-1, embedding_dis.shape[1], 1)
                out = torch.bmm(embedding_gene, embedding_dis)
                class_op = torch.LongTensor([1 if l == 1 else -1 for l in y_label]).view(-1, 1, 1).to(device)
                train_loss = -torch.mean(F.logsigmoid(out * class_op)) + net.get_loss() * omiga

                t2 = time.time()
                dur2.append(t2 - t1)

                # autograd
                optimizer.zero_grad()
                train_loss.backward()
                optimizer.step()

                t3 = time.time()
                dur3.append(t3 - t2)

                # print training info
                if iteration % 100 == 0:
                    print(
                        'Epoch {:05d} | Iteration {:05d} | Train_Loss {:.4f} | Time1(s) {:.4f} | Time2(s) {:.4f} | '
                        'Time3(s) {:.4f}'.format(
                            epoch, iteration, train_loss.item(), np.mean(dur1), np.mean(dur2), np.mean(dur3)))
            # validation
            net.eval()
            val_loss = []
            with torch.no_grad():
                for iteration in range(val_idx_generator.num_iterations()):
                    # forward
                    val_idx_batch = val_idx_generator.next()
                    val_pos_gene_dis_batch = val_pos_gene_dis[val_idx_batch].tolist()
                    val_neg_gene_dis_batch = val_neg_gene_dis[val_idx_batch].tolist()
                    val_pos_g_lists, val_pos_indices_lists, val_pos_idx_batch_mapped_lists = parse_minibatch(
                        adjlists_gd, val_pos_gene_dis_batch, num_gene, neighbor_samples)
                    val_neg_g_lists, val_neg_indices_lists, val_neg_idx_batch_mapped_lists = parse_minibatch(
                        adjlists_gd, val_neg_gene_dis_batch, num_gene, neighbor_samples)

                    pos_embedding_gene, pos_embedding_dis = net(
                        val_pos_g_lists, features_list, type_mask, val_pos_indices_lists,
                        val_pos_idx_batch_mapped_lists)
                    neg_embedding_user, neg_embedding_artist = net(
                        val_neg_g_lists, features_list, type_mask, val_neg_indices_lists,
                        val_neg_idx_batch_mapped_lists)
                    pos_embedding_gene = pos_embedding_gene.view(-1, 1, pos_embedding_gene.shape[1])
                    pos_embedding_dis = pos_embedding_dis.view(-1, pos_embedding_dis.shape[1], 1)
                    neg_embedding_user = neg_embedding_user.view(-1, 1, neg_embedding_user.shape[1])
                    neg_embedding_artist = neg_embedding_artist.view(-1, neg_embedding_artist.shape[1], 1)

                    pos_out = torch.bmm(pos_embedding_gene, pos_embedding_dis)
                    neg_out = -torch.bmm(neg_embedding_user, neg_embedding_artist)
                    val_loss.append(-torch.mean(F.logsigmoid(pos_out) + F.logsigmoid(neg_out)) + net.get_loss() * omiga)
                val_loss = torch.mean(torch.tensor(val_loss))
            t_end = time.time()
            # print validation info
            print('Epoch {:05d} | Val_Loss {:.4f} | Time(s) {:.4f}'.format(
                epoch, val_loss.item(), t_end - t_start))
            # early stopping
            early_stopping(val_loss, net)
            if early_stopping.early_stop:
                print('Early stopping!')
                break

        test_idx_generator = index_generator(batch_size=batch_size, num_data=len(test_pos_gene_dis), shuffle=False)
        net.load_state_dict(
            torch.load(
                'checkpoint/checkpoint-{}-{}-{}-{}-{}-{}.pt'.format(save_postfix, num_latent, out_dim, hidden_dim,
                                                                    int(omiga * 10), num_heads)))
        net.eval()
        pos_proba_list = []
        neg_proba_list = []
        with torch.no_grad():
            for iteration in range(test_idx_generator.num_iterations()):
                # forward
                test_idx_batch = test_idx_generator.next()
                test_pos_user_artist_batch = test_pos_gene_dis[test_idx_batch].tolist()
                test_neg_user_artist_batch = test_neg_gene_dis[test_idx_batch].tolist()
                test_pos_g_lists, test_pos_indices_lists, test_pos_idx_batch_mapped_lists = parse_minibatch(
                    adjlists_gd, test_pos_user_artist_batch, num_gene, neighbor_samples)
                test_neg_g_lists, test_neg_indices_lists, test_neg_idx_batch_mapped_lists = parse_minibatch(
                    adjlists_gd, test_neg_user_artist_batch, num_gene, neighbor_samples)

                pos_embedding_gene, pos_embedding_dis = net(
                    test_pos_g_lists, features_list, type_mask, test_pos_indices_lists,
                    test_pos_idx_batch_mapped_lists)
                neg_embedding_user, neg_embedding_artist = net(
                    test_neg_g_lists, features_list, type_mask, test_neg_indices_lists,
                    test_neg_idx_batch_mapped_lists)
                pos_embedding_gene = pos_embedding_gene.view(-1, 1, pos_embedding_gene.shape[1])
                pos_embedding_dis = pos_embedding_dis.view(-1, pos_embedding_dis.shape[1], 1)
                neg_embedding_user = neg_embedding_user.view(-1, 1, neg_embedding_user.shape[1])
                neg_embedding_artist = neg_embedding_artist.view(-1, neg_embedding_artist.shape[1], 1)

                pos_out = torch.bmm(pos_embedding_gene, pos_embedding_dis).flatten()
                neg_out = torch.bmm(neg_embedding_user, neg_embedding_artist).flatten()
                pos_proba_list.append(torch.sigmoid(pos_out))
                neg_proba_list.append(torch.sigmoid(neg_out))
            y_prob_test = torch.cat(pos_proba_list + neg_proba_list)
            y_prob_test = y_prob_test.cpu().numpy()
        np.savez('prediction_result-{}-{}-{}-{}-{}-{}.npz'.format(save_postfix, num_latent, out_dim, hidden_dim,
                                                                  int(omiga * 10), num_heads), y_true=y_true_test,
                 y_pred=y_prob_test)
        auc = roc_auc_score(y_true_test, y_prob_test)
        average_precision = average_precision_score(y_true_test, y_prob_test)
        with open('auc_ap--{}-{}-{}-{}-{}-{}.txt'.format(save_postfix, num_latent, out_dim, hidden_dim, int(omiga * 10),
                                                      num_heads), 'w+') as f:
            f.write('AUC = {}'.format(auc) + '\n')
            f.write('AP = {}'.format(average_precision) + '\n')

        print('Link Prediction Test')
        print('AUC = {}'.format(auc))
        print('AP = {}'.format(average_precision))
        auc_list.append(auc)
        ap_list.append(average_precision)

    print('----------------------------------------------------------------')
    print('Link Prediction Tests Summary')
    print('AUC_mean = {}, AUC_std = {}'.format(np.mean(auc_list), np.std(auc_list)))
    print('AP_mean = {}, AP_std = {}'.format(np.mean(ap_list), np.std(ap_list)))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='FactorHNE testing for the biomedical network dataset')
    ap.add_argument('--feats-type', type=int, default=0,
                    help='Type of the node features used. ' +
                         '0 - all id vectors; ' +
                         '1 - all zero vector. Default is 0.')
    ap.add_argument('--hidden-dim', type=int, default=64, help='Dimension of the node hidden state. Default is 64.')
    ap.add_argument('--omiga', type=float, default=0.5, help='Weight of the factor graph loss. Default is 0.2.')
    ap.add_argument('--out-dim', type=int, default=64, help='Dimension of the node final embedding. Default is 64.')
    ap.add_argument('--num-latent', type=int, default=2,
                    help='Number of the factor graph in factorization step. Default is 4.')
    ap.add_argument('--num-heads', type=int, default=8, help='Number of the attention heads. Default is 8.')
    ap.add_argument('--epoch', type=int, default=100, help='Number of epochs. Default is 100.')
    ap.add_argument('--patience', type=int, default=3, help='Patience. Default is 5.')
    ap.add_argument('--batch-size', type=int, default=8, help='Batch size. Default is 8.')
    ap.add_argument('--samples', type=int, default=100, help='Number of neighbors sampled. Default is 100.')
    ap.add_argument('--repeat', type=int, default=1, help='Repeat the training and testing for N times. Default is 1.')
    ap.add_argument('--save-postfix', default='FactorHNE',
                    help='Postfix for the saved model and result. Default is LastFM.')

    args = ap.parse_args()
    run_model_LastFM(args.feats_type, args.hidden_dim, args.out_dim, args.num_latent, args.num_heads, args.epoch,
                     args.patience, args.batch_size, args.samples, args.repeat, args.save_postfix, args.omiga)
