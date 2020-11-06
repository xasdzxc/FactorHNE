import pickle

import networkx as nx
import numpy as np
import scipy


def load_BioNet_data(prefix='data/preprocessed'):
    infile = open(prefix + '/0/0-1-0.adjlist', 'r')
    adjlist00 = [line.strip() for line in infile]
    adjlist00 = adjlist00
    infile.close()
    infile = open(prefix + '/0/0-2-0.adjlist', 'r')
    adjlist01 = [line.strip() for line in infile]
    adjlist01 = adjlist01
    infile.close()
    infile = open(prefix + '/0/0-0.adjlist', 'r')
    adjlist02 = [line.strip() for line in infile]
    adjlist02 = adjlist02
    infile.close()
    infile = open(prefix + '/1/1-0-1.adjlist', 'r')
    adjlist10 = [line.strip() for line in infile]
    adjlist10 = adjlist10
    infile.close()
    infile = open(prefix + '/1/1-0-0-1.adjlist', 'r')
    adjlist11 = [line.strip() for line in infile]
    adjlist11 = adjlist11
    infile.close()
    infile = open(prefix + '/1/1-3-1.adjlist', 'r')
    adjlist12 = [line.strip() for line in infile]
    adjlist12 = adjlist12
    infile.close()

    adjM = scipy.sparse.load_npz(prefix + '/adjM.npz')
    type_mask = np.load(prefix + '/node_types.npy')
    train_val_test_pos_gene_dis = np.load(prefix + '/train_val_test_pos_gene_dis.npz')
    train_val_test_neg_gene_dis = np.load(prefix + '/train_val_test_neg_gene_dis.npz')

    return [[adjlist01, adjlist02], [adjlist10, adjlist12]], \
           adjM, type_mask, train_val_test_pos_gene_dis, train_val_test_neg_gene_dis


# load skipgram-format embeddings, treat missing node embeddings as zero vectors
def load_skipgram_embedding(path, num_embeddings):
    count = 0
    with open(path, 'r') as infile:
        _, dim = list(map(int, infile.readline().strip().split(' ')))
        embeddings = np.zeros((num_embeddings, dim))
        for line in infile.readlines():
            count += 1
            line = line.strip().split(' ')
            embeddings[int(line[0])] = np.array(list(map(float, line[1:])))
    print('{} out of {} nodes have non-zero embeddings'.format(count, num_embeddings))
    return embeddings


# load metapath2vec embeddings
def load_metapath2vec_embedding(path, type_list, num_embeddings_list, offset_list):
    count = 0
    with open(path, 'r') as infile:
        _, dim = list(map(int, infile.readline().strip().split(' ')))
        embeddings_dict = {type: np.zeros((num_embeddings, dim)) for type, num_embeddings in
                           zip(type_list, num_embeddings_list)}
        offset_dict = {type: offset for type, offset in zip(type_list, offset_list)}
        for line in infile.readlines():
            line = line.strip().split(' ')
            # drop </s> token
            if line[0] == '</s>':
                continue
            count += 1
            embeddings_dict[line[0][0]][int(line[0][1:]) - offset_dict[line[0][0]]] = np.array(
                list(map(float, line[1:])))
    print('{} node embeddings loaded'.format(count))
    return embeddings_dict
