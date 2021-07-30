import sys
import logging
import numpy as np

import collections
from utils import *

def generic_dataloader(model_id, nb_test_genes, gene_shuffled_indx, gene_1, gene_2, fully_annotated_genes, 
                   gene_1_annotation, gene_2_annotation, prediction_value, sub_ontology_interested, experiment_mode):
    
    gene_indices_4_test = gene_shuffled_indx[model_id * (nb_test_genes):(model_id + 1) * (nb_test_genes)]    
    fully_annotated_genes_test = np.asarray(fully_annotated_genes)[gene_indices_4_test].tolist()
    fully_annotated_genes_train = list(set(fully_annotated_genes) - set(fully_annotated_genes_test))
        
    def _gene_list_to_gene_indeces_dict(genes):
        g_dic = {}
        for i, g in enumerate(genes):
            if g in g_dic:
                g_dic[g].add(i)
            else:
                g_dic[g] = {i} # starting point as a set
        return g_dic  # example {'S000003325': {0, 14756, 24551}, ..., 'S000000956': {9, 18615, 18, 15156}}
    
    g1_dic = _gene_list_to_gene_indeces_dict(gene_1)
    g2_dic = _gene_list_to_gene_indeces_dict(gene_2)

    for k in fully_annotated_genes_test: # keeping the ppi indeces for which the associated gene is NOT among the test genes
        g1_dic.pop(k, None) # removing test genes from g1_dic
        g2_dic.pop(k, None) # removing test genes from g2_dic

    g1_ind, g2_ind = set(), set()
    for g in g1_dic: g1_ind.update(g1_dic[g])
    for g in g2_dic: g2_ind.update(g2_dic[g])

    train_indices = list(g1_ind & g2_ind) # genes within ppi training pairs are NOT among the test genes
    if experiment_mode==1: test_indices = list(set(range(len(prediction_value))) - set(train_indices)) # mode 1: at least one of the testing genes in the pair is unique
    if experiment_mode==2: test_indices = list(set(range(len(prediction_value))) - set(list(g1_ind | g2_ind))) # mode 2: both testing genes in the pair are unique
    
    Gene_train_pair = [" ".join(i) for i in np.array(list(zip(gene_1, gene_2, map(str, prediction_value.astype(int).tolist()))))[train_indices]]
    Gene_test_pair = [" ".join(i) for i in np.array(list(zip(gene_1, gene_2, map(str, prediction_value.astype(int).tolist()))))[test_indices]]
    
    X_train_G1, X_train_G2, X_test_G1, X_test_G2 = [], [], [], []
    
    for sbo in sub_ontology_interested:
        
        # Training data (inclules (G1, G2, score) and (G2, G1, score) both due to the presence of highway layer)
        X_train_G1.append(np.concatenate((gene_1_annotation[sbo][train_indices], gene_2_annotation[sbo][train_indices]), axis=0))
        X_train_G2.append(np.concatenate((gene_2_annotation[sbo][train_indices], gene_1_annotation[sbo][train_indices]), axis=0))

        # test data
        X_test_G1.append(gene_1_annotation[sbo][test_indices])
        X_test_G2.append(gene_2_annotation[sbo][test_indices])
        
    y_train = np.concatenate((prediction_value[train_indices], prediction_value[train_indices]))
    y_test = prediction_value[test_indices]

    return Gene_train_pair, X_train_G1 + X_train_G2, y_train, Gene_test_pair, X_test_G1 + X_test_G2, y_test, fully_annotated_genes_train, fully_annotated_genes_test