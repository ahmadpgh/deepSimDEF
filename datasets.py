import os
import sys
import logging
import random
import operator
import numpy as np
import math
import tensorflow.keras
import tensorflow as tf

from tensorflow.keras import regularizers, initializers, optimizers
from tensorflow.python.keras.layers import Input, Embedding, AveragePooling1D, MaxPooling1D, Flatten, Dense, Dropout
from tensorflow.python.keras.layers import Activation, Reshape, Multiply, Add, Lambda, SpatialDropout1D, InputSpec, Dot
from tensorflow.python.keras.layers.merge import Concatenate
from tensorflow.python.keras.layers.noise import GaussianDropout, GaussianNoise
from tensorflow.keras.models import Model, model_from_json
from tensorflow.compat.v1.keras import backend as K
from tensorflow.keras.callbacks import Callback

from scipy.stats.stats import pearsonr, spearmanr
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score

import easydict
import socket
import datetime
from pytz import timezone

import collections
from collections import OrderedDict
from utils import *

def ppi_dataset(data_dir, data_file_name, sub_ontology_all, fully_annotated_genes, 
                              gene_annotations, gene_indeces, max_ann_len, partial_shuffle_percent, sub_ontology_interested):
    """Function to read the ppi file (contains positive and negative samples) and return a balanced dataset"""
    annotation_G1, annotation_G2 = {}, {}
    
    for sbo in sub_ontology_interested:
        annotation_G1[sbo], annotation_G2[sbo] = [], []
        G1, G2, _X1_ann, _X2_ann, _y = [], [], [], [], []
        with open('{}/{}'.format(data_dir, data_file_name)) as file_reader:
            positive_counter, negative_counter = 0, 0
            for i, line in enumerate(file_reader):
                if i == 0: continue  # skip the header
                values = line.split()
                if ((not values[0] in fully_annotated_genes) or (not values[1] in fully_annotated_genes)): continue
                G1.append(values[0])
                G2.append(values[1])
                _X1_ann.append(gene_annotations[sbo][gene_indeces[sbo][values[0]] - 1])
                _X2_ann.append(gene_annotations[sbo][gene_indeces[sbo][values[1]] - 1])
                _y.append(float(values[2]))
                if '1' in values[2]:
                    positive_counter += 1
                else:
                    negative_counter += 1
                    if negative_counter == positive_counter: break # making sure the dataset is balanced
                        
            annotation_G1[sbo] = pad_annotations(_X1_ann, maxlen=max_ann_len[sbo]) # zero-pad short annotations 
            annotation_G2[sbo] = pad_annotations(_X2_ann, maxlen=max_ann_len[sbo]) # zero-pad short annotations
            _y = partial_shuffle(_y, partial_shuffle_percent)
            interaction_pred = np.asarray(_y)
            
    return G1, G2, annotation_G1, annotation_G2, interaction_pred

def sequence_homology_dataset(data_dir, data_file_name, sub_ontology_all, fully_annotated_genes, gene_annotations,
                              gene_indeces, max_ann_len, sequence_homology_metric, partial_shuffle_percent, sub_ontology_interested):
    """Function to read the ppi file (contains positive and negative samples) and return a balanced dataset"""
    annotation_G1, annotation_G2 = {}, {}
    
    for sbo in sub_ontology_interested:
        annotation_G1[sbo], annotation_G2[sbo] = [], []
        G1, G2, _X1_ann, _X2_ann, _y = [], [], [], [], []
        with open('{}/{}'.format(data_dir, data_file_name)) as file_reader:
            for i, line in enumerate(file_reader):
                if i == 0: continue  # skip the header
                values = line.split()
                if ((not values[0] in fully_annotated_genes) or (not values[1] in fully_annotated_genes)): continue
                G1.append(values[0])
                G2.append(values[1])
                _X1_ann.append(gene_annotations[sbo][gene_indeces[sbo][values[0]] - 1])
                _X2_ann.append(gene_annotations[sbo][gene_indeces[sbo][values[1]] - 1])
                if sequence_homology_metric=="LRBS":
                    _y.append(float(values[2]))
                elif sequence_homology_metric=="RRBS":
                    _y.append(float(values[3]))
                        
            annotation_G1[sbo] = pad_annotations(_X1_ann, maxlen=max_ann_len[sbo]) # zero-pad short annotations 
            annotation_G2[sbo] = pad_annotations(_X2_ann, maxlen=max_ann_len[sbo]) # zero-pad short annotations
            _y = partial_shuffle(_y, partial_shuffle_percent)
            interaction_pred = np.asarray(_y)
            
    return G1, G2, annotation_G1, annotation_G2, interaction_pred

def gene_expression_dataset(data_dir, data_file_name, sub_ontology_all, fully_annotated_genes, gene_annotations,
                              gene_indeces, max_ann_len, partial_shuffle_percent, sub_ontology_interested):
    """Function to read the ppi file (contains positive and negative samples) and return a balanced dataset"""
    annotation_G1, annotation_G2 = {}, {}
    
    for sbo in sub_ontology_interested:
        annotation_G1[sbo], annotation_G2[sbo] = [], []
        G1, G2, _X1_ann, _X2_ann, _y = [], [], [], [], []
        with open('{}/{}'.format(data_dir, data_file_name)) as file_reader:
            for i, line in enumerate(file_reader):
                if i == 0: continue  # skip the header
                values = line.split()
                if ((not values[0] in fully_annotated_genes) or (not values[1] in fully_annotated_genes)): continue
                G1.append(values[0])
                G2.append(values[1])
                _X1_ann.append(gene_annotations[sbo][gene_indeces[sbo][values[0]] - 1])
                _X2_ann.append(gene_annotations[sbo][gene_indeces[sbo][values[1]] - 1])
                _y.append(float(values[2]))
                        
            annotation_G1[sbo] = pad_annotations(_X1_ann, maxlen=max_ann_len[sbo]) # zero-pad short annotations 
            annotation_G2[sbo] = pad_annotations(_X2_ann, maxlen=max_ann_len[sbo]) # zero-pad short annotations
            _y = partial_shuffle(_y, partial_shuffle_percent)
            interaction_pred = np.asarray(_y)
            
    return G1, G2, annotation_G1, annotation_G2, interaction_pred