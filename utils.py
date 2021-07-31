"""extra utility functions"""
import os
import sys
import logging
import random
import operator
import numpy as np
import math
import tensorflow.keras
import tensorflow as tf

from tensorflow.python.keras.layers import Dropout
from tensorflow.keras.callbacks import Callback
from sklearn.metrics import f1_score, precision_score, recall_score

import socket
import datetime

import collections
from collections import OrderedDict

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
        
def log(args):
    """Create logging directory structure according to args."""
    if hasattr(args, "checkpoint") and args.checkpoint:
        return _log_from_checkpoint(args)
    else:
        stamp = datetime.date.strftime(datetime.datetime.now(), "%Y.%m.%d-%Hh%Mm%Ss") + "_{}".format(socket.gethostname())
        full_logdir = os.path.join(args.log_dir, args.log_name, stamp)
        os.makedirs(full_logdir, exist_ok=True)
        args.log_dir = "{}:{}".format(socket.gethostname(), full_logdir)
        _log_args(full_logdir, args)
    return full_logdir, 0

def _log_from_checkpoint(args):
    """Infer logging directory from checkpoint file."""
    int_dir, checkpoint_name = os.path.split(args.checkpoint)
    logdir = os.path.dirname(int_dir)
    checkpoint_num = int(checkpoint_name.split('_')[1])
    _log_args(logdir, args, modified_iter=checkpoint_num)
    return logdir, checkpoint_num

def _log_args(logdir, args, modified_iter=0):
    """Write log of current arguments to text."""
    keys = sorted(arg for arg in dir(args) if not arg.startswith("_"))
    args_dict = {key: getattr(args, key) for key in keys}
    with open(os.path.join(logdir, "config.log"), "a") as f:
        f.write("Values at iteration {}\n".format(modified_iter))
        for k in keys:
            s = ": ".join([k,str(args_dict[k])]) + "\n"
            f.write(s)

def tokenize_annotations(annotations):
    """Function to tokenize & convert a list of genes GO term annotations to their equivalent list of GO term annotation ids"""
    go_terms = []
    for annotation in annotations:
        go_terms.extend(annotation.split())
    go_terms_freq = OrderedDict({k: v for k, v in sorted(collections.Counter(go_terms).items(), key=lambda item: item[1], reverse=True)}) # example output: OrderedDict([('GO0006810', 804), ('GO0006351', 512), ('GO0006355', 497), ..., ('GO0006351', 56), ('GO0006873', 13), ('GO0034427', 2)])
    go_term_indeces = {go_term:indx+1 for indx, go_term in enumerate(go_terms_freq)} # each index represents a one-hot vector for its assiciate GO term
    annotations_to_annotation_ids = []
    for annotation in annotations:
        annotations_to_annotation_ids.append([go_term_indeces[go_term] for go_term in annotation.split()])
    return annotations_to_annotation_ids, go_term_indeces

def pad_annotations(annotations, maxlen):
    """Function to zero-pad a lists of annotations (the ids), each annotation belongs to one gene"""
    return np.array([[0]*(maxlen-len(annotation))+annotation for annotation in annotations])

def exp_decay(epoch, initial_lrate):
    """Function to apply exponential decay to the initial learning rate"""
    k = 0.1
    lrate = initial_lrate * math.exp(-k*epoch)
    return lrate

def save_gene_pairs(logdir, model_id, train_pair, test_pair, train_gene, test_gene):
    """Function to store the gene pairs used for training and testing in different models"""
    path="{}/gene_pairs/model_{}".format(logdir, model_id+1)
    os.makedirs(path, exist_ok=True)
    with open("{}/train_pair.txt".format(path), "w") as fw: fw.write("\n".join(train_pair))
    with open("{}/test_pair.txt".format(path), "w") as fw: fw.write("\n".join(test_pair))
    with open("{}/train_gene.txt".format(path), "w") as fw: fw.write("\n".join(train_gene))
    with open("{}/test_gene.txt".format(path), "w") as fw: fw.write("\n".join(test_gene))

def save_model(path, models, epoch, verbose=True):
    """Function to save deepSimDEF models (checkpointing)"""
    path = "{}/model_checkpoints/epoch_{}".format(path, epoch)
    os.makedirs(path, exist_ok=True) # create the directory if it does not exist
    for ind in range(len(models)):
        if verbose: print("Saving model {} to disk ...".format(ind+1))
        model_json = models[ind].to_json()
        with open("{}/model_{}.json".format(path, ind+1, epoch), "w") as json_file:
            json_file.write(model_json)
            models[ind].save_weights("{}/model_{}.h5".format(path, ind+1, epoch))
        if verbose: print("The model and its weights are saved!!")

def save_embeddings(path, models, go_term_indeces, sub_ontology_interested, embedding_save, epoch, verbose=True):
    """Function to save deepSimDEF updated embeddings"""
    path = "{}/model_embeddings_updated/epoch_{}".format(path, epoch)
    for ind in range(len(models)):
        names = [weight.name for layer in models[ind].layers for weight in layer.weights]
        weights = models[ind].get_weights()
        for sbo in sub_ontology_interested:
            sbo_path = "{}/{}".format(path, sbo)
            os.makedirs(sbo_path, exist_ok=True) # create the directory if it does not exist
            for name, weight in zip(names, weights):
                if "embedding_{}_{}".format(sbo, ind) in name:
                    embeddings=weight
            go_ids = [i for i, _ in sorted(go_term_indeces[sbo].items(), key=operator.itemgetter(1))] # list of all GO terms in this particular ontology, sorted by the numbers of their annotations
            with open("{}/{}_Model_{}.emb".format(sbo_path, embedding_save[sbo], ind+1), "w") as file_writer:
                for i in range(len(go_ids)):
                    file_writer.write((go_ids[i] + " ").replace("\r", "\\r"))
                    file_writer.write(" ".join([str(j) for j in embeddings[i+1]])+"\n")
        if verbose: print("The GO term embeddings for model {} are Saved!!".format(ind+1))

class DropAnnotation(Dropout):
    """Timestep Dropout.

    This version performs the same function as Dropout, however it drops
    entire timesteps (e.g., words embeddings in NLP tasks) instead of individual elements (features).

    # Arguments
        rate: float between 0 and 1. Fraction of the timesteps to drop.

    # Input shape
        3D tensor with shape:
        `(samples, timesteps, channels)`

    # Output shape
        Same as input

    # References
        - A Theoretically Grounded Application of Dropout in Recurrent Neural Networks (https://arxiv.org/pdf/1512.05287)
    """
    def __init__(self, rate, **kwargs):
        super(DropAnnotation, self).__init__(rate, **kwargs)
        self.input_spec = InputSpec(ndim=3)

    def _get_noise_shape(self, inputs):
        input_shape = K.shape(inputs)
        noise_shape = (input_shape[0], input_shape[1], 1)
        return noise_shape

def partial_shuffle(array, percent=0.0):
    """Function for partial shuffling of (annotations of) ground truth (for 'Negative Control' experiments)"""
    # which characters are to be shuffled:
    idx_todo = random.sample(range(len(array)), int(len(array) * percent))

    # what are the new positions of these to-be-shuffled characters:
    idx_target = idx_todo[:]
    random.shuffle(idx_target)

    # map all "normal" character positions {0:0, 1:1, 2:2, ...}
    mapper = dict((i, i) for i in range(len(array)))

    # update with all shuffles in the string: {old_pos:new_pos, old_pos:new_pos, ...}
    mapper.update(zip(idx_todo, idx_target))

    # use mapper to modify the string:
    return [array[mapper[i]] for i in range(len(array))]

def cal_iter_time(former_iteration_endpoint, e, args, tz):
    """Calculating 'Computation Time' for this round of iteration"""
    current_iteration_endpoint = datetime.datetime.now(tz)
    current_iteration_elapsed = str(current_iteration_endpoint - former_iteration_endpoint).split(".")[0]
    expected_running_time_left = str((current_iteration_endpoint - former_iteration_endpoint)*(args.nb_epoch-(e+1))).split(".")[0]
    temp = current_iteration_elapsed.split(":")
    if int(temp[0])==0 and int(temp[1])==0: current_iteration_elapsed = temp[2]
    elif int(temp[0])==0: current_iteration_elapsed = temp[1]+":"+temp[2]
    former_iteration_endpoint = current_iteration_endpoint
    print("This epoch took {} seconds to run; the expected running time left to go through all epochs is {}!\n".format(
            current_iteration_elapsed, expected_running_time_left))
    return former_iteration_endpoint, current_iteration_elapsed

def log_model_result_for_sequence_homology(epoch, model, lr, 
                     best_pearson, progress_pearson, pearson,
                     best_spearman, progress_spearman, spearman, logdir):
    """logging the results of each model (within each epoch)"""
    with open('{}/result.log'.format(logdir), 'a') as fw:
        fw.write('| epoch {:3d} | model {:2d} | lr {:.5f} '
                 '| best-pearson {:.4f} | progress-prs {:+.4f} | pearson {:.4f} '
                 '| best-spearman {:.4f} | progress-spr {:+.4f} | spearman {:.4f}\n'.format(
        epoch, model, lr, best_pearson, progress_pearson, pearson, best_spearman, progress_spearman, spearman))

def log_epoch_result_for_sequence_homology(epoch, epochs, time, best_pearson, pearson, best_epoch_pearson, 
                     best_spearman, spearman, best_epoch_spearman, logdir):
    """logging the results of the given epoch (aggregated across all models)"""
    with open('{}/result.log'.format(logdir), 'a') as fw:
        fw.write("-"*161+"\n")
        fw.write("| epoch result {:3d}/{} | time: {}s | best-pearson {:.4f} | pearson {:.4f} | best epoch prs {:3d} | best-spearman {:.4f} | spearman {:.4f} | best epoch spr {:3d}\n".format(
        epoch, epochs, time, best_pearson, pearson, best_epoch_pearson, 
                     best_spearman, spearman, best_epoch_spearman))
        fw.write("-"*161+"\n")

def log_model_result_for_gene_expression(epoch, model, lr, 
                     best_pearson, progress_pearson, pearson,
                     best_spearman, progress_spearman, spearman, logdir):
    """logging the results of each model (within each epoch)"""
    with open('{}/result.log'.format(logdir), 'a') as fw:
        fw.write('| epoch {:3d} | model {:2d} | lr {:.5f} '
                 '| best-pearson {:.4f} | progress-prs {:+.4f} | pearson {:.4f} '
                 '| best-spearman {:.4f} | progress-spr {:+.4f} | spearman {:.4f}\n'.format(
        epoch, model, lr, best_pearson, progress_pearson, pearson, best_spearman, progress_spearman, spearman))

def log_epoch_result_for_gene_expression(epoch, epochs, time, best_pearson, pearson, best_epoch_pearson, 
                     best_spearman, spearman, best_epoch_spearman, logdir):
    """logging the results of the given epoch (aggregated across all models)"""
    with open('{}/result.log'.format(logdir), 'a') as fw:
        fw.write("-"*161+"\n")
        fw.write("| epoch result {:3d}/{} | time: {}s | best-pearson {:.4f} | pearson {:.4f} | best epoch prs {:3d} | best-spearman {:.4f} | spearman {:.4f} | best epoch spr {:3d}\n".format(
        epoch, epochs, time, best_pearson, pearson, best_epoch_pearson, 
                     best_spearman, spearman, best_epoch_spearman))
        fw.write("-"*161+"\n")

def log_model_result_for_ppi(epoch, model, lr, 
                     best_f1_measure, progress_f1_measure, current_f1_measure, logdir):
    """logging the results of each model (within each epoch)"""
    with open('{}/result.log'.format(logdir), 'a') as fw:
        fw.write('| epoch {:3d} | model {:2d} | lr {:.5f} '
                 '| best-F1score {:.2f} | progress-F1score {:+.2f} | F1score {:.2f}\n'.format(
        epoch, model, lr, best_f1_measure, progress_f1_measure, current_f1_measure))

def log_epoch_result_for_ppi(epoch, epochs, time, best_cv_f1_measure, f1_res, best_epoch, logdir):
    """logging the results of the given epoch (aggregated across all models)"""
    with open('{}/result.log'.format(logdir), 'a') as fw:
        fw.write("-"*97+"\n")
        fw.write("| epoch result {:3d}/{} | time: {}s | best-F1score {:.2f} | F1score {:.2f} | best epoch {:3d}\n".format(
        epoch, epochs, time, best_cv_f1_measure, f1_res, best_epoch))
        fw.write("-"*97+"\n")

class F1Score(Callback):
    """F1Score should be computed across all data (not in single batches defined in metrics in model.compile)"""
    def __init__(self, validation_data=None):
        super(F1Score, self).__init__()
        self.validation_data = validation_data
        
    def on_train_begin(self, logs={}):
        self.val_f1s = []
        self.val_recalls = []
        self.val_precisions = []

    def on_epoch_end(self, epoch, logs={}):
        val_predict = (np.asarray(self.model.predict(self.validation_data[0]))).round()
        val_targ = self.validation_data[1]
        _val_f1 = f1_score(val_targ, val_predict)
        _val_recall = recall_score(val_targ, val_predict)
        _val_precision = precision_score(val_targ, val_predict)
        self.val_f1s.append(_val_f1)
        self.val_recalls.append(_val_recall)
        self.val_precisions.append(_val_precision)
        print(" val_f1: {:.5f}".format(_val_f1))


def extract_annotation_1st_form(sub_ontology_all, gene_annotations_dir, include_electronic_annotation=False, verbose=True):
    """Function to extract annotation information from gene-annotation files (for every subontolgy)"""
    max_ann_len = {}   # maximum annotation length (for every subontolgy)
    max_ann_len_indx = {} # index of the gene with maximum annotation length (for every subontolgy)
    gene_indeces = {} # indeces of every genes (for every subontolgy)
    gene_annotations = {} # genes annotations; ie. their one-hot vector ids (for every subontolgy)
    go_term_indeces = {} # indeces of GO terms (for every subontolgy); assigned based on their higher annotation frequencies to the genes

    for sbo in sub_ontology_all:
        gene_indeces[sbo] = {}
        max_ann_len[sbo] = 0

        without_iea_genes = [] # first, for our experiments genes should have at least one IEA- annotation 
        with open("{}/gene_protein_GO_terms_without_IEA.{}".format(gene_annotations_dir, sbo)) as fr:
            for gene in [line.split()[0] for line in fr.readlines()]:
                without_iea_genes.append(gene)

        if include_electronic_annotation:
            file_reader = open("{}/gene_protein_GO_terms_with_IEA.{}".format(gene_annotations_dir, sbo))
        else:
            file_reader = open("{}/gene_protein_GO_terms_without_IEA.{}".format(gene_annotations_dir, sbo))

        index_counter = 1
        annotations = []
        for line in file_reader:
            #values = line.rstrip().replace(':', '').split()  
            values = line.rstrip().split()  
            if values[0] not in without_iea_genes: continue # making sure to experiment "without IEA" and "with IEA" we work with same data
            gene_indeces[sbo][values[0]] = index_counter
            if len(values[2:]) > max_ann_len[sbo]:
                max_ann_len[sbo] = len(values[2:])
                max_ann_len_indx[sbo] = index_counter
            annotations.append(' '.join(values[2:]))
            index_counter += 1

        gene_annotations[sbo], go_term_indeces[sbo] = tokenize_annotations(annotations)

        if verbose:
            print("Found {} annotating GO terms from {}".format(len(go_term_indeces[sbo]), sbo))
            most_freq_count = 10
            print("Top {} most frequent GO terms annotating genes in {}:".format(most_freq_count, sbo))
            for GO_ID, indx in sorted(go_term_indeces[sbo].items(), key=operator.itemgetter(1))[:most_freq_count]:
                print("  >>> {}   {}".format(GO_ID, indx))
            print("Number of annotated gene products by '{}' terms: {}".format(sbo, len(gene_annotations[sbo])))
            print("Maximum annotation length of one gene product ('{}' sub-ontology): {}".format(sbo, max_ann_len[sbo]))
            print("Index of the gene with the maximum annotations ('{}' sub-ontology): {}\n".format(sbo, max_ann_len_indx[sbo]))
        file_reader.close()
    return gene_indeces, gene_annotations, go_term_indeces, max_ann_len, max_ann_len_indx


def extract_annotation_2nd_form(sub_ontology_all, gene_annotations_dir, include_electronic_annotation=False, verbose=True):
    """Function to extract annotation information from gene-annotation files (for every subontolgy)"""
    max_ann_len = {}   # maximum annotation length (for every subontolgy)
    max_ann_len_indx = {} # index of the gene with maximum annotation length (for every subontolgy)
    gene_indeces = {} # indeces of every genes (for every subontolgy)
    gene_annotations = {} # genes annotations; ie. their one-hot vector ids (for every subontolgy)
    go_term_indeces = {} # indeces of GO terms (for every subontolgy); assigned based on their higher annotation frequencies to the genes

    for sbo in sub_ontology_all:
        gene_indeces[sbo] = {}
        max_ann_len[sbo] = 0

        without_iea_genes = [] # first, for our experiments genes should have at least one IEA- annotation 
        with open("{}/gene_protein_GO_terms_without_IEA.{}".format(gene_annotations_dir, sbo)) as fr:
            for gene in [line.split()[1] for line in fr.readlines()]:
                without_iea_genes.append(gene)

        if include_electronic_annotation:
            file_reader = open("{}/gene_protein_GO_terms_with_IEA.{}".format(gene_annotations_dir, sbo))
        else:
            file_reader = open("{}/gene_protein_GO_terms_without_IEA.{}".format(gene_annotations_dir, sbo))

        index_counter = 1
        annotations = []
        for line in file_reader:
            #values = line.rstrip().replace(':', '').split()  
            values = line.rstrip().split()  
            if values[1] not in without_iea_genes: continue # making sure to experiment "without IEA" and "with IEA" we work with same data
            gene_indeces[sbo][values[1]] = index_counter
            if len(values[2:]) > max_ann_len[sbo]:
                max_ann_len[sbo] = len(values[2:])
                max_ann_len_indx[sbo] = index_counter
            annotations.append(' '.join(values[2:]))
            index_counter += 1

        gene_annotations[sbo], go_term_indeces[sbo] = tokenize_annotations(annotations)

        if verbose:
            print("Found {} annotating GO terms from {}".format(len(go_term_indeces[sbo]), sbo))
            most_freq_count = 10
            print("Top {} most frequent GO terms annotating genes in {}:".format(most_freq_count, sbo))
            for GO_ID, indx in sorted(go_term_indeces[sbo].items(), key=operator.itemgetter(1))[:most_freq_count]:
                print("  >>> {}   {}".format(GO_ID, indx))
            print("Number of annotated gene products by '{}' terms: {}".format(sbo, len(gene_annotations[sbo])))
            print("Maximum annotation length of one gene product ('{}' sub-ontology): {}".format(sbo, max_ann_len[sbo]))
            print("Index of the gene with the maximum annotations ('{}' sub-ontology): {}\n".format(sbo, max_ann_len_indx[sbo]))
        file_reader.close()
    return gene_indeces, gene_annotations, go_term_indeces, max_ann_len, max_ann_len_indx

def make_reproducible(seed):
    """makes sure the results are going to be as reproducible as possible (on GPUs might notice slight differences) for a fixed seed"""
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
    sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=session_conf)
    tf.compat.v1.keras.backend.set_session(sess)