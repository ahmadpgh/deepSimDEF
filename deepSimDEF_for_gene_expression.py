# CUDA_VISIBLE_DEVICES=3 python deepSimDEF_for_gene_expression.py

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
import sys
import logging
import random
import operator
import numpy as np
import math
import pprint
import argparse
import tensorflow.keras
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

from networks import deepSimDEF_network
from datasets import gene_expression_dataset
from dataloaders import generic_dataloader
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

import socket
import datetime
from pytz import timezone

import collections
from collections import OrderedDict
from utils import *

tz = timezone('US/Eastern')  # To monitor training time (showing start & end points of a fixed timezone when the code runs on a remote server)
pp = pprint.PrettyPrinter(indent=4)

#checkpoint = '[base_dir]/2020.03.04-23h40m37s_server_name/model_checkpoints/epoch_58'
checkpoint = None

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--nb_fold', default=2, type=int, help='number of folds of training and evaluation in n-fold cross-validation (default: 10)')
parser.add_argument('--dropout', default=0.3, type=float, help='dropout applied to dense layers of the network (default: 0.3)')
parser.add_argument('--embedding_dropout', default=0.15, type=float, help='dropout applied to embedding layers of the network; i.e., percentage of features dropped out completely (default: 0.15)')
parser.add_argument('--annotation_dropout', default=0.0, type=float, help='dropout applied to annotations of a gene at training time; i.e., percentage of annotations ignored (default: 0.0)')
parser.add_argument('--pretrained_embedding', default=True, type=bool, help='whether the GO term embeddings loaded should be computed in advance from a pretrained unsupervised model (default: True)')
parser.add_argument('--updatable_embedding', default=True, type=bool, help='whether the GO term embeddings should be updatable during the traning (default: True)')
parser.add_argument('--activation_hidden', default='relu', type=str, help='activation function of hidden layers (default: "relu")')
parser.add_argument('--activation_highway', default='relu', type=str, help='activation function of highway layer (default: "relu")')
parser.add_argument('--activation_output', default='linear', type=str, help='activation function of last, i.e. output, layer (default: "linear")')
parser.add_argument('--embedding_dim', default=100, type=int, help='dimentionality of GO term embeddings, i.e. number of latent features (default: 100)')
parser.add_argument('--nb_epoch', default=2, type=int, help='number of epochs for training')
parser.add_argument('--batch_size', default=256, type=int, help='batch size (default: 256)')
parser.add_argument('--loss', default='mean_squared_error', type=str, help='loss type of the onjective function that gets optimized ("binary_crossentropy" or "mean_squared_error")')
parser.add_argument('--optimizer', default='adam', type=str, help='optimizer algorithm, can be: "adam", "rmsprop", etc. (default: "adam")')
parser.add_argument('--learning_rate', default=0.001, type=float, help='starting learning rate for optimization')
parser.add_argument('--iea', default=True, type=bool, help='whether to consider "inferred from electronic annotations" or not')
parser.add_argument('--checkpoint', default=checkpoint, help='starting from scratch or using model checkpoints')
parser.add_argument('--save_model', default=False, type=bool, help='model checkpointing, whether to save the models during training')
parser.add_argument('--save_embeddings', default=False, type=bool, help='storing weights of the embedding layers, whether to save updated embeddings')
parser.add_argument('--save_interval', default=5, type=int, help='-1, checkpoint if see improvement in the result; otherwise after each interval (default: -1)')
parser.add_argument('--sub_ontology', default='all', type=str, help='considering annotations of what subontologies, "bp", "cc", "mf", or "all" (default: "all")')
parser.add_argument('--verbose', default=False, type=bool, help='if print extra information during model training')
parser.add_argument('--inpute_file', default='default', type=str, help='inpute file of the gene product terms and the score(s), if not provide use default file')
parser.add_argument('--log_dir', default='log/', type=str, help='base log folder (will be created if it does not exist)')
parser.add_argument('--log_name', default='GE_test', type=str, help='prefix name to use when logging this model')
parser.add_argument('--reproducible', default=True, type=bool, help='whether we want to have a reproducible result (mostly helpful with training on a CPU at the cost of training speed)')
parser.add_argument('--seed', default=313, type=int, help='seed used for Random Number Generation if "reproducible=True"')
parser.add_argument('--experiment_mode', default=2, type=int, help='1: any pairs of unseen genes; 2: only pair in which both genes are unseen')
parser.add_argument('--partial_shuffle_percent', default=0.0, type=float, help='Should be more than 0.0 for "Negative Control" experiments (default: 0.0)')
parser.add_argument('--highway_layer', default=True, type=bool, help='True if highway layer instead of cosince similarity (default: True)')
parser.add_argument('--cosine_similarity', default=False, type=bool, help='True cosince similarity instead of highway layer (default: False)')
parser.add_argument('--species', default='yeast', type=str, help='the species of interest for evaluation (human, yeast, etc)')
parser.add_argument('--adaptive_lr', default=True, type=bool, help='whether to use adavtive learning rate or not')
parser.add_argument('--adaptive_lr_rate', default=10, type=int, help='after how many epoch, decay the learning rate')


def fit_gene_expression(models, args):
    best_epoch_pearson, best_epoch_spearman = 0, 0
    final_pearson, final_spearman = [], []

    start_time = datetime.datetime.now(tz)
    former_iteration_endpoint = start_time
    print("~~~~~~~~~~~~~ TIME ~~~~~~~~~~~~~~")
    print("Time started: {}".format(start_time.strftime("%Y-%m-%d %H:%M:%S")))

    """Training loop"""
    for e in range(checkpoint_baseline, args.nb_epoch):
        print("~~~~~~~~ {} ({}) ~~~~~~~~ EPOCH {}/{} (Embedding dimention: {}) ~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n".format(
                '/'.join(sub_ontology_interested), args.species, e+1, args.nb_epoch, args.embedding_dim))
        if args.adaptive_lr:
            learning_rate = exp_decay(epoch=e//args.adaptive_lr_rate, initial_lrate=args.learning_rate) # claculating the desired learning rate using the exponential decay formula
        else:
            learning_rate = args.learning_rate
        
        epoch_pearsons, epoch_spearmans = [], []
        for model_index in range(len(models)): # Going through each model one by one
            # Preparing the data
            train_pair, X_train, y_train, test_pair, X_test, y_test, train_gene, test_gene = generic_dataloader(
                model_index, nb_test_genes, gene_shuffled_indx, gene_1, gene_2, fully_annotated_genes,
                gene_1_annotation, gene_2_annotation, prediction_value, sub_ontology_interested,
                args.experiment_mode)
            if e==0: save_gene_pairs(logdir=logdir, model_id=model_index, train_pair=train_pair, 
                                     test_pair=test_pair, train_gene=train_gene, test_gene=test_gene)
            if args.nb_fold==1: X_train, y_train = X_test, y_test # If single model, we need to redefine the training data due to absence of folds
            # Training and Prediction
            model = models[model_index]
            model.optimizer.lr = learning_rate # decreasing a model's learning rate already calculated by exponential decay formula
            history = model.fit(X_train, y_train, batch_size=args.batch_size, epochs=1, shuffle=True)
            if args.nb_fold!=1: # Only evaluation and report in n-fold cross-validation set up
                predictions = model.predict(X_test)
                """Pearson"""
                #finding the best pearson
                pr = np.round(pearsonr(y_test.reshape(y_test.shape[0]), predictions.reshape(predictions.shape[0]))[0], 5)
                epoch_pearsons.append(pr)
                # Keeping track of improving or receding pearson
                best_pearson = best_pearsons[model_index]
                if best_pearson < pr: 
                    best_pearsons[model_index] = pr
                    st = "(+){}".format(best_pearsons[model_index])
                else: 
                    st = "(-){}".format(best_pearsons[model_index])
                print(">>> Pearson ({}): {} Best({}): {} ({})".format(
                    model_index+1, pr, model_index+1, st, np.round(pr-best_pearson, 5)))
                """Spearman"""
                #finding the best spearman
                sp = np.round(spearmanr(y_test.reshape(y_test.shape[0]), predictions.reshape(predictions.shape[0]))[0], 5)
                epoch_spearmans.append(sp)
                # Keeping track of improving or receding spearman
                best_spearman = best_spearmans[model_index]
                if best_spearman < sp: 
                    best_spearmans[model_index] = sp
                    st = "(+){}".format(best_spearmans[model_index])
                else: 
                    st = "(-){}".format(best_spearmans[model_index])
                print(">>> Spearman ({}): {} Best({}): {} ({})\n".format(
                    model_index+1, sp, model_index+1, st, np.round(sp-best_spearman, 5)))
                # loging the model results
                log_model_result_for_gene_expression(e+1, model_index+1, learning_rate, 
                                     best_pearsons[model_index], pr-best_pearson, pr, 
                                     best_spearmans[model_index], sp-best_spearman, sp, logdir)
        if args.nb_fold!=1: # Stats on all folds in this epoch
            """Pearson"""
            pr_res = np.mean(epoch_pearsons) # best pearson for all models
            final_pearson.append(pr_res)
            # Stats on what we have done so far from the begining of training
            if e==checkpoint_baseline:
                best_epoch_pearson = e+1
                best_cv_pearson = final_pearson[0]
            else:
                for epoch, final_result in enumerate(final_pearson):
                    if best_cv_pearson < final_pearson[epoch]:
                        best_epoch_pearson = checkpoint_baseline+epoch+1
                        best_cv_pearson = final_pearson[epoch]
            print(" Pearson for this epoch: {:.2f}% -- Best Pearson::==> {:.2f}% (for epoch #{} of {})".format(
            pr_res*100, best_cv_pearson*100, best_epoch_pearson, args.nb_epoch))
            """Spearman"""
            sp_res = np.mean(epoch_spearmans) # best spearman for all models
            final_spearman.append(sp_res)
            # Stats on what we have done so far from the begining of training
            if e==checkpoint_baseline:
                best_epoch_spearman = e+1
                best_cv_spearman = final_spearman[0]
            else:
                for epoch, final_result in enumerate(final_spearman):
                    if best_cv_spearman < final_spearman[epoch]:
                        best_epoch_spearman = checkpoint_baseline+epoch+1
                        best_cv_spearman = final_spearman[epoch]
            print(" Spearman for this epoch: {:.2f}% -- Best Spearman::==> {:.2f}% (for epoch #{} of {})\n".format(
            sp_res*100, best_cv_spearman*100, best_epoch_spearman, args.nb_epoch))
        # save models and embeddings
        if args.save_interval==-1 and best_epoch_spearman==e+1: # save if improved the result
            if args.save_model: # save models
                save_model(path=logdir, models=models, epoch=e+1, verbose=args.verbose)
            if args.save_embeddings: # save (updated) GO term embeddings
                save_embeddings(path=logdir,
                                models=models,
                                go_term_indeces=go_term_indeces, 
                                sub_ontology_interested=sub_ontology_interested,
                                embedding_save=go_term_embedding_save_in,
                                epoch=e+1,
                                verbose=args.verbose)
        elif args.save_interval!=-1 and (e+1)%args.save_interval==0: # save after each interval
            if args.save_model: # save models
                save_model(path=logdir, models=models, epoch=e+1, verbose=args.verbose)
            if args.save_embeddings: # save (updated) GO term embeddings
                save_embeddings(path=logdir,
                                models=models,
                                go_term_indeces=go_term_indeces, 
                                sub_ontology_interested=sub_ontology_interested,
                                embedding_save=go_term_embedding_save_in,
                                epoch=e+1,
                                verbose=args.verbose)
        # Calculating 'Computation Time' for this round of iteration
        former_iteration_endpoint, current_iteration_elapsed = cal_iter_time(former_iteration_endpoint, e, args, tz)
        # loging the epoch results
        if args.nb_fold!=1: # Stats on all folds in this epoch
            log_epoch_result_for_gene_expression(e+1, args.nb_epoch, current_iteration_elapsed, 
                             best_cv_pearson, pr_res, best_epoch_pearson, 
                             best_cv_spearman, sp_res, best_epoch_spearman, logdir)


if __name__ == "__main__":

    args = parser.parse_args()

    # some assertions before proceeding
    assert not ((args.nb_fold == 1) and (args.save_interval == -1)), "'--save_interval' cann't be -1  when '--nb_fold' is 1; define a 'positive integer' interval"
    assert not (args.highway_layer and args.cosine_similarity), "Either '--highway_layer' can be True or '--cosine_similarity', not both."

    # printing out the argument of the model
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print("Arguments are:")
    pp.pprint(vars(args))
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n")

    # directory to the files needed for traning and testing
    if args.inpute_file=='default':
        data_file_name = f'{args.species}_gene_expression.tsv'
    else:
        data_file_name = args.inpute_file
    gene_expression_data_dir = './data/species/{}/gene_expression/processed'.format(args.species) # directory to the ppi datasets, pay attention to the file names and thier content format
    embedding_dir = f'./data/gene_ontology/definition_embedding/{args.embedding_dim}_dimensional' # directory to the GO term embeddings, pay attention to the file names and thier content format
    gene_annotations_dir = './data/species/{}/association_file/processed'.format(args.species) # directory to the gene annotations, pay attention to the file names and thier content format

    # set RNG
    if args.reproducible: make_reproducible(args.seed)

    # some variables needed to work with sub-ontologies
    if args.sub_ontology=='all': sub_ontology_interested = ['BP', 'CC', 'MF']
    elif args.sub_ontology=='bp': sub_ontology_interested = ['BP']
    elif args.sub_ontology=='cc': sub_ontology_interested = ['CC']
    elif args.sub_ontology=='mf': sub_ontology_interested = ['MF']
    sub_ontology_all = ['BP', 'CC', 'MF'] # for experimets, to make sure all genes have annotations from all three ontologies

    # do we use a checkpointed model, if not we can set it
    if args.checkpoint is None:
        args.log_name = f"{args.log_name}_{args.species}/pretrained_emb_{args.pretrained_embedding}_iea_{args.iea}_ontology_{args.sub_ontology}"
        logdir, checkpoint_baseline = log(args)
    else:
        logdir = args.checkpoint.rsplit("/", 2)[0]
        checkpoint_baseline = int(args.checkpoint.rsplit("_")[-1])
        args.log_name = checkpoint.rsplit("/", 3)[0]

    print(f"The checkpoint directory is: '{logdir}'\n")

    # some variables to work with GO-term embeddings later
    go_term_embedding_file_path = {}  # directory of embedding files (for every subontolgy)
    go_term_embedding_save_in = {}  # files into which the updated GO term embeddings will be stored
    for sbo in sub_ontology_interested:
        go_term_embedding_file_path[sbo] = '{}/GO_{}_Embeddings_{}D.emb'.format(embedding_dir, sbo, args.embedding_dim)
        go_term_embedding_save_in[sbo] = 'GO_{}_Embeddings_{}D_Updated'.format(sbo, args.embedding_dim)
        
    # getting GO annotations
    gene_indeces, gene_annotations, go_term_indeces, max_ann_len, max_ann_len_indx = extract_annotation_1st_form(
        sub_ontology_all, gene_annotations_dir, args.iea, args.verbose)

    fully_annotated_genes = []  # we keep only those genes for which we have annatoation from all sub-ontologies
    for sbo in sub_ontology_all:
        fully_annotated_genes.append(gene_indeces[sbo].keys())
    fully_annotated_genes = sorted(list(set(fully_annotated_genes[0]).intersection(*fully_annotated_genes)))

    if args.verbose: print("Number of fully annotated gene products: {}\n".format(len(fully_annotated_genes)))

    """Shuffling the genes"""
    gene_shuffled_indx = np.arange(len(fully_annotated_genes))
    np.random.shuffle(gene_shuffled_indx)
    VALIDATION_SPLIT = 1.0/args.nb_fold
    nb_test_genes = int(VALIDATION_SPLIT * len(fully_annotated_genes))

    gene_1, gene_2, gene_1_annotation, gene_2_annotation, prediction_value = gene_expression_dataset(
        gene_expression_data_dir, data_file_name, sub_ontology_all, fully_annotated_genes, gene_annotations,
        gene_indeces, max_ann_len, args.partial_shuffle_percent, sub_ontology_interested)

    VALIDATION_SPLIT = 1.0/args.nb_fold
    gene_pair_indx = np.arange(gene_1_annotation[sub_ontology_interested[0]].shape[0])
    np.random.shuffle(gene_pair_indx)
    nb_test_gene_pairs = int(VALIDATION_SPLIT * gene_1_annotation[sub_ontology_interested[0]].shape[0])

    if args.verbose:
        for sbo in sub_ontology_interested:
            print("Shape of data tensor for gene 1 ({}): {}".format(sbo, gene_1_annotation[sbo].shape))
            print("Shape of data tensor for gene 2 ({}): {}\n".format(sbo, gene_2_annotation[sbo].shape))
        print("Shape of output sequence homology tensors: {}\n".format(prediction_value.shape))

    models = []
    best_pearsons = []
    best_spearmans = []
    for m in range(args.nb_fold):
        network = deepSimDEF_network(
            embedding_dim=args.embedding_dim, 
            model_ind=m, 
            max_ann_len=max_ann_len, 
            go_term_embedding_file_path=go_term_embedding_file_path,
            sub_ontology_interested=sub_ontology_interested,
            go_term_indeces=go_term_indeces, 
            activation_hidden=args.activation_hidden, 
            activation_highway=args.activation_highway, 
            activation_output=args.activation_output, 
            dropout=args.dropout,
            embedding_dropout=args.embedding_dropout,
            annotation_dropout=args.annotation_dropout,
            pretrained_embedding=args.pretrained_embedding,
            updatable_embedding=args.updatable_embedding,
            loss=args.loss,
            optimizer=args.optimizer,
            learning_rate=args.learning_rate,
            checkpoint=args.checkpoint,
            verbose=args.verbose,
            highway_layer=args.highway_layer,
            cosine_similarity=args.cosine_similarity
        )
        models.append(network)
        best_pearsons.append(0)
        best_spearmans.append(0)

    fit_gene_expression(models, args)

    save_model(path=logdir, models=models, epoch=args.nb_epoch, verbose=args.verbose)
        
    save_embeddings(path=logdir,
        models=models,
        go_term_indeces=go_term_indeces,
        sub_ontology_interested=sub_ontology_interested,
        embedding_save=go_term_embedding_save_in,
        epoch=args.nb_epoch,
        verbose=args.verbose)