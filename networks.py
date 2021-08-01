import sys
import logging
import numpy as np
import tensorflow.keras

from tensorflow.keras import initializers
from tensorflow.python.keras.layers import Input, Embedding, AveragePooling1D, MaxPooling1D, Flatten, Dense, Dropout
from tensorflow.python.keras.layers import Activation, Reshape, Multiply, Add, Lambda, SpatialDropout1D, InputSpec, Dot
from tensorflow.python.keras.layers.merge import Concatenate
from tensorflow.keras.models import Model, model_from_json
from tensorflow.compat.v1.keras import backend as K

from utils import *

def load_embedding(emb_dir, embedding_dim, go_term_indeces):
    # with using pre-trained word embedings
    file_reader = open(emb_dir)
    word_embeddings = {}
    
    for line in file_reader:
        values = line.split()
        word = values[0]
        vector = np.asarray(values[1:], dtype='float32')
        word_embeddings[word] = vector
    file_reader.close()
    
    embedding_matrix = np.zeros((len(go_term_indeces) + 1, embedding_dim)) - 300.0
    for word, i in go_term_indeces.items():
        embedding_vector = word_embeddings.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector
    return embedding_matrix

def highway(value, activation="tanh", transform_gate_bias=-1.0):
    dim = K.int_shape(value)[-1]
    transform_gate_bias_initializer = initializers.Constant(transform_gate_bias)
    transform_gate = Dense(units=dim, bias_initializer=transform_gate_bias_initializer)(value)
    transform_gate = Activation("sigmoid")(transform_gate)
    carry_gate = Lambda(lambda x: 1.0 - x, output_shape=(dim,))(transform_gate)
    transformed_data = Dense(units=dim)(value)
    transformed_data = Activation(activation)(transformed_data)
    transformed_gated = Multiply()([transform_gate, transformed_data])
    identity_gated = Multiply()([carry_gate, value])
    value = Add()([transformed_gated, identity_gated])
    return value

def deepSimDEF_network(args, model_ind, max_ann_len=None, go_term_embedding_file_path=None, sub_ontology_interested=None, go_term_indeces=None, model_summary=False):

    embedding_dim = args.embedding_dim
    activation_hidden = args.activation_hidden
    activation_highway = args.activation_highway
    activation_output = args.activation_output
    dropout = args.dropout
    embedding_dropout = args.embedding_dropout
    annotation_dropout = args.annotation_dropout
    pretrained_embedding = args.pretrained_embedding
    updatable_embedding = args.updatable_embedding
    loss = args.loss
    optimizer = args.optimizer
    learning_rate = args.learning_rate
    checkpoint = args.checkpoint
    verbose = args.verbose
    highway_layer = args.highway_layer
    cosine_similarity = args.cosine_similarity
    deepsimdef_mode = args.deepsimdef_mode
    
    _inputs = [] # used to represent the input data to the network (from different channels)
    _embeddings = {} # used for weight-sharing of the embeddings
    _denses = [] # used for weight-sharing of dense layers whenever needed
    _Gene_channel = [] # for the middle part up-until highway

    if checkpoint:
        with open('{}/model_{}.json'.format(checkpoint, model_ind+1), 'r') as json_file:
            model = model_from_json(json_file.read()) # load the json model
            model.load_weights('{}/model_{}.h5'.format(checkpoint, model_ind+1)) # load weights into new model
            if deepsimdef_mode=='training':
                model.compile(loss=loss, optimizer=optimizer)
            if verbose: print("Loaded model {} from disk".format(model_ind+1))
            return model

    for i in range(2): # bottom-half of the network, 2 for 2 channels

        _GO_term_channel = [] # for bottom-half until flattening maxpooled embeddings

        for sbo in sub_ontology_interested:

            _inputs.append(Input(shape=(max_ann_len[sbo],), dtype='int32'))

            if sbo in _embeddings: 
                embedding_layer = _embeddings[sbo] # for the second pair when we need weight-sharing
            else:
                if pretrained_embedding:
                    embedding_matrix = load_embedding(go_term_embedding_file_path[sbo], embedding_dim, go_term_indeces[sbo])
                    if verbose: print("Loaded {} word vectors for {} (Model {})".format(len(embedding_matrix), sbo, model_ind+1))
                    embedding_layer = Embedding(input_dim=len(go_term_indeces[sbo])+1, output_dim=embedding_dim,
                                                weights=[embedding_matrix], input_length=max_ann_len[sbo],
                                                trainable=updatable_embedding, name="embedding_{}_{}".format(sbo, model_ind))
                else: # without using pre-trained word embedings
                    embedding_layer = Embedding(input_dim=len(go_term_indeces[sbo])+1, output_dim=embedding_dim,
                                                input_length=max_ann_len[sbo], name="embedding_{}_{}".format(sbo, model_ind))
                _embeddings[sbo] = embedding_layer

            GO_term_emb = embedding_layer(_inputs[-1])

            if 0 < annotation_dropout: GO_term_emb = DropAnnotation(annotation_dropout)(GO_term_emb)
            if 0 < embedding_dropout: GO_term_emb = SpatialDropout1D(embedding_dropout)(GO_term_emb)

            GO_term_emb = MaxPooling1D(pool_size=max_ann_len[sbo])(GO_term_emb)
            GO_term_emb = Flatten()(GO_term_emb)
            _GO_term_channel.append(GO_term_emb)

        Gene_emb = Concatenate(axis=-1)(_GO_term_channel) if 1 < len(sub_ontology_interested) else _GO_term_channel[0]
        Dns = _denses[0] if len(_denses) == 1 else Dense(units=embedding_dim*len(sub_ontology_interested), activation=activation_hidden)
        _denses.append(Dns)

        Gene_emb = Dns(Gene_emb)
        Gene_emb = Dropout(dropout)(Gene_emb)
        _Gene_channel.append(Gene_emb)

    if cosine_similarity:
        preds = Dot(axes=1, normalize=True)(_Gene_channel)
    else:
        merge = Concatenate(axis=-1)(_Gene_channel)
        if highway_layer:
            merge = highway(merge, activation=activation_highway)
            merge = Dropout(dropout)(merge)
        merge = Dense(units=embedding_dim*len(sub_ontology_interested), activation=activation_hidden)(merge)
        merge = Dropout(dropout)(merge)
        preds = Dense(units=1, activation=activation_output)(merge)

    model = Model(inputs=_inputs, outputs=preds)

    model.compile(loss=loss, optimizer=optimizer)

    model.optimizer.lr = learning_rate # setting the learning rate of the model optimizer

    if model_summary: print(model.summary())

    if verbose: print("Model for fold number {} instantiated!!\n".format(model_ind+1))

    return model