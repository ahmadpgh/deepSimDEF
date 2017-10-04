import numpy as np
import keras.backend as K

from keras.layers import Input, Embedding, AveragePooling1D, MaxPooling1D, Flatten, Dense, Dropout, Merge, Highway, Activation, Reshape
from keras.layers.merge import Concatenate
from keras.layers.noise import GaussianDropout, GaussianNoise
from keras.models import Model, model_from_json


def precision(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def recall(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def fmeasure(y_true, y_pred, beta=1):
    if beta < 0:
        raise ValueError('The lowest choosable beta is zero (only precision).')

    # If there are no true positives, fix the F score at 0 like sklearn.
    if K.sum(K.round(K.clip(y_true, 0, 1))) == 0:
        return 0

    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    bb = beta ** 2
    fbeta_score = (1 + bb) * (p * r) / (bb * p + r + K.epsilon())
    return fbeta_score


def deepSimDEF_PPI(EMBEDDING_DIM, 
                   model_ind, 
                   MAX_SEQUENCE_LENGTH, 
                   WORD_EMBEDDINGS, 
                   SUB_ONTOLOGY_work,
                   word_indeces, 
                   ACTIVATION_HIDDEN, 
                   ACTIVATION_HIGHWAY, 
                   ACTIVATION_OUTPUT,
                   DROPOUT,
                   OPTIMIZER,
                   TRANSFER_LEARNING=False,
                   PRE_TRAINED=True,
                   UPDATABLE=True,
                   PRINT_deepSimDEF_SUMMARY=False):
    
    EMBEDDINGS = {}
    INPUTS = []
    DENSES = []
    CHANNELS = []
    CHANNELS2 = []

    Dense1_weights = []
    if TRANSFER_LEARNING:
        # load json and create model
        #json_file = open('model_repository/model_PPI_' + str(ind) + '.json', 'r')
        json_file = open('model_repository/model_0.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        # load weights into new model
        #loaded_model.load_weights('model_repository/model_PPI_' + str(ind) + '.h5')
        loaded_model.load_weights('model_repository/model_0.h5')
        #Dense1_weights = loaded_model.get_layer('gene_product_dense').get_weights()
        print("Loaded model from disk")
        model = loaded_model
        model.compile(loss = 'binary_crossentropy', optimizer = OPTIMIZER, metrics=[fmeasure])
        return model, 0

    for i in range(2):
        for sbo in SUB_ONTOLOGY_work:
            
            protein_input = Input(shape=(MAX_SEQUENCE_LENGTH[sbo],), dtype='int32')
            INPUTS.append(protein_input)
            
            if sbo in EMBEDDINGS:
                embedding_layer = EMBEDDINGS[sbo]
            else:
                if PRE_TRAINED:
                    # with using pre-trained word embedings
                    file_reader = open(WORD_EMBEDDINGS[sbo])
                    word_embeddings = {}
                    for line in file_reader:
                        values = line.split()
                        word = values[0]
                        vector = np.asarray(values[1:], dtype='float32')
                        word_embeddings[word] = vector
                    file_reader.close()

                    print 'Loaded', len(word_embeddings), 'word vectors for', sbo, '(Model ' + str(model_ind + 1) + ')'

                    embedding_size = len(word_embeddings[np.random.choice(word_embeddings.keys())])
                    embedding_matrix = np.zeros((len(word_indeces[sbo]) + 1, embedding_size)) - 300.0
                    for word, i in word_indeces[sbo].items():
                        embedding_vector = word_embeddings.get(word)
                        if embedding_vector is not None:
                            # words not found in embedding index will be all-zeros.
                            embedding_matrix[i] = embedding_vector

                    embedding_layer = Embedding(input_dim=len(word_indeces[sbo]) + 1, 
                                                  output_dim=embedding_size, 
                                                  weights=[embedding_matrix],
                                                  input_length=MAX_SEQUENCE_LENGTH[sbo], 
                                                  trainable=UPDATABLE)
                else:
                    # without using pre-trained word embedings
                    embedding_layer = Embedding(input_dim=len(word_indeces[sbo]) + 1, 
                                                  output_dim=EMBEDDING_DIM, 
                                                  input_length=MAX_SEQUENCE_LENGTH[sbo])

                EMBEDDINGS[sbo] = embedding_layer

            #protein_input = Input(shape=(MAX_SEQUENCE_LENGTH[sbo],), dtype='int32')
            #INPUTS.append(protein_input)

            GO_term = embedding_layer(protein_input)

            Ch = MaxPooling1D(pool_size=MAX_SEQUENCE_LENGTH[sbo])(GO_term)
            Ch = Flatten()(Ch)
            CHANNELS.append(Ch)
    
    num_pair = 2    
    for i in range(num_pair):
        #for j in range(len(CHANNELS)/2):
        if len(SUB_ONTOLOGY_work) > 1:
            Mrg = Concatenate(axis=-1)(CHANNELS[i*len(SUB_ONTOLOGY_work):len(SUB_ONTOLOGY_work)*(i+1)])
        else:
            Mrg = CHANNELS[i]

        if len(DENSES) == 1:
            Dns = DENSES[0]
        else:
            Dns = Dense(units = EMBEDDING_DIM * len(SUB_ONTOLOGY_work), activation = ACTIVATION_HIDDEN)
            #Dns = Dense(units = EMBEDDING_DIM * len(SUB_ONTOLOGY_work), activation = ACTIVATION_HIDDEN, name='gene_product_dense',weights=Dense1_weights, trainable=UPDATABLE)
            DENSES.append(Dns)
        Ch = Dns(Mrg)

        DrpOut = Dropout(DROPOUT)
        Ch = DrpOut(Ch)

        CHANNELS2.append(Ch)
    
    merge = Concatenate(axis=-1)(CHANNELS2)
    merge = Highway(activation = ACTIVATION_HIGHWAY, name="highway_layer")(merge)
    merge = Dropout(DROPOUT)(merge)

    merge = Dense(units = EMBEDDING_DIM * len(SUB_ONTOLOGY_work), 
                  activation = ACTIVATION_HIDDEN)(merge)
    merge = Dropout(DROPOUT)(merge)

    preds = Dense(units = 1, activation = ACTIVATION_OUTPUT)(merge)

    model = Model(inputs = INPUTS, outputs = preds)

    model.compile(loss = 'binary_crossentropy', optimizer = OPTIMIZER, metrics=[fmeasure])
    
    if PRINT_deepSimDEF_SUMMARY:
        print model.summary()
        
    print "Model for Fold Number", model_ind + 1, "Instantiated!!\n"
    
    return model, EMBEDDINGS
