import numpy as np
from keras.preprocessing.sequence import pad_sequences


def gene_pair_data_reader(data_dir,
                          SUB_ONTOLOGY_work,
                          fully_annotated_sequences,
                          sequences,
                          protein_index,
                          MAX_SEQUENCE_LENGTH):
    annotation_G1 = {}
    annotation_G2 = {}
    interaction_pr = []

    for sbo in SUB_ONTOLOGY_work:
        annotation_G1[sbo] = []
        annotation_G2[sbo] = []

        X1_t = []
        X2_t = []
        y_t = []
        file_reader = open(data_dir)

        header_flag = True
        for line in file_reader:
            if header_flag:
                header_flag = False
                continue
            values = line.split()
            if ((not values[0] in fully_annotated_sequences) or (not values[1] in fully_annotated_sequences)):
                continue
            X1_t.append(sequences[sbo][protein_index[sbo][values[0]] - 1])
            X2_t.append(sequences[sbo][protein_index[sbo][values[1]] - 1])
            y_t.append(float(values[2]))

        interaction_pr = np.asarray(y_t)

        annotation_G1[sbo] = pad_sequences(X1_t, maxlen=MAX_SEQUENCE_LENGTH[sbo])
        annotation_G2[sbo] = pad_sequences(X2_t, maxlen=MAX_SEQUENCE_LENGTH[sbo])
        print 'Shape of data tensor 1 (' + sbo + '):', annotation_G1[sbo].shape
        print 'Shape of data tensor 2 (' + sbo + '):', annotation_G2[sbo].shape
        print 'Shape of similarity tensor (' + sbo + '):', interaction_pr.shape, "\n"

        file_reader.close()

    print 'Number of positive classes/interactions:', int(np.sum(interaction_pr))

    return annotation_G1, annotation_G2, interaction_pr


def input_data_maker(model_id,
                     test_size,
                     indices,
                     annotation_G1_dic_MC,
                     annotation_G2_dic_MC,
                     interaction_pr_list_MC,
                     annotation_G1_dic_HT,
                     annotation_G2_dic_HT,
                     interaction_pr_list_HT,
                     WITH_HIGH_THROUPUT,
                     SUB_ONTOLOGY_work):
    test_indices = indices[model_id * (test_size):(model_id + 1) * (test_size)]
    train_indices = list(set(indices) - set(test_indices))

    X_train_G1 = []
    X_train_G2 = []

    X_test_G1 = []
    X_test_G2 = []

    for sbo in SUB_ONTOLOGY_work:
        if (WITH_HIGH_THROUPUT):
            X_train_G1_t = np.concatenate(
                (annotation_G1_dic_MC[sbo][train_indices], annotation_G2_dic_MC[sbo][train_indices],
                 annotation_G1_dic_HT[sbo], annotation_G2_dic_HT[sbo]), axis=0)
            X_train_G2_t = np.concatenate(
                (annotation_G2_dic_MC[sbo][train_indices], annotation_G1_dic_MC[sbo][train_indices],
                 annotation_G2_dic_HT[sbo], annotation_G1_dic_HT[sbo]), axis=0)
        else:
            X_train_G1_t = np.concatenate(
                (annotation_G1_dic_MC[sbo][train_indices], annotation_G2_dic_MC[sbo][train_indices]), axis=0)
            X_train_G2_t = np.concatenate(
                (annotation_G2_dic_MC[sbo][train_indices], annotation_G1_dic_MC[sbo][train_indices]), axis=0)

        X_train_G1.append(X_train_G1_t)
        X_train_G2.append(X_train_G2_t)

        X_test_G1_t = annotation_G1_dic_MC[sbo][test_indices]
        X_test_G2_t = annotation_G2_dic_MC[sbo][test_indices]
        X_test_G1.append(X_test_G1_t)
        X_test_G2.append(X_test_G2_t)

    if (WITH_HIGH_THROUPUT):
        y_train = np.concatenate((interaction_pr_list_MC[train_indices], interaction_pr_list_MC[train_indices],
                                  interaction_pr_list_HT, interaction_pr_list_HT))
    else:
        y_train = np.concatenate((interaction_pr_list_MC[train_indices], interaction_pr_list_MC[train_indices]))
    y_test = interaction_pr_list_MC[test_indices]

    return X_train_G1 + X_train_G2, y_train, X_test_G1 + X_test_G2, y_test

