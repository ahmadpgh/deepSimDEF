import os
import operator


def save_simDEF_model(FOLD, models):

    project_path = os.getcwd() + "/model_repository/model_weights/"

    if not os.path.exists(project_path):
        os.makedirs(project_path)

    for i in range(FOLD):
        print("Saving model " + str(i+1) + " to disk ...")
        model_json = models[i].to_json()
        with open(project_path + "model_PPI_" + str(i+1) + ".json", "w") as json_file:
            json_file.write(model_json)
            models[i].save_weights(project_path + "model_PPI_" + str(i+1) + ".h5")
        print "The Model and its Weights Are Saved!!\n"
                
                
def save_simDEF_embeddings(FOLD, embedding_layers, word_indeces, SUB_ONTOLOGY_work):
    
    for ind in range(FOLD):
        
        project_path = os.getcwd() + "/model_repository/model_embeddings/"

        if not os.path.exists(project_path):
            os.makedirs(project_path)

        for sbo in SUB_ONTOLOGY_work:
            embeddings = embedding_layers[ind][sbo].get_weights()
            index =  []
            for i, j in sorted(word_indeces[sbo].items(), key=operator.itemgetter(1)):
                index.append(i)

            file_writer = open(project_path + "EMB_PPI_"+ sbo + "_" + str(ind + 1), "w")
            for i in range(len(index)):
                file_writer.write((index[i] + " ").replace("\r", "\\r"))
                for j in range(embeddings[0].shape[1]):
                    if j == (embeddings[0].shape[1] - 1):
                        file_writer.write(str(embeddings[0][i + 1][j]))
                    else:
                        file_writer.write(str(embeddings[0][i + 1][j]) + " ")
                file_writer.write("\n")
            file_writer.close()
            
    print "The Word Embeddings Are Saved!!"  
