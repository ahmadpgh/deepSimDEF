![Language](https://img.shields.io/badge/language-Python-blue.svg)
![Stars](https://img.shields.io/github/stars/ahmadpgh/deepSimDEF?color=r)
![Repo Size](https://img.shields.io/github/repo-size/ahmadpgh/deepSimDEF?color=tomato)

<br>
<img align="left" src="imgs/dna_logo.png" width="110"> 

## deepSimDEF: deep neural embeddings of gene products and Gene Ontology terms for functional analysis of genes

<br>
<br>

### Motivation

The [Gene Ontology](http://www.geneontology.org/) ([GO](http://www.geneontology.org/)) is the _de facto_ standard for the functional description of gene products, providing a consistent, information-rich terminology applicable across species and information repositories. Due to the fast increase of biomedical data annotated by GO vocabulary, an intelligent method for functional similarities of genes based on their GO annotations is of the greatest importance. This similarity measurement is highly critical since compared with sequence and structure similarity, the functional similarity is more informative for the understanding of the biological roles and cellular functions of genes. Many important applications in computational molecular biology such as gene clustering, protein function prediction, protein interaction evaluation and disease gene prioritization require functional similarity. Also, to expedite the selection of candidate genes for gene-disease research, genetic association studies, biomarker, and drug target selection, biomedical search engines can retrieve genes based on their functions from proteome databases.

### Problem

Existing GO-based gene functional similarity measures combine semantic similarity scores of single GO-term pairs to estimate genes' functional similarity. However, these measures are strictly dependent on the ever-changing topological structure of GO; they are very slow and task-dependent leaving no room for generalization. Our previous model, [simDEF](https://academic.oup.com/bioinformatics/article/32/9/1380/1743954), avoided these drawbacks by taking into account the significant advantage of distributed (vector-based) representation of GO terms using their textual definitions. simDEF, however, suffers from some unaddressed yet important shortcomings, many of which are still shared with the previous models. Manual feature engineering, relatively large dimensions of distributed GO-term vectors, the use of traditional metrics to aggregate GO-term similarity scores prior to computation of genes functional similarity, and, resorting to separate evaluation of each sub-ontology in GO ([_biological process_ or _BP_](http://geneontology.org/page/biological-process-ontology-guidelines), [_cellular component_ or _CC_](http://geneontology.org/page/cellular-component-ontology-guidelines), or [_molecular function_ or _MF_](http://geneontology.org/page/molecular-function-ontology-guidelines)) in a biological task, are some of these inadequacies.

### Contribution

In this project, by relying on the expressive power of deep neural networks, we introduce and develop deepSimDEF, an efficient method for measuring functional similarity of proteins and other gene products (e.g. microRNA and mRNA) using - natural language definitions of - GO terms annotating those genes. For this purpose, deepSimDEF neural network(s) (single-channel, and multi-channel depicted in Fig. 1) learn low-dimensional vectors of GO terms and gene products and then learn how to calculate the functional similarity of protein pairs using these learned vectors (aka embeddings). Relative to existing similarity measures, validation of deepSimDEF on yeast and human reference datasets yielded increases in [protein-protein interactions (PPIs)](https://en.wikipedia.org/wiki/Protein%E2%80%93protein_interaction) predictability by >4.5% and ~5%, respectively; a correlation improvement of ~9% and ~6% with yeast and human [gene co-expression](https://en.wikipedia.org/wiki/Gene_expression) values; and improved correlation with [sequence homology](https://en.wikipedia.org/wiki/Sequence_homology) by up to 6% for both organisms studied.

<br>
<p align="center">
<img src="imgs/deepSimDEF_multi_channel_network.jpg" width="940"> <br>
<br>
<b>Figure 1</b>: deepSimDEF multi-channel network architecture.
</p>

For a picturial view of a single-channel deepSimDEF see: [deepSimDEF_single_channel_BP](https://github.com/ahmadpgh/deepSimDEF/blob/master/imgs/deepSimDEF_single_channel_BP.jpg)
### Datasets for deepSimDEF Evaluation

The table below provides an overview of the prepared datasets for the evaluation tasks in the study (for more details refer to the paper). 


<div align="center">
<table>
  <tr>
   <td>
   </td>
   <td colspan="2" ><strong><sub>Yeast Dataset</sub></strong>
   </td>
   <td colspan="2" ><strong><sub>Human Dataset</sub></strong>
   </td>
   <td rowspan="2" colspan="2" ><strong><sub>Task</sub></strong>
   </td>
  </tr>
  <tr>
   <td>
   </td>
   <td><sub>Number of
<p>
gene pairs</sub>
   </td>
   <td><sub>Number of
<p>
genes</sub>
   </td>
   <td><sub>Number of
<p>
gene pairs</sub>
   </td>
   <td><sub>Number of
<p>
genes</sub>
   </td>
  </tr>
  <tr>
   <td><strong><sub>Protein-Protein Interaction</sub></strong>
   </td>
   <td><sub>50,154</sub>
   </td>
   <td><sub>4,591</sub>
   </td>
   <td><sub>65,542</sub>
   </td>
   <td><sub>14,096</sub>
   </td>
   <td colspan="2" ><sub>Classification of protein interactions</sub>
   </td>
  </tr>
  <tr>
   <td><strong><sub>Sequence Homology</sub></strong>
   </td>
   <td><sub>26,757</sub>
   </td>
   <td><sub>3,972</sub>
   </td>
   <td><sub>381,379</sub>
   </td>
   <td><sub>13,626</sub>
   </td>
   <td colspan="2" ><sub>Prediction of sequence similarity score</sub>
   </td>
  </tr>
  <tr>
   <td><strong><sub>Gene Expresion</sub></strong>
   </td>
   <td><sub>37,405</sub>
   </td>
   <td><sub>2,239</sub>
   </td>
   <td><sub>62,470</sub>
   </td>
   <td><sub>2,361</sub>
   </td>
   <td colspan="2" ><sub>Prediction of co-expression value</sub>
   </td>
  </tr>
</table>
</div>


## Code Instruction
The deepSimDEF networks were natively implemented and tested using deep learning API [tensorflow](https://www.tensorflow.org/) 2.4.0. Even though manual installation of python packages is an option, for ease of use and also to avoid any platform misconfiguration and package incompatibility we recommend you have [Anaconda](https://www.anaconda.com/products/individual) downloaded and installed and then create a conda virtual environment with the `environment.yml` provided with the project using the command:
```
conda env create -f environment.yml
```
The first line of the yml file sets the new environment's name. <br>

To activate the environment with the current name use:
```
conda activate deepSimDEF_env
```
You should be able to run the project code from this point forward. Also, for further familiarity with conda commands, please refer to [managing environments page](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html).

### Datasets
The datasets prepared and used in the experiments of the study (see table above) are provided in the `data` folder of the project under `data/species/[human|yeast]/` subdirectories. Since these datasets are generated based on the latest available resources at the time (e.g., annotations of genes from Gene Ontology, or PPI interactions from [STRING database](https://string-db.org/cgi/download?sessionId=bScolWa04rvN), etc) three jupyter notebooks which were responsible to create these datasets from the available data resources are shared in the `data` directory as well. For the generation of the same or more recent version of the datasets, follow the instruction provided in the jupyter notebooks. Typically the latest releases of the data resources would be downloaded by default unless otherwise is indicated or set by the user. The three jupyter notebooks, namely are:
* `protein_protein_interaction_data_prepration.ipynb`
* `sequence_homology_data_prepration.ipynb`
* `gene_expression_data_prepration.ipynb`

Notice: Since the gene expression dataset is built from the protected [GTEx database](https://gtexportal.org/home/) with restricted access, the generated data is password protected in its directory. Please contact us to see if we can help you with granting access.

### Gene Ontology Term Embedding
Ideally, the first layer of a deepSimDEF network gets initialized by pre-trained GO-term embeddings while they get fine-tuned during training. This scheme facilitates network optimization and improves model accuracy. The precomputed GO-term embeddings based on Fig. 2 are provided in `data/gene_ontology/definition_embedding/[50|100|150|200|300]_dimensional` directory (GO release version: 2021-07-02). Regarding deepSimDEF networks and our experiments GO-term embedding size of 100 yielded the best results (see the paper). 
<br>
<p align="center">
<img src="imgs/deepSimDEF_GO_term_embedding.jpg" width="650"> <br>
<br>
<b>Figure 2</b>: deepSimDEF definition-based GO-term embedding approach.
</p>
Since Gene Ontology gets constantly updated by having new terms added and a few old ones marked as obsolete (if needed), the jupyter notebook <code>gene_ontology_term_embedding.ipynb</code> allows you to create GO-term embeddings of the latest release of GO in the future (follow the instruction in the jupyter notebook and make certain you have enough physical memory). In case not enough resources are available, for the new GO term, the avarage of all the current GO-terms embeddings (or their immediate neighbor GO-terms embeddings) could also represent the embeddings and most probabely help with your application. Additionally, the jupyter networks <code>embeddings_similarity_evaluation.ipynb</code> allows to evaluate the quality of the generated embeddings based on their "semantic" similarity.

### Gene-GO term Associations
The Gene Ontology Consortium stores annotation data, the representation of gene product attributes using GO terms, in [standardized tab-delimited text files named GAF files](http://current.geneontology.org/products/pages/downloads.html). Each line in the file represents a single association between a gene product and a GO term, with an evidence code and the reference to support the link. These annotations from the latest GAF files for Yeast and Human are processed and the results annotations for the genes of interest are saved in `data/species/[human|yeast]/association_file/processed/` directories (IEA+ and IEA-). Additionally, the jupyter notebook of this process is included in the `data` directory for future use; it is named: `gene_association_and_annotations_preprocessing.ipynb`.

### deepSimDEF Model and Networks
With respect to the three experiments described above three different main python scripts are provided:
* `deepSimDEF_for_protein_protein_interaction.py`
* `deepSimDEF_for_sequence_homology.py`
* `deepSimDEF_for_gene_expression.py`

We recommend running these scripts on GPUs with a command similar to:

```
CUDA_VISIBLE_DEVICES=gpu-number python deepSimDEF_for_protein_protein_interaction.py arguments
```
A sample of such run for human species and BP sub-ontology (hence single-channel deepSimDEF) would be:
```
CUDA_VISIBLE_DEVICES=0 python deepSimDEF_for_protein_protein_interaction.py --species human --sub_ontology bp
```

All these scripts make use of the deepSimDEF network implementation provided in `networks.py` script. We strictly advise you to read the description of the arguments in these main files as they provide ample information regarding how to set a run of your interest properly. Briefly, though, these scripts could be run on either three deepSimDEF modes: training (default), evaluation, or production.
* In **training mode**, as the name suggests, the model(s) would be trained and evaluated on the fly (number of models would be equal to the number of folds). During this process, the statistical results would be logged and the network(s) would be checkpointed according to the provided setting in the arguments. For training and evaluation, by default, the model(s) use data provided with this project unless you provide another set of data with `--inpute_file` (ideally prepared by the preparation `.ipynb` files).
* In **evaluation mode**, after providing checkpointed model(s), the statistical results of interests for the test split(s) would be printed out (should obtain similar results from the same epoch in the training mode). These statistical results are F1-score for the PPI experiment, and Pearson's and Spearman's correlation values for the sequence homology and gene expression experiments.
* In **production mode**, after providing a checkpointed model (typically "trained" on all existing data by setting `--nb_fold 1`) as well as a file of (typically new) gene-product pairs, the estimated scores of them would be saved or printed out. For this purpose, in the `--production_input_file` each gene-product pair should be shown on one line while gene names are separated by a tab or space). NOTICE: In this mode, if a gene in the gene pair is not present in the given processed association files or it has GO annotations not seen before during training, their pairs would be eliminated in the output. To deal with this matter, before training a model, you need to make sure the association files are up-to-date, and also relax our restriction in the study (i.e., modify the main `.py` and data and association `.ipynb` files as we consider only genes that are annotated by all three GO sub-ontologies).

Depending also on the GO sub-ontology of your choice using the argument `--sub_ontology`, if you provide `bp`, `cc`, or `mf`, automatically single-channel deepSimDEF would be set. However, if `--sub_ontology all` (default) multi-channel deepSimDEF would be triggered. These types of settings should be consistent across the deepSimDEF modes described above.

## Cite

Please cite our papers, code, and dataset if you use them in your work.

deepSimDEF paper, and aforementioned code, and dataset:
```
@article{,
  title = {deepSimDEF: deep neural embeddings of gene products and Gene Ontology terms for functional analysis of genes},
  author = {Pesaranghader, Ahmad and Matwin, Stan and Sokolova, Marina and Beiko, Robert G, and Grenier, Jean-Christophe and Hussin, Julie},
  journal = {under review},
  year = {2021},
  publisher={under review}
}
```

[simDEF](https://academic.oup.com/bioinformatics/article/32/9/1380/1743954) paper:
```
@article{pesaranghader2016simdef,
  title={simDEF: definition-based semantic similarity measure of gene ontology terms for functional similarity analysis of genes},
  author={Pesaranghader, Ahmad and Matwin, Stan and Sokolova, Marina and Beiko, Robert G},
  journal={Bioinformatics}, volume={32}, number={9},
  pages={1380--1387}, year={2016}, publisher={Oxford University Press}
}
```
<sub>Ahmad Pesaranghader © 2021</sub>
