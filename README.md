# deepSimDEF: Deep Definition-based Semantic Similarity of Gene Ontology-Term and Gene-Product Embeddings for Functional Analysis of Genes

## Background

The Gene Ontology (GO) is the _de facto_ standard for the functional description of gene products, providing a consistent, information-rich terminology applicable across species and information repositories. Due to the fast increase of biomedical data annotated by GO vocabulary, an intelligent method for semantic similarity measurement between GO terms which facilitates analysis of functional similarities of genes is of the greatest importance. In practice, this similarity measurement is highly critical since compared with sequence and structure similarity, the functional similarity is more informative for the understanding of the biological roles and cellular functions of genes. Many important applications in computational molecular biology such as gene clustering, protein function prediction, protein interaction evaluation and disease gene prioritization require functional similarity. In addition, to expedite the selection of candidate genes for gene-disease research, genetic association studies, biomarker and drug target selection, and animal models of human diseases, it is essential to have search engines that can retrieve genes by their functions from proteome databases. By the fast advancement in the domain, these engines can substantially gain benefits from the functional similarity calculation of gene products.

## Problem

As far as GO is concerned, most of existing gene functional similarity measures combine semantic similarity scores of single GO term pairs to estimate genes functional similarity (pair-wise measures), whereas others compare GO terms in groups for this measurement (group-wise measures). However, almost all of these measures are strictly dependent on the ever-changing topological structure of GO; they are very slow and extremely task-dependent leaving no room for their generalization, and none of them takes the valuable textual definition of GO terms into consideration. Our previous model, simDEF, avoids these drawbacks by taking into account the significant advantage of distributed (vector-based) representation of GO terms using their textual definitions. However, simDEF suffers from some unaddressed yet important shortcomings, many of which are still shared with the previous models. Manual feature engineering, relatively large dimensions of distributed GO-term vectors, the use of traditional metrics to aggregate GO-term similarity scores prior to computation of genes functional similarity, and, resorting to separate evaluation of each sub-ontology in GO (biological process, cellular component, or molecular function) in a biological task, are some of these inadequacies. These limitations present the challenges of measuring genes functional similarity reliably.

## Contribution

In this project, by relying on the expressive power of deep neural networks, we lay out and develop deepSimDEF, an efficient method for measuring functional similarity of proteins and gene products (e.g. microRNA and mRNA) using - natural language definitions of - GO terms annotating those genes. For this purpose, deepSimDEF neural network(s) (single-channel and multi-channel) learn low-dimensional vectors of GO terms and gene products and then learn how to calculate the functional similarity of protein pairs using these learned vectors (aka embeddings). Relative to existing similarity measures, when validated on a yeast (_Saccharomyces cerevisiae_) reference database, deepSimDEF improves correlation with **sequence homology** by up to 28%. deepSimDEF also outperforms the existing measures by great margins in the task of prediction of **protein-protein interactions (PPIs)** and **gene expression** analysis through the use of GO annotations.  
  
**Extra experiments:** Apart from remarkable results and the valuable distributed representations of GO terms and gene products, the introduction of the powerful yet flexible, easily transferable and adaptable architectures of deepSimDEF to a wide range of problems in proteomics and genomics lies at the heart of this study. Along with these experiments, evaluation of electronic GO annotations (IEAs), visualization of gene products for a deeper inference from the embeddings, and examination of transfer learning among various biological data and their associated networks shape other parts of our study.

## Datasets for the evaluation

The datasets built in the study and employed in the evaluation analyses include (see the 'EXPERIMENTAL DATA' section, 'Validation datasets' subsection for detail):
1. Sequence Homology Data (_~20,000 protein pairs_)
2. Gene Expression Data (_~2,000,000 protein pairs_)
3. PPI Data (_~33,000 protein pairs_)

<br/>

_**The project is under completion at this moment.**_

<br>
<br>

<sub>Ahmad Pesaranghader © 2017</sub>
