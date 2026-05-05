# A beginner’s guide to using DeepVirFinder for viral sequence identification from metagenomic datasets

Version: 1.2

Authors: Yuqian Mo, Nathan Ahlgren, Jed A. Fuhrman, Fengzhu Sun, Shengwei Hou

Maintainer: Yuqian Mo, Jie Ren renj@usc.edu, Chao Deng chaodeng@usc.edu


## Description

Identifying viral sequences from metagenomic datasets is critical for investigating their origins, evolutionary patterns, and ecological functions. Previously, we developed a novel deep learning software, DeepVirFinder, to predict viral sequences from shotgun metagenomic assemblies. This method employs a siamese convolutional neural network model to extract features from known viral and prokaryotic host genomic sequences for binary classification of input query sequences. With the rapid accumulation of environmental metagenomic data, this approach has accelerated the discovery of novel viruses from diverse environments through an alignment-free and reference-free deep learning strategy. To facilitate the rapid adoption of this software for beginning users, here we have further improved DeepVirFinder by optimizing its runtime performance, while maintaining the essential user interface of the original version. This comprehensive guide provides basic workflows for the most common use cases of DeepVirFinder. Additionally, to assist users in downstream analyses, supplementary scripts were provided in the software for extracting viral sequences and inspecting the results, thereby helping researchers more effectively mine viral information from metagenomic datasets.







Copyright and License Information
-----------------------------------

Copyright (C) 2019 University of Southern California

Authors: Jie Ren, Kai Song, Chao Deng, Nathan Ahlgren, Jed Fuhrman, Yi Li, Xiaohui Xie, Ryan Poplin, Fengzhu Sun

This program is available under the terms of USC-RL v1.0. 

Commercial users should contact Dr. Sun at fsun@usc.edu, copyright at the University of Southern California.



**<span class="mark">Basic Protocol 1: PREDICTING VIRAL SEQUENCES IN
METAGENOMIC ASSEMBLIES</span>**
Here, we provide a detailed workflow for predicting viral sequences in
metagenomic assemblies using DeepVirFinder, including the preparation of
dependencies, testing the installation, and demonstrating the complete
viral sequence prediction pipeline using the *Tara* Oceans viromic
dataset as a case study.

**<span class="mark">Required Resources</span>**

**<span class="mark">Hardware</span>**

DeepVirFinder is a command-line tool requiring a Linux system. We
recommend deploying it in computational environments with at least 4 CPU
threads, 4 GB RAM, and 2 GB disk space.

**<span class="mark">Software</span>**

DeepVirFinder requires Python 3 along with essential packages, including
NumPy, PyTorch, and other specified packages listed in the
dvf_protocol.yml file. We recommend using Miniconda/Conda to create an
isolated environment for software dependencies and version management.

**<span class="mark">Input files</span>**

The pipeline requires assembled metagenomic sequences in standard FASTA
format.

1.  We recommend installing DeepVirFinder by cloning the source code
    from the GitHub repository:

> \$ git clone https://github.com/secdio/DeepVirFinder_protocol.git
>
> \$ cd DeepVirFinder_protocol/

2.  Users can create a Miniconda/Conda environment for running
    DeepVirFinder based on the provided dvf_protocol.yml configuration
    file, and then activate the Conda environment and run the
    installation test using the sample data to demonstrate the standard
    running command:

> \$ conda env create -f dvf_protocol.yml -n dvf_protocol
>
> \$ conda activate dvf_protocol
>
> \$ python dvf.py -i ./test/CRC_meta.fa -o ./test/ -l 300 -c 16
>
> The program will complete in about one minute. Upon successful
> completion, the following message should be displayed on the screen,
> indicating that DeepVirFinder has been installed correctly and the
> test has passed.
>
> python dvf.py -i ./test/CRC_meta.fa -o ./test/ -l 300 -c 16
>
> Using CPU/GPU: running on CPU
>
> 1\. Loading Models.
>
> model directory /home/data/dvf_protocol/DeepVirFinder/models
>
> 2\. Encoding and Predicting Sequences.
>
> Processed 50 sequences...
>
> Processed 100 sequences...
>
> Processed 150 sequences...
>
> Processed 200 sequences...
>
> Processed 250 sequences...
>
> Processed 300 sequences...
>
> Processed 350 sequences...
>
> Processed 400 sequences...
>
> Processed 450 sequences...
>
> Processed 500 sequences...
>
> Processed 550 sequences...
>
> Processed 600 sequences...
>
> Processed 650 sequences...
>
> Processed 700 sequences...
>
> Processed 750 sequences...
>
> Processed 800 sequences...
>
> Processed 850 sequences...
>
> Processed 900 sequences...
>
> Processed 950 sequences...
>
> Processed 1000 sequences...
>
> 3\. Done. Thank you for using DeepVirFinder.
>
> output in /home/data/
> dvf_protocol/DeepVirFinder/test/CRC_meta.fa_gt300bp_dvfpred.txt

3.  After verifying that DeepVirFinder can be successfully run, users
    can analyze their metagenomic data by specifying the following
    parameters:

> \(1\) a fasta file (mandatory, see Table 1);
>
> \(2\) an output directory (optional, see Table 2);
>
> \(3\) a sequence length cutoff for filtering (optional, see Table 2);
>
> \(4\) the number of CPU cores (optional, see Table 2).
>
> \(5\) whether to enable GPU acceleration for the run (optional, see
> Table 2).
>
> \$ python dvf.py -i your_metagenomic_file.fasta -o your_output -l 300
> -c 16

4.  Upon successful completion, users may examine the results in the
    output directory. The primary output file is a text file (.txt)
    structured as described in Table 3, containing the following fields:

<!-- -->

1)  Sequence identifier

2)  Sequence length

3)  Predicted viral score (0-1)

4)  Calculated *p*-value (0-1)

5.  (Optional) To enhance the statistical rigor of viral sequence
    identification, we recommend calculating *q*-values for false
    discovery rate (FDR) control as a more conservative metric than
    *p*-values. Users can compute *q*-values using the R package as
    follows:

<!-- -->

1)  To install the package "qvalue" in R:

> \> install.packages("BiocManager")
>
> \> BiocManager::install("qvalue")

2)  To compute the *q*-values, load the package and call the function
    'qvalue'. For example,

> \# load the package qvalue
>
> \> library(qvalue)
>
> \# read the prediction results
>
> \> result \<- read.csv("./test/CRC_meta.fa_gt300bp_dvfpred.txt",
> sep='\t')
>
> \# estimate q-values (false discovery rates) based on p-values
>
> \> result\$qvalue \<- qvalue(result\$pvalue)\$qvalues
>
> \# sort sequences by q-value in ascending order
>
> \> result\[order(result\$qvalue),\]

6.  (Optional) To facilitate further filtering of prediction results, we
    provide a post-processing script extract_virus.py. This script
    allows users to extract sequences identified as viral by
    DeepVirFinder based on user-defined score and *p*-value thresholds.

> To run the script, the user needs to provide the DeepVirFinder output
> file, the original metagenomic FASTA file, and the desired filtering
> thresholds for the score and *p*-value fields. The script will then
> extract sequences that meet the specified criteria, such as a score
> ＞0.9 and a p-value ＜0.05, and generate a filtered FASTA file
> containing only those sequences predicted to be viral.
>
> \$ python extract_virus.py -j ./test/CRC_meta.fa_gt300bp_dvfpred.txt
> \\
>
> -f ./test/CRC_meta.fa -o ./test/filter -s 0.9 -p 0.05
>
> The screen output is as follows:
>
> Reading prediction file...
>
> Total: 1000, Passed: 90, Not passed: 910, Percent: 9.00%
>
> Saving filtered IDs...
>
> Filtering fasta...
>
> Filtered 90 sequences. Output folder: ./test/filter

7.  (Optional) We also provide a visualization script to comprehensively
    analyze all results. Users can employ the visualize.py script to (1)
    examine viral score distribution, (2) visualize *p*-value
    distribution, (3) calculate the proportion of filtered sequences
    relative to the total sequence, and (4) identify the distributions
    of high-confidence/high-score sequences.

> \$ python visualize.py -j ./test/CRC_meta.fa_gt300bp_dvfpred.txt -f
> ./test/CRC_meta.fa -o ./test/filter -s 0.9 -p 0.05
