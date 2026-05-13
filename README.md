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






-----------------------------------

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

    ```bash
    $ git clone https://github.com/secdio/DeepVirFinder_protocol.git
    $ cd DeepVirFinder_protocol/
    ```

2.  Users can create a Miniconda/Conda environment for running
    DeepVirFinder based on the provided dvf_protocol.yml configuration
    file, and then activate the Conda environment and run the
    installation test using the sample data to demonstrate the standard
    running command:

    ```bash
    $ conda env create -f dvf_protocol.yml -n dvf_protocol
    $ conda activate dvf_protocol
    $ python dvf.py -i ./test/CRC_meta.fa -o ./test/ -l 300 -c 16
    ```

    The program will complete in about one minute. Upon successful
    completion, the following message should be displayed on the screen,
    indicating that DeepVirFinder has been installed correctly and the
    test has passed.

    ```text
    python dvf.py -i ./test/CRC_meta.fa -o ./test/ -l 300 -c 16
    
    Using CPU/GPU: running on CPU
    
    1. Loading Models.
    
    model directory /home/data/dvf_protocol/DeepVirFinder/models
    
    2. Encoding and Predicting Sequences.
    
    Processed 50 sequences...
    
    Processed 100 sequences...
    
    Processed 150 sequences...
    
    Processed 200 sequences...
    
    Processed 250 sequences...
    
    Processed 300 sequences...
    
    Processed 350 sequences...
    
    Processed 400 sequences...
    
    Processed 450 sequences...
    
    Processed 500 sequences...
    
    Processed 550 sequences...
    
    Processed 600 sequences...
    
    Processed 650 sequences...
    
    Processed 700 sequences...
    
    Processed 750 sequences...
    
    Processed 800 sequences...
    
    Processed 850 sequences...
    
    Processed 900 sequences...
    
    Processed 950 sequences...
    
    Processed 1000 sequences...
    
    3. Done. Thank you for using DeepVirFinder.
    
    output in /home/data/
    dvf_protocol/DeepVirFinder/test/CRC_meta.fa_gt300bp_dvfpred.txt
    ```

3.  After verifying that DeepVirFinder can be successfully run, users
    can analyze their metagenomic data by specifying the following
    parameters:

    (1) a fasta file (mandatory);

    (2) an output directory (optional);

    (3) a sequence length cutoff for filtering (optional);

    (4) the number of CPU cores (optional).

    (5) whether to enable GPU acceleration for the run (optional).

    ```bash
    $ python dvf.py -i your_metagenomic_file.fasta -o your_output -l 300 -c 16
    ```

4.  Upon successful completion, users may examine the results in the
    output directory. The primary output file is a text file (.txt)
   , containing the following fields:

    (1) Sequence identifier

    (2) Sequence length

    (3) Predicted viral score (0-1)

    (4) Calculated p-value (0-1)

5.  (Optional) To enhance the statistical rigor of viral sequence
    identification, we recommend calculating q-values for false
    discovery rate (FDR) control as a more conservative metric than
    p-values. Users can compute q-values using the R package as
    follows:

    (1) To install the package "qvalue" in R:

    ```r
    install.packages("BiocManager")
    BiocManager::install("qvalue")
    ```
  
    (2) To compute the q-values, load the package and call the function
    qvalue. For example,

    ```r
    # load the package qvalue
    library(qvalue)
    
    # read the prediction results
    result <- read.csv("./test/CRC_meta.fa_gt300bp_dvfpred.txt", sep = "\t")
    
    # estimate q-values (false discovery rates) based on p-values
    result$qvalue <- qvalue(result$pvalue)$qvalues
    
    # sort sequences by q-value in ascending order
    result[order(result$qvalue), ]
    ```

6.  (Optional) To facilitate further filtering of prediction results, we
    provide a post-processing script extract_virus.py. This script
    allows users to extract sequences identified as viral by
    DeepVirFinder based on user-defined score and p-value thresholds.

    To run the script, the user needs to provide the DeepVirFinder output
    file, the original metagenomic FASTA file, and the desired filtering
    thresholds for the score and p-value fields. The script will then
    extract sequences that meet the specified criteria, such as a score
    ＞0.9 and a p-value ＜0.05, and generate a filtered FASTA file
    containing only those sequences predicted to be viral.

    ```bash
    $ python extract_virus.py -j ./test/CRC_meta.fa_gt300bp_dvfpred.txt \
      -f ./test/CRC_meta.fa -o ./test/filter -s 0.9 -p 0.05
    ```

    The screen output is as follows:

    ```text
    Reading prediction file...
    
    Total: 1000, Passed: 90, Not passed: 910, Percent: 9.00%
    
    Saving filtered IDs...
    
    Filtering fasta...
    
    Filtered 90 sequences. Output folder: ./test/filter
    ```

7.  (Optional) We also provide a visualization script to comprehensively
    analyze all results. Users can employ the visualize.py script to (1)
    examine viral score distribution, (2) visualize *p*-value
    distribution, (3) calculate the proportion of filtered sequences
    relative to the total sequence, and (4) identify the distributions
    of high-confidence/high-score sequences.

    ```bash
    $ python visualize.py -j ./test/CRC_meta.fa_gt300bp_dvfpred.txt -f ./test/CRC_meta.fa -o ./test/filter -s 0.9 -p 0.05
    ```

**<span class="mark">Basic Protocol 2: AN INTEGRATED PIPELINE FOR VIRAL
SEQUENCE ANALYSIS: PREDICTION, EXTRACTION, AND VISUALIZATION</span>**

We have developed an integrated pipeline script that streamlines the
entire analytical workflow, encompassing: (1) running of dvf.py for
viral sequence prediction, (2) filtration of results based on
user-defined viral score and *p*-value thresholds, and (3) automated
visualization of outputs.

**<span class="mark">Required Resources</span>**

**Hardware**

DeepVirFinder compatible Linux machine meeting the minimum hardware
requirements as specified in Basic Protocol 1.

**Software**

DeepVirFinder and dependencies as specified in Basic Protocol 1.

**Input files**

The pipeline requires assembled metagenomic sequences in standard FASTA
format.

This pipeline demonstration utilizes a test dataset (provided in
./test/) to validate the analytical workflow. For user-specific
analyses, simply replace the example data with your own metagenomic
files while maintaining the required input format (FASTA).

1.  Users can run the pipeline script with the following command,
    specifying the required parameters:

    \(1\)  Input fasta files

    \(2\) Output directory

    \(3\) Cutoff length

    \(4\)  Number of cores

    \(5\) Threshold of scores

    \(6\) Threshold of *p*-values

    ```bash
    $ python pipeline.py -i ./test/TOV_43_sampled_80M.fna -o ./test/TOV -l 1500 -c 16 --score 0.9 --pvalue 0.05
    ```

2.  Upon successful completion, users may examine the complete results
    in the output directory:


    \(1\) Prediction Results from dvf.py

    \(2\)  Filtered Viral Sequences

    \(3\)  Viral Sequence IDs

    \(4\) Visualization Outputs

**<span class="mark">Basic Protocol 3：RETRAINING THE DeepVirFinder
MODEL USING A CUSTOMIZED DATASET</span>**

Users are welcome to retrain an updated deep learning model using their
own datasets. Here, we provide the scripts for processing the genomic
data and retraining the model.

**<span class="mark">Required Resources</span>**

**A. Hardware**

DeepVirFinder compatible Linux machine meeting the minimum hardware
requirements as specified in Basic Protocol 1.

**B. Software**

DeepVirFinder and dependencies as specified in Basic Protocol 1.

**C. Input files**

\(1\) the host genomic sequences for training,

\(2\) the host genomic sequences for validation,

\(3\) the virus genomic sequences for training,

\(4\) the virus genomic sequences for validation.

1.  Running encode.py to encode the input files.

    This script is designed to segment input genomic sequences into
    fixed-length fragments and perform one-hot encoding for each fragment.
    The encoding results are separated into forward and reverse strands
    and are output as .npy files and .fasta files. The script encode.py
    processes the input genomic sequences by fragmenting them into
    fixed-length sequences \[-l\] and encoding them using the one-hot
    encoding scheme. The contig type \[-p\] indicates the type of the
    sequences, either virus or host. This indicator will be encoded into
    the file name and will be used in the following steps for data type
    recognition. Users can use the following command to encode viral and
    host genome files separately:

    ```bash
    # for training
    $ python encode.py -i ./train_example/tr/host_tr.fa -l 150 -p host
    $ python encode.py -i ./train_example/tr/virus_tr.fa -l 150 -p virus
    
    # for validation
    $ python encode.py -i ./train_example/val/host_val.fa -l 150 -p host
    $ python encode.py -i ./train_example/val/virus_val.fa -l 150 -p virus
    ```

    Part of the output is as follows,

    ```text
    Encoded sequences are saved in:
    
    - host#host_tr#0.15k_num1_seq19994_codefw.npy
    
    - host#host_tr#0.15k_num1_seq19994_codebw.npy
    
    - host#host_tr#0.15k_num1_seq19994.fasta
    
    Total fragments processed: 19994
    ```

3.  Running training.py to train the new model with encoded custom
    datasets

    The script training.py takes the encoded sequences and trains a deep
    learning model for classifying viruses from hosts. We strongly
    recommend using a GPU for this step; otherwise, the runtime may be
    significantly prolonged.

    The directory of the encoded training data \[-i\] and the directory of
    the encoded validation data \[-j\] need to be specified.
    Hyperparameters of the deep learning model include the number of
    filters in the convolutional layer \[-n\], the length of the filter
    \[-f\], and the number of neurons in the dense layer \[-d\]. Since
    viral sequences in real data can be of various lengths, we train
    multiple models using sequences of different lengths, e.g., 150, 300,
    500, 1000 bp, for predicting sequences of different length ranges. The
    option \[-l\] specifies the length of the sequences used for training.

    ```bash
    $ python training.py -l 150 -i ./train_example/tr/encoded \
      -j ./train_example/val/encoded -o ./train_example/test_models \
      -f 10 -n 500 -d 500 -e 10
    ```

    Part of the output is as follows,

    ```text
    ...loading data...
    
    ...loading virus data...
    
    data for training virus#virus_tr#0.15k_num1_seq20000_codefw.npy
    
    data for validation virus#virus_val#0.15k_num1_seq2000_codefw.npy
    
    ...loading host data...
    
    data for training host#host_tr#0.15k_num1_seq19994_codefw.npy
    
    data for validation host#host_val#0.15k_num1_seq1996_codefw.npy
    
    ...combining V and H...
    
    ...shuffling training data...
    
    ...building model...
    
    ...fitting model...
    
    0.15k_fl10_fn500_dn500_ep10
    
    Epoch 1/10 - train_loss: 0.619883 val_loss: 0.586210 val_auc: 0.778823
    
    Epoch 1: val_loss improved from nan to 0.586210, saving model to
    ./train_example/models/model_siamese_varlen_0.15k_fl10_fn500_dn500.pth
    
    Epoch 2/10 - train_loss: 0.579221 val_loss: 0.561745 val_auc: 0.786800
    
    Epoch 2: val_loss improved from 0.586210 to 0.561745, saving model to
    ./train_example/models/model_siamese_varlen_0.15k_fl10_fn500_dn500.pth
    
    Epoch 3/10 - train_loss: 0.560137 val_loss: 0.580362 val_auc: 0.790158
    
    Epoch 4/10 - train_loss: 0.541015 val_loss: 0.592850 val_auc: 0.786805
    
    Epoch 5/10 - train_loss: 0.524401 val_loss: 0.558194 val_auc: 0.799774
    
    Epoch 5: val_loss improved from 0.561745 to 0.558194, saving model to
    ./train_example/models/model_siamese_varlen_0.15k_fl10_fn500_dn500.pth
    
    Epoch 6/10 - train_loss: 0.506052 val_loss: 0.560106 val_auc: 0.804088
    
    Epoch 7/10 - train_loss: 0.487972 val_loss: 0.561029 val_auc: 0.811139
    
    Epoch 8/10 - train_loss: 0.472881 val_loss: 0.583559 val_auc: 0.801717
    
    Epoch 9/10 - train_loss: 0.459005 val_loss: 0.592404 val_auc: 0.808126
    
    Epoch 10/10 - train_loss: 0.446490 val_loss: 0.577975 val_auc:
    0.797927
    
    ...predicting tr...
    
    auc_tr=0.918028155946784
    
    ...predicting val...
    
    auc_val=0.7979273547094188
    
    Model (best) is at
    ./train_example/models/model_siamese_varlen_0.15k_fl10_fn500_dn500.pth
    ```

The following section demonstrates a complete test workflow. If you
need to train your own model, you should first divide the sequence
into fixed lengths in base pairs (such as 150, 300, 500, 1000, etc.)
and then train the model. We strongly recommend using a GPU-equipped
machine for training.


```bash
# Fragmenting sequences into fixed lengths, and encoding them using one-hot encoding (may take about 5 minutes)
for l in 150 300 500 1000; do
  # for training
  python encode.py -i ./train_example/tr/host_tr.fa -l $l -p host
  python encode.py -i ./train_example/tr/virus_tr.fa -l $l -p virus

  # for validation
  python encode.py -i ./train_example/val/host_val.fa -l $l -p host
  python encode.py -i ./train_example/val/virus_val.fa -l $l -p virus
done

# Training multiple models for different contig lengths
for l in 150 300 500 1000; do
  python training.py -l $l -i ./train_example/tr/encoded \
    -j ./train_example/val/encoded \
    -o ./train_example/new_models \
    -f 10 -n 500 -d 500 -e 10
done
```

4.  Using the new model to predict your metagenomic files.

    To predict sequences using the newly trained model, specify the model
    directory using the option -m,

    ```bash
    $ python dvf.py -i ./test/crAssphage.fa -o ./train_example/test \
      -l 300 -m ./train_example/new_models
    ```

