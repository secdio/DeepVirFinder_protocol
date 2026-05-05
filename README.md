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
<!--You should have received a copy of the GNU General Public License along with this program. If not, see http://www.gnu.org/licenses/.-->

