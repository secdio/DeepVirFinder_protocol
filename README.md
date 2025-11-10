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

<!--You should have received a copy of the GNU General Public License along with this program. If not, see http://www.gnu.org/licenses/.-->

