# Phylogenetic Manifold Regularization
This repo presents codes for generating results with phylogenetic manifold regularization presented in the paper "Phylogenetic Manifold Regularization: Asemi-supervised approach to predict transcriptionfactor binding sites"

## Dependencies
PhyloReg requires Python 3.7.3. 

The dependencies can be installed by running

`pip install -R Requirements.txt`

## Experiments with Simulated Datasets
To get the available options

`python expt_sim.py -h`

To run 10 experiments for the assessment of phylogenetic regularization in the euclidean space of 10 dimension, 10 labelled examples, selection pressure of 100, 7 orthologs per example, 

`time python -W ignore expt_sim.py --dim 10 --num-examples 10 --sel 100 --num-nodes 7 --repeat 10 `

## Experiments with ChIP-Seq Datasets
We show the efficiency of PhyloReg by showing improvement in AUROC score for a hybrid RNN-CNN architecture called Factornet proposed by Quang et al. D.  Quang  and  X.  Xie,  “Factornet:  a  deep  learning  framework  forpredicting cell type specific transcription factor binding from nucleotide-resolution sequential data,”Methods, vol. 166, pp. 40–47, 2019.

### Data sets
We use the ChipSeq data available at 
`http://cnn.csail.mit.edu/motif_occupancy/` , where each ChIP-Seq experiment consist of train and test sets of 101 bps in hg19 reference genome.

We extended the 101 bps regions to 1001 bps because factornet is designed for input size of 1000 bps, much larger than 101 bps and the binding sites in orthologous regions might be located in neighbor regions. The data processing is described in our paper. The processed data is available at 

### Running Factornet with PhyloReg

To get the available options

`time python expt_factornet.py -h`

To run the model with the data provided in this repository

To evaluate 
