# Phylogenetic Manifold Regularization
This repo presents codes for generating results with phylogenetic manifold regularization presented in the paper "Phylogenetic Manifold Regularization: Asemi-supervised approach to predict transcriptionfactor binding sites"

# Dependencies
Python 3.7.3

## Experiments with Simulated Datasets
To get the available options

`python expt_sim.py -h`

To run 10 experiments for the assessment of phylogenetic regularization in the euclidean space of 10 dimension, 10 labelled examples, selection pressure of 100, 7 orthologs per example, 

`time python -W ignore expt_sim.py --dim 10 --num-examples 10 --sel 100 --num-nodes 7 --repeat 10 `

## Experiments with ChIP-Seq Datasets

We observed that dropout was not helpful

### Data sets
We use the ChipSeq data available at 
`http://cnn.csail.mit.edu/motif_occupancy/`
The training regions overlapping with test regions are removed
1. the original files
-- subtract test regions from original train regions
2. add 50 bps on both sides and then remove the overlapping regions

***before liftover subtract 1 position in hg19 

Split chromosome file into non-overlapping files (for both 1 and 2)

Create big bed file for each chromosome

TO DO
1. create just hg19 train and hg19 test bed files
2. format the hg19 bed files with one position subtracted in the start
3. liftover hg19 to hg38 
[should be present in the github] 
[using liftover gives unmapped regions as well (splits, missing etc), alt chrom are ignored]
4. create big bed file
5. separate by chromosomes
6. create part files for chromosomes
7. run maf region
8. aggregate maf regions
9. final files [should be present in the github]
