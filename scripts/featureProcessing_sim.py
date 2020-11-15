"""creates feature of sequence files
Input: seqfile {0: regionTag, 1: spTag, 2: sequence, 4: seqLength
Output: Labelled example feature matrix, list of labelled species,
        corresponding orthologs with their species name {speciesList, featurevectorList},
        labels

(c) Faizy Ahsan
email: zaifyahsan@gmail.com"""

import numpy as np
from tqdm import tqdm

def create_feature_matrix(labelled_species_list, sample_bag, mode, sp_to_id, num_species):
    """ Processes bed4 like file for phyloreg models

    -------Parameters--------
    labelled_species_list: list of strings, e.g. ['hg38']
    feature_filename: path to bed4 file

    -------Returns-----------
    feature_info: dictionary, contains processed examples, its orthologs and labels
    """
    orthologs = dict()
    labels = dict()
    labelled_examples = dict()

    for line_index in tqdm(range(len(sample_bag))):
        line = sample_bag[line_index]
        region_tag = float(line[0])
        sptag = int(float(line[1]))
        region_label = float(line[-1])
        feature_vector = np.array([float(item) for item in line[2:-1]])

        # add ortholog features
        if mode == 'train':
            if region_tag in orthologs.keys():
                orthologs[region_tag][sp_to_id[sptag]] = feature_vector

            else:
                orthologs[region_tag] = np.zeros(shape=(num_species , feature_vector.shape[0]))
                orthologs[region_tag][sp_to_id[sptag]] = feature_vector

        # add labelled values
        if sptag in labelled_species_list:
            # add labels
            if region_tag not in labels.keys():
                labels[region_tag] = region_label
            # add features
            if region_tag not in labelled_examples.keys():
                labelled_examples[region_tag] = feature_vector

    if mode == 'train':
        feature_info = {'labelled_examples': labelled_examples,
                        'orthologs': orthologs,
                        'labels': labels
                        }
    else:
        feature_info = {'labelled_examples': labelled_examples,
                        'labels': labels
                        }
    del labelled_examples, labels, orthologs

    return feature_info
