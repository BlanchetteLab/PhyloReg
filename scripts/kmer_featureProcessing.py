"""creates feature of sequence files
Input: seqfile {0: regionTag, 1: spTag, 2: sequence, 4: seqLength
Output: Labelled example feature matrix, list of labelled species,
        corresponding orthologs with their species name {speciesList, featurevectorList},
        labels

(c) Faizy Ahsan
email: zaifyahsan@gmail.com"""


import numpy as np
from tqdm import tqdm
import itertools

global kmer_to_id, id_to_kmer


def get_rev_comp(kmer):
    rev_d = {'A': 'T', 'C':'G', 'G':'C', 'T':'A'}
    rev_seq = kmer[::-1]
    return ''.join([rev_d[item] for item in rev_seq])


def count_kmer(seq):
    global kmer_to_id, id_to_kmer
    feat = np.zeros(2080, dtype=int)
    for i in range(len(seq) - 5):
        kmer = seq[i:i + 6]
        r_kmer = get_rev_comp(kmer)
        if kmer in kmer_to_id.keys():
            pos = kmer_to_id[kmer]
        else:
            pos = kmer_to_id[r_kmer]

        feat[pos] += 1
    return feat

def processFile(feature_filename=None, mode='train', c_id=None,
                parent_to_child=None, retain_ids=None,
                primate_and_anc=None, data_type='mammals'):
    """Processes bed4 like file for phyloreg models
    Reads featureFile line by line
    creates the three output variables
    1. labelled_examples
    2. labels
    3. orthologs

    ---------Parameters----------------------
    feature_filename: list of labelled species
    feature_filename: string, path to data file
    mode: string

    ---------Returns--------------------------
    labelled_examples: dictionary
    labels: dictionary
    orthologs: dictionary
    """

    global kmer_to_id, id_to_kmer

    bases = ['A', 'C', 'G', 'T']
    k = 6
    all_kmers = [''.join(p) for p in itertools.product(bases, repeat=k)]
    # create a ordered dictionary
    kmer_occured = []
    kmer_to_id = {}
    id_to_kmer = {}
    # put kmers only if either itself or its rev kmer is not present in the dictionary
    i = 0
    for kmer in all_kmers:
        r_kmer = get_rev_comp(kmer)
        if kmer in kmer_occured or r_kmer in kmer_occured:
            pass
        else:
            kmer_occured += [kmer, r_kmer]
            kmer_to_id[kmer] = i
            id_to_kmer[i] = kmer
            i += 1

    num_children = len(c_id)
    # print('num_children:', num_children)
    # mapper = {'A': [1., 0., 0., 0.],
    #           'C': [0., 1., 0., 0.],
    #           'G': [0., 0., 1., 0.],
    #           'T': [0., 0., 0., 1.],
    #           }
    mapper = {'A': [True, False, False, False],
              'C': [False, True, False, False],
              'G': [False, False, True, False],
              'T': [False, False, False, True],
              }
    orthologs = dict()
    labels = dict()
    labelled_examples = dict()

    # print('feature_filename:', feature_filename)

    lines = open(feature_filename, 'r').readlines()

    # for line_index in range(1000):
    for line_index in tqdm(range(1, len(lines))):
        line = lines[line_index]
        line = line.strip().split(',')

        sptag = line[1]  # .split('.')[0].replace('_', '')
        if data_type == 'primates':
            if sptag not in primate_and_anc:
                # print('sptag:', sptag)
                continue

        if data_type == 'human':
            if sptag != 'hg38':
                continue

        region_tag = line[0]

        region_label = float(line[-1])

        seq = line[2]

        # feature_vector = np.asarray([mapper[item] for item in seq], dtype=np.bool_).transpose().reshape(4, 1, 101)
        feature_vector = count_kmer(seq)
        # print('feature_vector:', feature_vector.shape); exit(0)
        # print('region_tag:', region_tag,
        #       'sptag:', sptag,
        #       'region_label:', region_label,
        #       'seq:', seq,
        #       'feature_vector:', feature_vector.shape
        #       )
        # exit(1)

        # labelled examples
        if sptag == 'hg38':
            labelled_examples[region_tag] = feature_vector
            labels[region_tag] = region_label

        # orthologs
        if mode == 'train':
            if sptag == 'hg38' and region_tag not in retain_ids:
                continue
            curr_c_id = c_id[sptag]
            # if region_tag == 'chr10:12845890-12845990':
            #     print('curr_c_id:', curr_c_id); exit(1)
            # current example is from seen orthologous region
            if region_tag in orthologs.keys():
                if curr_c_id != -1:
                    # insert child
                    # print('curr_c_id:', curr_c_id, 'sptag:', sptag); exit(1)
                    orthologs[region_tag][0][curr_c_id] = feature_vector
                    orthologs[region_tag][2][curr_c_id] += 1

                # insert parent
                if sptag in parent_to_child.keys():
                    for k, v in parent_to_child[sptag].items():
                        curr_p_id = v
                        orthologs[region_tag][1][curr_p_id] = feature_vector
                        orthologs[region_tag][2][curr_p_id] += 1
                # orthologs[region_tag]['species'] += [sptag]

            # current exmaple is a new orthologous region
            else:
                # print('new orthologous region')
                orthologs[region_tag] = {0: np.zeros(shape=(num_children,
                                                       feature_vector.shape[0],
                                                       # feature_vector.shape[1],
                                                       # feature_vector.shape[2]
                                                       ),
                                                     dtype=np.bool_
                                                ),
                                         1: np.zeros(shape=(num_children,
                                                       feature_vector.shape[0],
                                                       # feature_vector.shape[1],
                                                       # feature_vector.shape[2]
                                                       ),
                                                     dtype=np.bool_
                                                ),
                                         2: np.zeros(shape=num_children),
                                         # 'species': [sptag]
                                         }
                # print('orthologs:', orthologs)

                if curr_c_id != -1:
                    # insert into orthologs
                    orthologs[region_tag][0][curr_c_id] = feature_vector
                    orthologs[region_tag][2][curr_c_id] += 1

                if sptag in parent_to_child.keys():
                    # insert into parent array
                    for k, v in parent_to_child[sptag].items():
                        # print('k:', k, 'v:', v)
                        curr_p_id = v
                        # print('curr_p_id:', curr_p_id)
                        # print('feature_vector:', feature_vector)
                        orthologs[region_tag][1][curr_p_id] = feature_vector
                        orthologs[region_tag][2][curr_p_id] += 1
                # if region_tag == 'chr10:12845890-12845990':
                #     print('orthologs:', orthologs[region_tag][2][curr_p_id]); exit(1)
    # print('orthologs:', orthologs['chr10:5499025-5499126']); exit(0)

    # delete one sided branches i.e. either parent or child is missing
    if mode == 'train':
        mark_del = []
        for k, v in orthologs.items():
            keep_ids = np.where(v[2] == 2)[0]
            if len(keep_ids) == 0:
                mark_del.append(k)
                continue
            # orthologs[k][0] = orthologs[k][0][keep_ids, :, :, :]
            orthologs[k][0] = orthologs[k][0][keep_ids, :]
            # orthologs[k][1] = orthologs[k][1][keep_ids, :, :, :]
            orthologs[k][1] = orthologs[k][1][keep_ids, :]
            orthologs[k][3] = len(keep_ids)
            del orthologs[k][2]
            # orthologs[k]['species'] = [orthologs[k]['species'][item] for item in keep_ids ]
        # print('mark_del:', mark_del[:5]); exit(1)
        for item in mark_del:
            del orthologs[item]
    return labelled_examples, labels, orthologs





