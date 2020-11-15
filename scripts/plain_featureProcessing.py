"""creates feature of sequence files
Input: seqfile {0: regionTag, 1: spTag, 2: sequence, 4: seqLength
Output: Labelled example feature matrix, list of labelled species,
        corresponding orthologs with their species name {speciesList, featurevectorList},
        labels

(c) Faizy Ahsan
email: zaifyahsan@gmail.com"""


import numpy as np
from tqdm import tqdm


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

    num_children = len(c_id)
    max_len = 1000
    print('num_children:', num_children)
    # mapper = {'A': [1., 0., 0., 0.],
    #           'C': [0., 1., 0., 0.],
    #           'G': [0., 0., 1., 0.],
    #           'T': [0., 0., 0., 1.],
    #           }
    mapper = {'A': [True, False, False, False],
              'C': [False, True, False, False],
              'G': [False, False, True, False],
              'T': [False, False, False, True],
              'P': [False, False, False, False],
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

        region_tag = line[0]

        # region_label = float(line[-1])
        region_label = 1. if '-1-chr' in region_tag else 0.

        seq = line[2]

        if len(seq)<500:
            continue

        if len(seq)<max_len:
            req_pad = max_len-len(seq)
            seq += 'P'*req_pad
        if len(seq)>max_len:
            seq = seq[:max_len]

        feature_vector = np.asarray([mapper[item] for item in seq],
                                    dtype=np.bool_).transpose().reshape(4, 1, max_len)

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
            # if region_tag == 'chr10:12845890-12845990':
            #     print('curr_c_id:', curr_c_id); exit(1)
            # current example is from seen orthologous region
            if region_tag in orthologs.keys():
                if sptag != 'hg38':
                    curr_c_id = c_id[sptag]
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
                orthologs[region_tag]['species'] += [sptag]

            # current exmaple is a new orthologous region
            else:
                # print('new orthologous region')
                orthologs[region_tag] = {0: np.zeros(shape=(num_children,
                                                       feature_vector.shape[0],
                                                       feature_vector.shape[1],
                                                       feature_vector.shape[2]
                                                       ),
                                                     dtype=np.bool_
                                                ),
                                         1: np.zeros(shape=(num_children,
                                                       feature_vector.shape[0],
                                                       feature_vector.shape[1],
                                                       feature_vector.shape[2]
                                                       ),
                                                     dtype=np.bool_
                                                ),
                                         2: np.zeros(shape=num_children),
                                         'species': [sptag]
                                         }
                # print('orthologs:', orthologs)

                if sptag != 'hg38':
                    curr_c_id = c_id[sptag]
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
            orthologs[k][0] = orthologs[k][0][keep_ids, :, :, :]
            orthologs[k][1] = orthologs[k][1][keep_ids, :, :, :]
            orthologs[k][3] = len(keep_ids)
            del orthologs[k][2]
            # orthologs[k]['species'] = [orthologs[k]['species'][item] for item in keep_ids ]
        # print('mark_del:', mark_del[:5]); exit(1)
        for item in mark_del:
            del orthologs[item]
    return labelled_examples, labels, orthologs





