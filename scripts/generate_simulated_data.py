import numpy as np, scipy
from multiprocessing import Pool
from scipy.stats import norm
from tqdm import tqdm


def get_sigmoid(arg):
    """
    Safe computing of sigmoid value
    :param arg:
    :return:
    """

    if arg >= 0:
        z = np.exp(-arg)
        return 1. / (1. + z)
    else:
        z = np.exp(arg)
        return z / (1 + z)


def get_activity(pt=None, activity_wt=None):
    """ Returns activity value of a pt
    -----Parameters----------
    pt: array-like, float features
    -----Returns-------------
    arg: float, activity value
    """
    if len(pt) != len(activity_wt):
        print("Pt. dimension doesn't match to desired dimension")
        print('pt dim:', len(pt), 'required dim:', len(activity_wt))
        exit(1)
    arg = get_sigmoid(np.dot(pt, activity_wt))
    # arg = np.dot(pt, activity_wt)
    return arg

def modify_pt(pt=None,):
    """ Modifies a pt in order to get non linear decision boundary
    -----Parameters----------
    pt: array-like, float features
    -----Returns-------------
    mod_rand_pt: array-like, float modified features
    """
    global dim
    mod_rand_pt = []

    for i_ in range(dim):
        for j_ in range(i_, dim):
            mod_rand_pt.append(pt[i_] * pt[j_])

    mod_rand_pt.append(1.)
    return mod_rand_pt


def generate_descendant_activity(pt):
    """ Generates a descendant for a given pt that satisfies the allowed mutations under a selection pressure
    -----Parameters-----------
    pt: array-like, float features
    -----Returns--------------
    new_pt: array-like, float features
    """
    global dim, sel, mu, br, activity_wt

    if len(pt) != dim:
        print ("generate_descendant_activity: Pt. dimension doesn't match to desired dimension")
        print ('pt:', len(pt), 'dim:', dim)
        exit(1)
    # make sure that descendants are chosen randomly under parallel computing
    scipy.random.seed()

    change_across_branch = br * mu
    norm_normalization = norm.pdf(0, 0, 1)
    while True:
        small_change = np.random.normal(0, change_across_branch, len(pt))
        new_pt = pt + np.asarray(small_change)
        delta = get_activity(modify_pt(pt), activity_wt) - get_activity(modify_pt(new_pt), activity_wt)
        pick_uniform_rv = np.random.uniform(0, 1, 1)[0]
        pdf_delta = norm.pdf(delta * sel, 0, 1) / norm_normalization

        if pick_uniform_rv < pdf_delta:
            return new_pt


def generate_bag(pt,):
    """ Generates a bag of examples
    -------Parameters---------
    pt: array-like, float features
    -------Returns------------
    bag: array-like, float
    """
    global sel, mu, br, num_nodes, activity_wt

    scipy.random.seed()
    bag = [pt]
    points = [pt]

    # generate tree with required number of nodes
    while True:
        # process the current node
        curr_pt = points.pop(0)
        curr_pt_id = curr_pt[0]
        curr_pt_sp_id = curr_pt[1]

        # generate left descendant with a certain probability
        # the value of zero means to always generate
        if np.random.random() > 0:
            left_descendant = generate_descendant_activity(curr_pt[2:-1])
            left_part = np.append(curr_pt_id, tree_lc[curr_pt_sp_id])
            left_part = np.append(left_part, np.append(left_descendant,
                                                       get_activity(modify_pt(left_descendant), activity_wt)
                                                       )
                                  )

        # generate right descendant with a certain probability
        # the value of zero means to always generate
        if np.random.random() > 0:
            right_descendant = generate_descendant_activity(curr_pt[2:-1])
            right_part = np.append(curr_pt_id, tree_rc[curr_pt_sp_id])
            right_part = np.append(right_part, np.append(right_descendant,
                                                         get_activity(modify_pt(right_descendant), activity_wt)
                                                         )
                                   )

        points.append(left_part)
        points.append(right_part)
        bag.append(left_part)
        bag.append(right_part)

        if len(bag) >= num_nodes:
            break

    return bag[:num_nodes]

def generate_tree(sp_root=0, num_nodes=7):
    """ Generates the pylogenetic tree
    -------Parameters---------
    sp_root: int, id of root species in the phylogenetic tree
    num_nodes: int, number of nodes in the phylogenetic tree
    -------Returns------------
    tree: dictionary, key is descendant id with value as its parent id
    tree_lc: dictionary, key is left descendant id with value as its parent id
    tree_rc: dictionary, key is right descendant id with value as its parent id
    """
    # generate tree
    tree = dict()
    tree_lc = dict()
    tree_rc = dict()
    tree[sp_root] = None
    queue = [sp_root]
    available_id = 0

    while True:
        curr_node = queue.pop(0)
        lc = available_id + 1
        if lc == num_nodes:
            break
        rc = available_id + 2
        if rc == num_nodes:
            break
        tree_lc[curr_node] = lc
        tree_rc[curr_node] = rc
        tree[lc] = curr_node
        tree[rc] = curr_node
        available_id += 2
        queue.append(lc)
        queue.append(rc)
        if len(tree) == num_nodes:
            break
    return tree, tree_lc, tree_rc


def generate(
             seeds=10,
             param_num_nodes=7,
             mode='train',
             param_dim=10,
             param_sel=100,
             param_mu=10,
             param_br=0.05,
             param_activity_wt=None,
             A=None,
             sp_to_id=None,
             min_coord=None,
             max_coord=None,
             org_pts=None,
             ):
    """function to simulate data points

    Parameters
    -----------
    seeds: int, number of examples
    num_nodes: int, number of orthologs per example
    dim: int, dimensionality of examples
    activity_wt: array-like, true activity weights to compute the activity value of an example
    A: array-like, adjacency matrix of the phylogenetic tree
    sp_to_id: dictionary, relates species to its ids
    min_coord: float, minimum coordinate value to generate an example
    max_coord:float, maximum coordinate value to generate an example

    Returns
    ----------
    generated_points: list, contains all the generated points
    A: array-like, adjacency matrix of the phylogenetic tree
    sp_to_id: dictionary, relates species to its ids
    activity_wt: array-like, true activity weights to compute the activity value of an example
    """
    global dim, sel, mu, br, activity_wt, tree_lc, tree_rc, num_nodes

    dim=param_dim
    sel=param_sel
    mu=param_mu
    br=param_br
    activity_wt=param_activity_wt
    num_nodes = param_num_nodes

    sp_root = 0
    tree = None

    if mode == 'train':
        tree, tree_lc, tree_rc = generate_tree(sp_root, num_nodes)
        if param_activity_wt is None:
            # weights for the linear activity function
            num_wts = int(((dim * (dim + 1))/2) + 1)
            activity_wt = np.random.normal(0, 1, num_wts)

    if org_pts is None:
        org_pts = []
        # simulate data points
        # format: exampleID, species, values
        # region, species, coord1, coord2, ...., activity_value

        for i in tqdm(range(int(seeds))):
            pt_id = i

            # pick a random point of d-dimension
            rand_pt = np.random.uniform(min_coord, max_coord, dim)
            curr_pt = np.append([pt_id, sp_root], rand_pt)
            curr_activity = get_activity(modify_pt(rand_pt), activity_wt)
            # print('curr_pt:', curr_pt, 'curr_activity:', curr_activity); exit(0)
            org_pts.append(np.append(curr_pt, curr_activity))

    generated_points = []
    full_org_pts = []

    if mode == 'train':
        pool = Pool(16)
        sample_bag = pool.map(generate_bag, org_pts)
        for item in sample_bag:
            for val in item:
                val = list(val)
                full_org_pts.append(val)
                generated_points.append(val[:2]+modify_pt(val[2:-1])+[val[-1]])
    else:
        for val in org_pts:
            val = list(val)
            generated_points.append(val[:2]+modify_pt(val[2:-1])+[val[-1]])

    return generated_points, activity_wt, org_pts, full_org_pts, tree
