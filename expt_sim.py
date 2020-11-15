import torch, torch.nn as nn, torch.nn.functional as F, torch.optim as optim
from random import shuffle
from sklearn.metrics import mean_squared_error
from scripts.generate_simulated_data import generate
import pandas as pd
import numpy as np
import argparse
from tqdm import tqdm

n_classes=1


class LR(nn.Module):
    def __init__(self, device,  IN_CH):
        super(LR, self).__init__()
        self.IN_CH = IN_CH
        self.fc = nn.Linear(self.IN_CH, n_classes, bias=False)
        self.device = device

    def forward(self, x):
        if self.device == "cuda":
            x = x.type(torch.cuda.FloatTensor)
        else:
            x.type(torch.FloatTensor)
        x = self.fc(x)
        # output layer
        x = torch.sigmoid(x)
        return x


def get_batch(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]


def train_phyloLR(train_X, train_y, orthologs,
                   model, device, optimizer,
                   batch_size,
                   beta=0.,
                   alpha=0.,
                   CTR=1.,
                  retain_ids=None
                   ):

    model.train()

    batch_train_ids = list(train_X.keys())
    shuffle(batch_train_ids)

    for counter, batch_ids in enumerate(get_batch(batch_train_ids, batch_size)):

        optimizer.zero_grad()

        # compute l2 norm
        model_weights = model.fc.weight.data
        # print('model_weights:', model_weights)
        # print('model_weights:', model_weights.shape)
        # l2_norm = torch.norm(model_weights, p=2)
        # print('l2_norm:', l2_norm)
        # print('chk mse:', (F.mse_loss(model_weights, torch.zeros_like(model_weights), size_average='sum'))**0.5)
        # exit(0)

        # compute labelled loss
        retain_batch_ids = set(batch_ids).intersection(retain_ids)
        if len(retain_batch_ids)>0:
            batch_y = torch.from_numpy(np.array([train_y[i] for i in retain_batch_ids], dtype=np.float)).to(
                device).float()
            curr_batch_X = np.array([train_X[i] for i in retain_batch_ids])
            batch_X = torch.from_numpy(curr_batch_X).to(device).float()
            output = model(batch_X).view(-1)
            loss_labelled = F.mse_loss(output, batch_y)
            # print('loss_labelled:', loss_labelled)
            # norm_loss = (alpha*l2_norm)
            # loss_labelled += norm_loss
            # print('norm_loss:', norm_loss)
            # print('loss_labelled:', loss_labelled)
        else:
            loss_labelled = 0.

        # compute ortho loss
        ortho_loss = 0.
        if (counter+1)%CTR==0:
            batch_curr_children = torch.from_numpy(np.vstack([orthologs[i][0] for i in batch_ids if i in orthologs.keys()])\
                                                   .astype(float)).to(device).float()
            batch_curr_parent = torch.from_numpy(np.vstack([orthologs[i][1] for i in batch_ids if i in orthologs.keys()]) \
                                                 .astype(float)).to(device).float()

            # num_branches
            num_branches = np.stack([orthologs[i][3] for i in batch_ids if i in orthologs.keys()])
            # compute output
            output_children = model(batch_curr_children).view(-1)
            output_parent = model(batch_curr_parent).view(-1)

            # compute loss
            given_pc_loss = (output_children - output_parent) ** 2
            ortho_loss = []
            start = 0
            for end in num_branches:
                curr_ortho_mse = torch.sum(given_pc_loss[start: start + end]) / end
                ortho_loss.append(curr_ortho_mse)
                # update start
                start += end
            ortho_loss = np.sum(ortho_loss) / len(ortho_loss)

        loss = loss_labelled + (beta*ortho_loss)
        loss.backward()
        optimizer.step()
    return


def evaluate(model=None, device=None, validate_X=None, validate_y=None, orthologs=None, beta=0., mode=False):
    val_loss_labelled=[]
    val_loss_ortho_with_beta=[]
    model.eval()
    pred = []
    with torch.no_grad():

        if mode:
            req_validate_y = []
            batch_train_ids = list(validate_X.keys())
            shuffle(batch_train_ids)
            batch_size=128
            for counter, batch_ids in enumerate(get_batch(batch_train_ids, batch_size)):

                batch_y = np.array([validate_y[i] for i in batch_ids], dtype=np.float)
                # compute labelled loss
                curr_batch_X = np.array([validate_X[i] for i in batch_ids])
                batch_X = torch.from_numpy(curr_batch_X).to(device).float()
                output = model(batch_X).view(-1).cpu().detach().numpy()
                val_loss_labelled.append(mean_squared_error(batch_y, output))
                pred = np.append(pred, output)
                req_validate_y = np.append(req_validate_y, batch_y)

                # compute ortho loss
                batch_curr_children = torch.from_numpy(
                    np.vstack([orthologs[i][0] for i in batch_ids if i in orthologs.keys()]) \
                    .astype(float)).to(device).float()
                batch_curr_parent = torch.from_numpy(
                    np.vstack([orthologs[i][1] for i in batch_ids if i in orthologs.keys()]) \
                    .astype(float)).to(device).float()
                # num_branches
                num_branches = np.stack([orthologs[i][3] for i in batch_ids if i in orthologs.keys()])
                # compute output
                output_children = model(batch_curr_children).view(-1)
                output_parent = model(batch_curr_parent).view(-1)

                # compute loss
                given_pc_loss = (output_children - output_parent) ** 2
                ortho_loss = []
                start = 0
                for end in num_branches:
                    curr_ortho_mse = torch.sum(given_pc_loss[start: start + end]) / end
                    ortho_loss.append(curr_ortho_mse)
                    # update start
                    start += end
                ortho_loss = np.sum(ortho_loss) / len(ortho_loss)

                val_loss_ortho_with_beta.append(beta*ortho_loss)


            # compute losses
            val_loss_labelled = sum(val_loss_labelled)/float(len(val_loss_labelled))
            val_loss_ortho_with_beta = sum(val_loss_ortho_with_beta)/float(len(val_loss_ortho_with_beta))
            val_loss_ortho_with_beta = val_loss_ortho_with_beta.item()

            val_total_loss = val_loss_labelled+val_loss_ortho_with_beta

            val_mse = mean_squared_error(req_validate_y, pred)

        else:
            for counter, batch_ids in enumerate(get_batch(range(validate_X.shape[0]), 50000)):
                batch_X = torch.from_numpy(validate_X[batch_ids, :]).to(device).float()
                pred_X = model(batch_X).view(-1)
                pred = np.append(pred, pred_X.cpu().detach().numpy())
            val_loss_labelled = mean_squared_error(validate_y, pred)
            val_total_loss = val_loss_labelled
            val_mse = mean_squared_error(validate_y, pred)
    return val_mse, val_loss_labelled, val_loss_ortho_with_beta, val_total_loss, pred


def run_phyloLR(train_X, train_y, orthologs,
                 val_X, val_y,
                 test_X, test_y,
                 batch_size,
                 lr,
                 epochs,
                 model_pth,
                 pred_pth,
                 model_verify,
                 IN_CH,
                 beta,
                 alpha,
                 patience,
                 CTR=1.,
                 retain_ids=None
                 ):

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print('device:', device)

    model = LR(device, IN_CH).to(device)

    if use_cuda:
        # print('check')
        if torch.cuda.device_count() > 1:
            print("We use", torch.cuda.device_count(), "GPUs!")
            model = nn.DataParallel(model)
        model.to(device)

    # Train
    adam_lr = lr

    optimizer = optim.Adam(model.parameters(),
                           lr=adam_lr,
                           weight_decay=alpha
                           )

    curr_best_validate_loss_labelled = np.inf; best_tr_epoch=0
    wait = 0
    for tr_epoch in range(epochs):
        train_phyloLR(train_X, train_y, orthologs, model, device, optimizer, batch_size, beta, alpha, CTR,
                      retain_ids)

        # validation
        curr_validate_mse, curr_validate_loss_labelled, _, _, _ = evaluate(model=model,
                                                                     device=device,
                                                                     validate_X=val_X,
                                                                     validate_y=val_y,
                                                                     )

        wait += 1

        # save best model
        if curr_best_validate_loss_labelled > curr_validate_loss_labelled:
            # reset wait
            wait = 0
            curr_best_validate_loss_labelled = curr_validate_loss_labelled
            best_tr_epoch = tr_epoch
            # save the model
            torch.save(model, model_pth)
        if wait >= patience:
            print('Did not improve for', patience, 'epochs')
            break


    # load the best model
    best_model = torch.load(model_pth)

    best_test_mse, _, _, _, _ = evaluate(
        model=best_model,
        device=device,
        validate_X=test_X,
        validate_y=test_y,
    )

    return best_test_mse, best_tr_epoch

def processData(data, mode='train', c_id=None, parent_to_child=None):

    num_children = len(c_id)
    print('num_children:', num_children)
    labelled_id = num_children-1
    print('labelled_id:', labelled_id)

    orthologs = dict()
    labels = dict()
    labelled_examples = dict()

    for line_index in tqdm(range(data.shape[0])):

        region_tag = data[line_index, 0]

        sptag = data[line_index, 1]

        region_label = data[line_index, -1]

        feature_vector = data[line_index, 2:-1]


        # labelled examples
        if sptag == labelled_id:
            labelled_examples[region_tag] = feature_vector
            labels[region_tag] = region_label

        # orthologs
        if mode == 'train':
            curr_c_id = c_id[sptag]
            if region_tag in orthologs.keys():
                if curr_c_id != -1:
                    # insert child
                    orthologs[region_tag][0][curr_c_id] = feature_vector
                    orthologs[region_tag][2][curr_c_id] += 1

                # insert parent
                if sptag in parent_to_child.keys():
                    for k, v in parent_to_child[sptag].items():
                        curr_p_id = v
                        orthologs[region_tag][1][curr_p_id] = feature_vector
                        orthologs[region_tag][2][curr_p_id] += 1

            # current exmaple is a new orthologous region
            else:
                orthologs[region_tag] = {0: np.zeros(shape=(num_children,
                                                       feature_vector.shape[0],
                                                       ),
                                                     dtype=np.float
                                                ),
                                         1: np.zeros(shape=(num_children,
                                                       feature_vector.shape[0],
                                                       ),
                                                     dtype=np.float
                                                ),
                                         2: np.zeros(shape=num_children)
                                         }

                if curr_c_id != -1:
                    # insert into orthologs
                    orthologs[region_tag][0][curr_c_id] = feature_vector
                    orthologs[region_tag][2][curr_c_id] += 1

                if sptag in parent_to_child.keys():
                    # insert into parent array
                    for k, v in parent_to_child[sptag].items():
                        curr_p_id = v
                        orthologs[region_tag][1][curr_p_id] = feature_vector
                        orthologs[region_tag][2][curr_p_id] += 1

    # delete one sided branches i.e. either parent or child is missing
    if mode == 'train':
        mark_del = []
        for k, v in orthologs.items():
            retain_ids = np.where(v[2] == 2)[0]
            if len(retain_ids) == 0:
                mark_del.append(k)
                continue
            orthologs[k][0] = orthologs[k][0][retain_ids, :]
            orthologs[k][1] = orthologs[k][1][retain_ids, :]
            orthologs[k][3] = len(retain_ids)
        for item in mark_del:
            del orthologs[item]
    return labelled_examples, labels, orthologs


def parse_args():
    parser = argparse.ArgumentParser(description="List of commands.")

    parser.add_argument('--dim', type=int, default=2, metavar='D',
                        help='euclidean space dimensionality. Default: 2')
    parser.add_argument('--num-examples', type=int, default=25, metavar='N',
                        help='Number of roots. Default: 25')
    parser.add_argument('--sel', type=float, default=1000., metavar='S',
                        help='Selection coefficient. Default: 1000.')
    parser.add_argument('--num-nodes', type=int, default=15, metavar='O',
                        help='number of nodes per tree. Default: 15')
    parser.add_argument('--min-coord', type=float, default=-1., metavar='MM',
                        help='minimum coordinate. Default: -1.')
    parser.add_argument('--max-coord', type=float, default=1., metavar='MX',
                        help='maximum coordinate. Default: 1.')
    parser.add_argument('--repeat', type=int, default=1, metavar='R',
                        help='number of times to repeat the experiment. Default: 1')
    parser.add_argument('--patience', type=int, default=20, metavar='P',
                        help='number of epochs to wait for no improvement. Default: 20')
    parser.add_argument('--epochs', type=int, default=10000, metavar='E',
                        help='number of epochs to train. Default: 10000')
    parser.add_argument('--tag', type=str, default='1', metavar='T',
                        help='tag id of experiments. Default: 1')
    parser.add_argument('--beta', type=float, default=0., metavar='B',
                        help='Beta. Default: 0.')
    parser.add_argument('--alpha', type=float, default=0., metavar='A',
                        help='Alpha, coefficient of l2norm. Default: 0.')
    parser.add_argument('--retain', type=float, default=1., metavar='R',
                        help='fraction of labelled examples to use. Default: 1.')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='Learning Rate. Default: 0.001')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    root_id=0

    print('dim:', args.dim)
    print('num_examples:', args.num_examples)
    print('sel:', args.sel)
    print('num_nodes:', args.num_nodes)
    print('repeat:', args.repeat)
    print('epochs:', args.epochs)
    print('patience:', args.patience)
    print('tag:', args.tag)
    print('retain:', args.retain)
    print('lr:', args.lr)
    print('alpha:', args.alpha)

    retain_ids = set(pd.DataFrame(list(range(args.num_examples)))[0].sample(frac=args.retain).values)
    print('retain_ids:', len(retain_ids))

    best_tr_epoch_dict = {}
    beta_list = [0.] #, 0.1, 1., 10., 100., 1000.]
    med_mse = {}
    # for item in beta_list:
    for item in beta_list:
        med_mse[item] = []
        best_tr_epoch_dict[item] = []

    for i in range(args.repeat):

        # generate train
        train_set, activity_wt, _, org_train_set, tree = generate(seeds=args.num_examples,
                                                               param_dim=args.dim,
                                                               param_num_nodes=args.num_nodes,
                                                               param_sel=args.sel,
                                                               min_coord=args.min_coord,
                                                               max_coord=args.max_coord,
                                                               mode='train'
                                                               )
        tag = 'tag-'+args.tag+'-dim-'+str(args.dim)+'-num-examples-'+str(args.num_examples)\
              + '-sel-'+str(args.sel)+'-num-nodes-'+str(args.num_nodes)+'-retain-'+str(args.retain)

        # generate validation
        val_set, _, org_val_set, _, _ = generate(seeds=2000,
                                              param_dim=args.dim,
                                              param_num_nodes=1.,
                                              param_sel=0.,
                                              min_coord=args.min_coord,
                                              max_coord=args.max_coord,
                                              param_activity_wt=activity_wt,
                                              mode='validation'
                                              )

        # generate test
        test_set, _, org_test_set, _, _ = generate(seeds=10000,
                                                param_dim=args.dim,
                                                param_num_nodes=1.,
                                                param_sel=0.,
                                                min_coord=args.min_coord,
                                                max_coord=args.max_coord,
                                                param_activity_wt=activity_wt,
                                                mode='test'
                                                )

        # create validation
        val_set = np.array(val_set)
        val_X = val_set[:, 2:-1]; val_y = val_set[:, -1]
        print('val_X:', val_X.shape, 'val_y:', val_y.shape)

        # create test
        test_set = np.array(test_set)
        test_X = test_set[:, 2:-1]; test_y = test_set[:, -1]
        print('test_X:', test_X.shape, 'test_y:', test_y.shape)

        # create train
        c_id = dict()
        parent_to_child = dict()
        counter = 0
        for k, v in tree.items():
            if k == root_id:
                continue
            c_id[k] = counter
            parent = v
            if parent not in parent_to_child.keys():
                parent_to_child[parent] = {k: counter}
            else:
                parent_to_child[parent][k] = counter
            counter += 1

        # declare root_id as root
        c_id[root_id] = -1

        train_set = np.array(train_set)
        train_X, train_y, orthologs = processData(train_set,
                                                  'train',
                                                  c_id,
                                                  parent_to_child
                                                  )
        train_size = len(train_X)

        print('train size:', train_size)

        # run phylo lr
        # for beta in beta_list:
        for beta in beta_list:
            model_pth = 'm-'+tag+'-beta-'+str(beta)
            pred_pth = 'p-'+tag+'-beta-'+str(beta)
            alpha = args.alpha if beta == 0 else 0
            test_mse, best_tr_epoch = run_phyloLR(train_X, train_y, orthologs,
                                   val_X, val_y, test_X, test_y,
                                   batch_size=128, lr=args.lr, epochs=args.epochs,
                                   model_pth=model_pth, pred_pth=pred_pth,
                                   model_verify=False,
                                   IN_CH=len(activity_wt), beta=beta, alpha=alpha,
                                   patience=args.patience,
                                   retain_ids=retain_ids,
                                   )
            med_mse[beta].append(test_mse)
            best_tr_epoch_dict[beta].append(best_tr_epoch)
            print('PhyloLR i:', i, 'alpha:', alpha, 'beta:', beta, 'Test MSE:', test_mse, 'Best_tr_epoch:', best_tr_epoch)

    # for beta in beta_list:
    for beta in beta_list:
        alpha = args.alpha if beta == 0 else 0
        print('Result tag:', args.tag,
              'dim:', args.dim,
              'beta:', beta,
              'alpha:', alpha,
              'num_examples:', args.num_examples,
              'sel:', args.sel,
              'num_nodes:', args.num_nodes,
              'best_tr_epoch:', np.median(best_tr_epoch_dict[beta]),
              'median Test MSE:', np.median(med_mse[beta])
              )




