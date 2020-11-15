import torch
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
from collections import defaultdict
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
import pickle
import sys, random
import torch.nn.functional as F
from tqdm import tqdm

if len(sys.argv) < 10:
    print('python file.py input_train input_val input_test batch_size epochs model_pth beta ITR pred_file')
    exit(0)

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

input_train = sys.argv[1]
input_val = sys.argv[2]
input_test = sys.argv[3]
BATCH_SIZE = int(sys.argv[4])
EPOCHS = int(sys.argv[5])
model_pth = sys.argv[6]
beta = float(sys.argv[7])
ITR = int(sys.argv[8])
pred_file = sys.argv[9]


class SequenceDataset(Dataset):
    def __init__(self, features, rev_comp_features, targets):
        self.features = features
        self.rev_comp_features = rev_comp_features
        self.targets = targets
        # print('features:',features.shape, features[0].shape, type(features), type(features[0]))
        # print('targets:', targets.shape); exit(0)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, item):
        review = self.features[item]
        review_rev_comp = self.rev_comp_features[item]
        target = self.targets[item]

        return {
            'review_text': torch.tensor(review, dtype=torch.float),
            'review_text_rev_comp': torch.tensor(review_rev_comp, dtype=torch.float),
            'targets': torch.tensor(target, dtype=torch.long)
        }


class OrthoSequenceDataset(Dataset):
    def __init__(self, orthologs):
        self.orthologs = orthologs

    def __len__(self):
        return len(self.orthologs)

    def __getitem__(self, item):
        sequences = self.orthologs[item]['seq']

        label = self.orthologs[item]['label']

        label_id = self.orthologs[item]['id'] if 'id' in self.orthologs[item].keys() else -1

        return {
            'sequences': sequences,
            'species': self.orthologs[item]['species'],
            'label': torch.tensor(label, dtype=torch.long),
            'id': label_id
        }


# read dataset
mapper = {'A': [True, False, False, False],
              'C': [False, True, False, False],
              'G': [False, False, True, False],
              'T': [False, False, False, True],
              'P': [False, False, False, False]
              }

rev_comp_mapper = {'A': 'T',
                   'C': 'G',
                   'G': 'C',
                   'T': 'A'
                   }

def get_df(fname, species='hg38', mode=''):

    df = pd.read_csv(fname, sep=',', header=None, )
    print('df:', df.head())
    print('df:', df.shape)

    if species == 'hg38':
        df = df[df[1]=='hg38']
    print('df:', df.shape)

    df['label'] = df[0].apply(lambda x: 1 if '-1-chr' in x else 0,
                                      axis=1)

    num_pos = df[df['label']==1].shape[0]
    num_neg = df[df['label']==0].shape[0]

    print('num_pos:', num_pos, 'num_neg:', num_neg)

    if mode == 'train':
        df_pos = df[df['label']==1]
        df_neg = df[df['label']==0]
        print('df_pos:', df_pos.shape, 'df_neg:', df_neg.shape)
        if num_pos>num_neg:
            df_pos = df_pos.sample(n=num_neg, random_state=42)
            print('df_pos:', df_pos.shape, 'df_neg:', df_neg.shape)

            df = pd.concat([df_pos, df_neg]).sample(frac=1)
        else:
            df_neg = df_neg.sample(n=num_pos, random_state=42)
            print('df_pos:', df_pos.shape, 'df_neg:', df_neg.shape)

            df = pd.concat([df_pos, df_neg]).sample(frac=1)
        print('df:', df.shape)
        # exit(0)


    # fix length to max len
    max_len = 1000

    df['fixed_len_seq'] = df[2]. \
        apply(lambda x: x + ('P' * (1000 - len(x))) if len(x) <= 1000 else x[:1000])

    # get features
    df['features'] = df['fixed_len_seq']. \
        apply(lambda row: np.array([mapper[item] for item in row], dtype=np.bool_).reshape(-1, 4).T)

    # get rev comp
    df['rev_comp'] = df[2]. \
        apply(lambda x: x[::-1])

    df['rev_comp'] = df['rev_comp']. \
        apply(lambda x: ''.join([rev_comp_mapper[item] for item in x]))

    df['rev_comp_fixed_len_seq'] = df['rev_comp']. \
        apply(lambda x: x + ('P' * (1000 - len(x))) if len(x) <= 1000 else x[:1000])

    df['rev_comp_features'] = df['rev_comp_fixed_len_seq']. \
        apply(lambda row: np.array([mapper[item] for item in row], dtype=np.bool_).reshape(-1, 4).T)

    return df

# train_df = get_df(input_train, mode='train')
train_df = get_df(input_train)
val_df = get_df(input_val)
# test_df = get_df(input_test)

print('train:', train_df.shape,
      'val_df:', val_df.shape,
      # 'test_df:', test_df.shape
      )

def create_data_loader(df, batch_size):
    # print('ok1')
    ds = SequenceDataset(
        # features=df.features.to_numpy(),
        features=np.stack(df['features'].values),
        rev_comp_features=np.stack(df['rev_comp_features'].values),
        targets=df.label.to_numpy(),
    )
    # print('ok2'); exit(0)
    return DataLoader(
        ds,
        batch_size=batch_size,
        num_workers=16,
        shuffle=True
    )


def create_ortho_data_loader(orthologs, batch_size):
    # print('ok1')
    ds = OrthoSequenceDataset(orthologs
                              )
    # print('ok2'); exit(0)
    return DataLoader(
        ds,
        # batch_size=batch_size,
        batch_size=1,
        num_workers=0,
        shuffle=True
    )


def process_ortho(fname):
    prev_region = ''
    orthologs = {}
    labelled_examples = {}
    # lines = [line.strip() for line in open(fname).readlines() if line.split(',')[0].split(':')[2] not in val_chr_list]
    lines = open(fname).readlines()
    print('lines:', len(lines))

    ortho_index = -1
    ortho_item = -1
    for line_index in tqdm(range(len(lines))):
        # line = lines[line_index].strip()
        line = lines[line_index].strip()
        line = line.split(',')

        region = line[0]
        species = line[1]
        seq = line[2]
        if len(seq) <= 500:
            continue

        label = 1 if '-1-chr' in region else 0

        if region == prev_region:
            orthologs[ortho_index]['seq'].append(seq)
            orthologs[ortho_index]['species'].append(species)

            ortho_item += 1
            if species == 'hg38':
                orthologs[ortho_index]['id'] = ortho_item

        else:
            prev_region = region
            ortho_index += 1
            ortho_item = 0

            orthologs[ortho_index] = {'seq': [seq],
                                      'species': [species],
                                      'label': label
                                      }
            if species == 'hg38':
                orthologs[ortho_index]['id'] = ortho_item
    return orthologs


child_id = pickle.load(open('../child_id.pkl', 'rb'))
parent_id = pickle.load(open('../parent_id.pkl', 'rb'))
hg38_id = parent_id['hg38']

orthologs = process_ortho(input_train)

train_data_loader = create_data_loader(train_df, BATCH_SIZE)
ortho_data_loader = create_ortho_data_loader(orthologs, BATCH_SIZE)
# pos_train_data_loader = create_data_loader(df_train_pos, BATCH_SIZE//2)
# neg_train_data_loader = create_data_loader(df_train_neg, BATCH_SIZE//2)
val_data_loader = create_data_loader(val_df, BATCH_SIZE)
# test_data_loader = create_data_loader(test_df, BATCH_SIZE)


class FactorNet(nn.Module):

    def __init__(self, device):
        super(FactorNet, self).__init__()
        self.device = device

        # self.OUT_CH = OUT_CH
        # self.FC = FC

        # conv layer 1
        self.conv1 = nn.Conv1d(in_channels=4, out_channels=32, stride=1, kernel_size=26,
                               padding=0)
        self.relu1 = nn.ReLU(inplace=False)
        # define dropout layer in __init__
        self.drop_layer1 = nn.Dropout(p=0.1)

        # pooling layer 1
        self.max_pool = nn.MaxPool1d(kernel_size=13, stride=13)

        # bidirectional lstm layer 1
        self.lstm = nn.LSTM(input_size=75,
                            hidden_size=32,
                            num_layers=1,
                            bidirectional=True,
                            batch_first=True
                            )
        self.drop_layer2 = nn.Dropout(p=0.5)

        # fully connected layer 1

        self.fc1 = nn.Linear(32 * 64, 128)
        self.relu2 = nn.ReLU(inplace=False)
        self.drop_layer3 = nn.Dropout(p=0.5)

        self.fc2 = nn.Linear(128, 1)

        self.sigmoid = nn.Sigmoid()

    def forward_one(self, x):
        # if self.device == "cuda":
        #   x = x.type(torch.cuda.FloatTensor)
        # else:
        #   x.type(torch.FloatTensor)
        # print('x:', x.shape)
        x = self.conv1(x)
        # print('after conv1d x:', x.shape)
        x = self.relu1(x)
        x = self.drop_layer1(x)
        x = self.max_pool(x)
        # print('after max pool x:', x.shape)

        x, (hn, cn) = self.lstm(x)
        # print('x:', x.shape)
        x = self.drop_layer2(x)
        # print('x:', x.shape)

        x = x.reshape(-1, 32 * 64)
        # print('x:', x.shape)

        x = self.fc1(x)
        # print('x:', x.shape)
        x = self.relu2(x)
        x = self.drop_layer3(x)

        x = self.fc2(x)
        # print('x:', x.shape)

        x = self.sigmoid(x)
        # print('x:', x.shape)

        return x

    def forward(self, input1, input2):
        output1 = self.forward_one(input1)  # .reshape(-1)
        output2 = self.forward_one(input2)  # .reshape(-1)
        # print('output1:', output1)
        # print('output2:', output2)
        cat_output = torch.cat((output1, output2), 1)
        # print('cat_output:', cat_output)
        out = torch.mean(cat_output, dim=1)
        # print(out); exit(0)

        return out


class_names = [0, 1]
OUT_CH = 128
FC = 32
DROPOUT = 0
n_classes = 2
model = FactorNet(device=device, )
model = model.to(device)

scheduler=''
loss_fn=''
optimizer = torch.optim.Adam(model.parameters())


def get_features(sequences, sp='hg38'):
    ids = {}
    max_len = 1000

    # compute rev comp seq
    rev_comp_sequences = [''.join([rev_comp_mapper[item] for item in seq[0][::-1]]) for seq in sequences]

    # pad sequences
    padded_seq = [seq[0] + ('P' * (max_len - len(seq[0]))) \
                      if len(seq[0]) <= max_len \
                      else seq[0][:max_len] \
                  for seq in sequences]
    padded_rev_comp_seq = [rev_comp_seq + ('P' * (max_len - len(rev_comp_seq))) \
                               if len(rev_comp_seq) <= max_len \
                               else rev_comp_seq[:max_len] \
                           for rev_comp_seq in rev_comp_sequences]
    # features
    feature = np.stack([np.array([mapper[item] for item in padded_seq_item], dtype=np.bool_).reshape(-1, 4).T \
                        for padded_seq_item in padded_seq])
    feature_rev_comp = np.stack(
        [np.array([mapper[item] for item in padded_rev_comp_seq_item], dtype=np.bool_).reshape(-1, 4).T \
         for padded_rev_comp_seq_item in padded_rev_comp_seq])

    if sp != 'hg38':
        c_id = [];
        c_id_map = []
        p_id = [];
        p_id_map = []
        # sp = orthologs[item]['species']
        for ortho_item, species in enumerate(sp):
            species = species[0]
            if species != 'hg38':
                c_id.append(child_id[species])
                c_id_map.append(ortho_item)
            if species in parent_id.keys():
                for p_item in parent_id[species]:
                    p_id.append(p_item)
                    p_id_map.append(ortho_item)
        ids['c_id'] = c_id;
        ids['c_id_map'] = c_id_map;
        ids['p_id'] = p_id;
        ids['p_id_map'] = p_id_map

    return torch.tensor(feature, dtype=torch.float), torch.tensor(feature_rev_comp, dtype=torch.float), ids


def train_epoch(
        model,
        data_loader,
        ortho_data_loader,
        loss_fn,
        optimizer,
        device,
        scheduler,
        n_examples,
        epoch,
        O_CTR,
        run_label
):
    model = model.train()

    # if (epoch + 1) % ITR != 0:
    if run_label:
        print('running labelled loss')
        losses = []
        correct_predictions = 0
        for d in data_loader:
            # input_ids = d["input_ids"].to(device)
            # attention_mask = d["attention_mask"].to(device)
            features = d['review_text'].to(device)
            rev_comp_features = d['review_text_rev_comp'].to(device)
            targets = d["targets"].to(device)
            outputs = model(features, rev_comp_features).reshape(-1)
            preds = (outputs >= 0.5).float() * 1
            # use factornet loss function
            loss = F.binary_cross_entropy(outputs, targets.float())

            correct_predictions += torch.sum(preds == targets)
            losses.append(loss.item())
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            # scheduler.step()
            optimizer.zero_grad()
        return correct_predictions.double() / n_examples, np.mean(losses)
    else:
        print('running ortho loss')
        losses = []
        correct_predictions = 0

        ortho_losses = torch.tensor([]).to(device)
        ctr = 0
        batch_ctr = 1
        print('O_CTR:', O_CTR)
        for di, d in enumerate(ortho_data_loader):

            if (di + 1) % BATCH_SIZE == 0:
                batch_ctr += 1


            if batch_ctr % O_CTR == 0:

                # batch_start = time.time()
                sequences = d['sequences']
                species = d['species']
                # label_id = d['id']

                feature, feature_rev_comp, ids = get_features(sequences, species)
                c_id = ids['c_id']
                c_id_map = ids['c_id_map']
                p_id = ids['p_id']
                p_id_map = ids['p_id_map']

                # print('c_id:', c_id)
                # print('c_id_map:', c_id_map)
                # print('p_id:', p_id)
                # print('p_id_map:', p_id_map)

                outputs = model(feature.to(device), feature_rev_comp.to(device))

                outs = -1. * torch.ones([114, 2], dtype=torch.float, device=device)
                # if c_id.shape[1]>0:
                if len(c_id) > 0:
                    # print('c_id:', c_id, 'c_id_map:', c_id_map)
                    outs[c_id, 1] = outputs[c_id_map]
                # if p_id.shape[1]>0:
                if len(p_id) > 0:
                    # print('p_id:', p_id, 'p_id_map:', p_id_map)
                    outs[p_id, 0] = outputs[p_id_map]

                # print('outs:', outs, outs.shape)
                outs = outs[outs[:, 0] != -1.]
                # print('outs:', outs, outs.shape)
                outs = outs[outs[:, 1] != -1.]
                # print('outs:', outs, outs.shape)
                # exit(0)

                if outs.shape[0] > 0:
                    ortho_loss = (outs[:, 0] - outs[:, 1]) ** 2
                    ortho_loss = torch.mean(ortho_loss, dim=0)

                    ortho_losses = torch.cat((ortho_losses, ortho_loss.reshape(-1)), dim=0)
                    # print('ortho_losses:', len(ortho_losses)); exit(0)
                    # print('ortho_loss:', ortho_loss)

            # exit(0)

            # ctr += 1
            # if ctr % BATCH_SIZE == 0:
            if batch_ctr % O_CTR == 1 and len(ortho_losses) > 1:

                loss = beta * torch.mean(ortho_losses, dim=0)
                losses.append(loss.item())
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                ortho_losses = torch.tensor([]).to(device)

                ctr += 1

            if ctr+1 % 100 == 0:
                break
        return 0, np.mean(losses)


def eval_model(model, data_loader, loss_fn, device, n_examples):
    model = model.eval()
    losses = []
    correct_predictions = 0
    with torch.no_grad():
        for d in data_loader:
            features = d['review_text'].to(device)
            rev_comp_features = d['review_text_rev_comp'].to(device)
            targets = d["targets"].to(device)

            outputs = model(features, rev_comp_features).reshape(-1)
            preds = (outputs >= 0.5).float() * 1
            loss = F.binary_cross_entropy(outputs, targets.float())

            correct_predictions += torch.sum(preds == targets)

            losses.append(loss.item())
    return correct_predictions.double() / n_examples, np.mean(losses)


# %%time
history = defaultdict(list)
best_accuracy = 0
best_epoch=0
best_loss = np.inf
toggler = 0; flag = True
for epoch in range(EPOCHS):
    O_CTR = random.randint(50, 150)
    print('O_CTR:', O_CTR)
    print(f'Epoch {epoch + 1}/{EPOCHS}')
    print('-' * 10)

    if toggler == 2*ITR:
        toggler = 0

    if toggler < ITR:
        run_label = True
    else:
        run_label=False

    toggler += 1

    train_acc, train_loss = train_epoch(
        model,
        train_data_loader,
        ortho_data_loader,
        loss_fn,
        optimizer,
        device,
        scheduler,
        len(train_df),
        epoch,
        O_CTR,
        run_label
    )
    print(f'Train loss {train_loss} accuracy {train_acc}')
    val_acc, val_loss = eval_model(
        model,
        val_data_loader,
        loss_fn,
        device,
        len(val_df)
    )
    print(f'Val   loss {val_loss} accuracy {val_acc}')
    print()
    history['train_acc'].append(train_acc)
    history['train_loss'].append(train_loss)
    history['val_acc'].append(val_acc)
    history['val_loss'].append(val_loss)
    if val_acc > best_accuracy:
    # if val_loss < best_loss:
        # torch.save(model.state_dict(), 'best_model_state.bin')
        torch.save(model.state_dict(), model_pth)
        best_accuracy = val_acc
        best_epoch=epoch
        best_loss = val_loss

print('best_epoch:', best_epoch, 'best_accuracy:', best_accuracy, 'best_loss:', best_loss)

