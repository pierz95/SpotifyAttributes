from random import shuffle
from tabnanny import verbose
import torch
# from torch.Tensor import scatter_
from torch.utils.data import Dataset
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch_scatter import scatter_add
from torch_scatter import scatter_mean
from torch_scatter import scatter_max
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import f1_score, mean_squared_error, accuracy_score
from sklearn.utils.class_weight import compute_class_weight

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

# set the seed
torch.manual_seed(123)

def exec_deepset(config, df, Xf, yf, train_id, val_id, test_id, exec_name, device):
    # ==== Construct the various combination of layers for phi e rho =================
    phi_rho_layers = []

    # retrieve the global variables from the config file
    max_phi_depth = config['max_phi_depth']
    max_rho_depth = config['max_rho_depth']

    # compute the number of classes
    n_classes = len(set(df[yf].tolist()))

    # iterate over the constructors
    for i in range(max_phi_depth):
        # initialize the phi layer.
        # the first dim MUST be equal to the input dim
        phi = [len(Xf)]
        for ii in range(i + 1):
            if ii == 0:  # first iteration: expand
                phi.append(phi[-1] * 2)
            else:  # compress
                phi.append(phi[-1] // 2)

        for j in range(max_rho_depth):
            # construct rho
            # the first dim is equal to 4 times last phi layer
            # this because we adopt 4 summary statistics
            rho = [phi[-1] * 3]
            for jj in range(j):
                rho.append(max(rho[-1] // 2, n_classes))

            # we include this iteration only if the last layer is valid
            if rho[-1] != 0:
                # add the output layer -- constrainted to the n_classes
                # note that in the reegression task, n_classes = 1
                rho.append(n_classes)

                # update the overall structure
                phi_rho_layers.append((phi, rho))

    # ========================== training phase ====================================
    best_val_loss = np.inf
    best_val_measure = -1
    log = pd.DataFrame(None, columns=['rho', 'phi', 'lr', 'activation', 'val_loss', 'train_f1', 'val_f1', 'test_f1'])
    for x, y in phi_rho_layers:
        for lr in config['lr']:
            for activation in config['activation']:
                curr_val_loss, curr_train_m, curr_val_m, curr_test_m, curr_hist_train, curr_hist_val, curr_hist_test = train_deepset(
                    config, df, Xf, yf, train_id, val_id, test_id, x, y, lr, activation, device)

                # # save the best new setting, if founded -- f1 measure
                if curr_val_m > best_val_measure:
                    print("NEW BEST MODEL")
                    best_train_measure = curr_train_m
                    best_val_measure = curr_val_m
                    # best_val_loss = curr_val_loss
                    best_test_measure = curr_test_m
                    best_hist_train = curr_hist_train
                    best_hist_val = curr_hist_val
                    best_hist_test = curr_hist_test

                # # save the best new setting, if founded -- loss
                # if curr_val_loss < best_val_loss:
                #     best_train_measure = curr_train_m
                #     best_val_measure = curr_val_m
                #     best_val_loss = curr_val_loss
                #     best_test_measure = curr_test_m
                #     best_hist_train = curr_hist_train
                #     best_hist_val = curr_hist_val
                #     best_hist_test = curr_hist_test

                # save the log
                log.loc[len(log)] = [x, y, lr, activation, curr_val_loss, curr_train_m, curr_val_m, curr_test_m]

    # ========================== save settings ====================================
    log.to_csv(f'{config["log_dir"]}{config["attribute"]}_deepset_{exec_name}.csv')

    # build the loss plot
    loss_df = pd.DataFrame(None)
    loss_df['epoch'] = list(range(len(best_hist_train))) + list(range(len(best_hist_val))) + list(
        range(len(best_hist_test)))
    loss_df['loss'] = best_hist_train + best_hist_val + best_hist_test
    loss_df['type'] = ['train loss'] * len(best_hist_train) + ['val loss'] * len(best_hist_val) + ['test loss'] * len(
        best_hist_test)
    plt.figure()
    sns.lineplot(data=loss_df, x='epoch', y='loss', hue='type')
    plt.savefig(f"{config['log_dir']}{config['attribute']}_deepset_{exec_name}.png")

    return best_train_measure, best_val_measure, best_test_measure  # returns the optimal F1-score found

def collate_custom(batch):
    # get the batch size
    # bs = len(batch)

    # batch: <user id, playlists, #playlist, y >
    user_id = [x[0] for x in batch]
    X = torch.cat([x[1] for x in batch])
    l = [x[2] for x in batch]
    # l = torch.cumsum(torch.tensor([0] + l), dim = 0)
    l2 = []  # torch.zeros(X.shape[0], dtype = torch.long) #

    for i, j in enumerate(l):
        # print(i, j, l2)
        l2.append(torch.full(size=(l[i], 1), fill_value=i, dtype=torch.long))

    l2 = torch.cat(l2)

    Y = torch.tensor([x[3] for x in batch])

    # print(X.shape, l2.shape, Y.shape)
    return [user_id, X, l2, Y]


def train_deepset(config, df, Xf, yf, train_id, val_id, test_id, phi_layers, rho_layers, lr, activation, device):
    # construct training, validation, and testing data
    train_ds = CustomDataset(df=df, uid=train_id, Xf=Xf, yf=yf, partition='train'
                             )

    val_ds = CustomDataset(df=df, uid=val_id, Xf=Xf, yf=yf, partition='val',
                           transform_X=train_ds.transform_X,
                           transform_y=train_ds.transform_y
                           )

    test_ds = CustomDataset(df=df, uid=test_id, Xf=Xf, yf=yf, partition='test',
                            transform_X=train_ds.transform_X,
                            transform_y=train_ds.transform_y
                            )

    # define the dataloader
    bs = 16
    dataloader = {
        'train': DataLoader(train_ds, batch_size=bs, shuffle=True, collate_fn=collate_custom),
        'val': DataLoader(val_ds, batch_size=bs, shuffle=False, collate_fn=collate_custom),
        'test': DataLoader(test_ds, batch_size=bs, shuffle=False, collate_fn=collate_custom),
    }

    # define the loss function, based on the type of task
    loss_fn = nn.NLLLoss()

    # define the classifier
    model = Deepset(phi_layers=phi_layers, rho_layers=rho_layers, activation= activation, device=device)
    model.to(device)

    if config['verbose'] > 0:
        print(f"\n\n{phi_layers}")
        print(model.phi)
        print(f"\n\n{rho_layers}")
        print(model.rho)

    # #define the optimizer
    # optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay = 5e-4)

    # ============= training phase ========================
    best_val_loss = np.inf
    best_val_f1 = -1
    no_updates_count = 0
    history_train, history_val, history_test = [], [], []

    for epoch in range(config['max_epochs']):
        # *** TRAINING SET ***
        # training mode
        model.train()

        train_loss = []
        y_true, y_pred = [], []
        for i, (_, x, l, y) in enumerate(dataloader['train']):
            # reset the gradient
            optimizer.zero_grad()

            #
            x = x.to(device)
            y = y.to(device)

            # forward
            y_i = model(x, l)

            # save the ground truth and output variable
            y_true.append(y)
            y_pred.append(torch.argmax(y_i,dim=1))

            # compute the loss
            loss = loss_fn(y_i, y)

            # optimize
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())

        # compute the metrics
        y_true=torch.cat(y_true)
        y_pred=torch.cat(y_pred)
        history_train.append(np.mean(train_loss))
        train_measure = f1_score(y_true=y_true, y_pred=y_pred, average='weighted')
        # train_measure = accuracy_score(y_true= y_true, y_pred= y_pred)

        # *** VALIDATION SET ***
        # evaluation mode
        model.eval()

        y_true, y_pred = [], []
        with torch.no_grad():  # avoid the grad computation
            val_loss = []
            for i, (_, x, l, y) in enumerate(dataloader['val']):
                x = x.to(device)
                y = y.to(device)

                # forward
                y_i = model(x, l)

                # loss
                loss = loss_fn(y_i, y)

                # save the ground truth and output variable
                y_true.append(y)
                y_pred.append(torch.argmax(y_i, dim=1))
                val_loss.append(loss.item())

            # compute the metrics
            y_true = torch.cat(y_true)
            y_pred = torch.cat(y_pred)
            history_val.append(np.mean(val_loss))
            val_measure = f1_score(y_true=y_true, y_pred=y_pred, average='weighted')
            # val_measure = accuracy_score(y_true= y_true, y_pred= y_pred)

        # *** TEST SET ***
        # evaluation mode
        model.eval()

        y_true, y_pred = [], []
        with torch.no_grad():  # avoid the grad computation
            test_loss = []
            for i, (_, x, l, y) in enumerate(dataloader['test']):
                x = x.to(device)
                y = y.to(device)

                # forward
                y_i = model(x, l)

                # loss
                loss = loss_fn(y_i, y)

                # save the ground truth and output variable
                y_true.append(y)
                y_pred.append(torch.argmax(y_i, dim=1))
                test_loss.append(loss.item())

            # compute the metrics
            y_true = torch.cat(y_true)
            y_pred = torch.cat(y_pred)
            history_test.append(np.mean(test_loss))
            test_measure = f1_score(y_true=y_true, y_pred=y_pred, average='weighted')

        # *** EARLY STOPPPING MECHANISM ***
        # if history_val[-1] < best_val_loss:  # improvement over the validation set
        if val_measure > best_val_f1:  # improvement over the validation set
            best_val_loss = history_val[-1]  # update best loss
            no_updates_count = 0  # reset the early stopping counter
            best_epoch = epoch  # save the current epoch number

            # update the matrics set accordingly
            best_val_f1 = val_measure
            best_train_measure = train_measure
            best_val_measure = val_measure
            best_test_measure = test_measure

        else:  # no improvements
            no_updates_count += 1  # update the early stopping counter

        # print epoch stats if required
        if config['verbose'] > 0:
            print(
                f"\t---> epoch {epoch}\ttrain loss: {history_train[-1]:.8f}\tval loss: {history_val[-1]:.8f}\ttest loss: {history_test[-1]:.8f}\tpatience: {no_updates_count}")
            print(
                f"\t---> epoch {epoch}\ttrain f1: {train_measure:.8f}\tval f1: {val_measure:.8f}\ttest f1: {test_measure:.8f}\tpatience: {no_updates_count}")
            print("\n")

        # early stopping mechanism checker
        if no_updates_count == config['patience']:
            # cut the histories
            history_train = history_train[:best_epoch]
            history_val = history_val[:best_epoch]
            history_test = history_test[:best_epoch]

            if config['verbose'] > 0:
                print(f"\tEARLY STOPPIN MECHANISM ACTIVATED\n\n")

            break

    return best_val_loss, best_train_measure, best_val_measure, best_test_measure, history_train, history_val, history_test
class CustomDataset(Dataset):
    def __init__(self, df, uid, Xf, yf, partition,
                 transform_X=None, transform_y=None):
        # save the settings
        self.df = df[df["id_owner"].isin(uid)]  # maintain only the user ids
        self.uid = uid
        self.Xf = Xf
        self.yf = yf
        self.partition = partition
        self.transform_X = transform_X
        self.transform_y = transform_y

        # if training partition, we fit a scaler
        if self.partition == 'train':
            self.transform_X = MinMaxScaler().fit(self.df[self.Xf])

        # construct the dataset
        self.data = []  # <user id, {samples}, # samples, y>
        for u in self.uid:
            # get the current user
            curr_df = self.df[self.df['id_owner'] == u]
            if len(curr_df) == 0:
                raise Exception(f"An error occured while extracting user: {u}")

            # extract X
            X = self.transform_X.transform(curr_df[self.Xf])

            # extract y
            y = curr_df[self.yf]
            y = list(y)[0]

            self.data.append((u, X, len(X), y))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        u, x, l, y = self.data[idx]
        return u, torch.tensor(x, dtype=torch.float32), l, torch.tensor(y, dtype=torch.long)


class Deepset(nn.Module):
    def __init__(self, phi_layers, rho_layers, activation, device):
        super(Deepset, self).__init__()
        # save the parameters
        self.phi_layers = phi_layers
        self.rho_layers = rho_layers
        self.is_classification = True
        self.activation = activation
        self.device = device

        # define the network
        layers = []  # phi
        for i in range(len(self.phi_layers) - 1):
            # dense layer
            layers.append(nn.Linear(self.phi_layers[i],
                                    self.phi_layers[i + 1]))

            layers.append(nn.Dropout(0.3))

            # activation function
            if activation == "relu":
                layers.append(nn.ReLU())
            else:
                layers.append(nn.Sigmoid())

        self.phi = nn.Sequential(*layers)

        layers2 = []
        for i in range(len(self.rho_layers) - 1):
            # dense layer
            layers2.append(nn.Linear(self.rho_layers[i],
                                     self.rho_layers[i + 1]))

            # activation function for all but last layer
            if i != (len(self.rho_layers) - 2):
                layers2.append(nn.Dropout(0.3))
                if activation == "relu":
                    layers.append(nn.ReLU())
                else:
                    layers.append(nn.Sigmoid())
            else:
                layers2.append(nn.LogSoftmax(dim=1))
            # elif not self.is_classification:
            #     #the last layers contains a sigmoid, if it is a regression
            #     #nothing otherwise
            #     layers2.append(nn.Sigmoid())

        self.rho = nn.Sequential(*layers2)

        # reset the parameters
        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data = nn.init.uniform_(m.weight.data,
                                                 a=-0.01, b=0.01)

    def forward(self, x, l):
        # remove the batch size sinze the number of songs / playlists is our batch size
        # x = x.squeeze(0)

        # forward phi
        x = self.phi(x)

        # compute summary statistics

        x_sum = torch.zeros((int(torch.max(l))+1, x.shape[1]))
        x_mean = torch.zeros((int(torch.max(l))+1, x.shape[1]))
        x_max = torch.zeros((int(torch.max(l))+1, x.shape[1]))

        scatter_add(index = l.T[0], src = x,out=x_sum, dim=0)
        scatter_mean(index = l.T[0], src = x,out=x_mean, dim=0)
        scatter_max(index = l.T[0], src = x,out=x_max, dim=0)
        # x_sum = torch.Tensor.scatter_(dim = 1, index = l, src = x, reduce = 'add')
        # x_max = torch.Tensor.scatter_(dim = 1, index = l, src = x, reduce = 'max')
        # x_avg = torch.Tensor.scatter_(dim = 1, index = l, src = x, reduce = 'mean')

        # #concatenate the results
        x_stat = torch.cat([x_sum, x_mean, x_max], dim=1)

        # x_stat[:, : x.shape[1]] = x_sum
        # x_stat[:, x.shape[1] : 2 * x.shape[1]] = x_max
        # x_stat[:, 2 * x.shape[1] : 3 * x.shape[1]] = x_avg
        # x_stat[:, 3 * x.shape[1] : 4 * x.shape[1]] = x_med

        # #compute summary statistics
        # x_sum = torch.sum(x, dim = 0, keepdim = True)
        # x_max = torch.max(x, dim = 0, keepdim = True)[0]
        # x_avg = torch.mean(x, dim = 0, keepdim = True)
        # x_med = torch.median(x, dim = 0, keepdim = True)[0]

        # #concatenate the results
        # x_stat = torch.zeros(1, 4 * x.shape[1]).to(self.device)
        # x_stat[:, : x.shape[1]] = x_sum
        # x_stat[:, x.shape[1] : 2 * x.shape[1]] = x_max
        # x_stat[:, 2 * x.shape[1] : 3 * x.shape[1]] = x_avg
        # x_stat[:, 3 * x.shape[1] : 4 * x.shape[1]] = x_med

        # forward rho
        y = self.rho(x_stat)
        return y

