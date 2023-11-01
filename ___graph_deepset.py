from random import shuffle
from tabnanny import verbose
import torch
from torch.utils.data import Dataset
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import f1_score, mean_squared_error
from sklearn.utils.class_weight import compute_class_weight

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

import torch_geometric

#set the seed
torch.manual_seed(123)

def exec_graph_deepset(config, df, Xf, yf, train_id, val_id, test_id, exec_name):
    
    #==== Construct the various combination of layers for phi e rho =================
    phi_rho_layers = []

    #retrieve the global variables from the config file
    max_phi_depth = config['max_phi_depth']
    max_rho_depth = config['max_rho_depth']

    #iterate over the constructors
    for i in range(max_phi_depth):
        #initialize the phi layer. 
        #the first dim MUST be equal to the input dim
        phi = [50] #Todo edit
        for ii in range(i + 1):
            if ii == 0: #first iteration: expand
                phi.append(phi[-1]*2)
            else: #compress
                phi.append(phi[-1] // 2)
        
        for j in range(max_rho_depth):
            #construct rho
            # the first dim is equal to 4 times last phi layer 
            # this because we adopt 4 summary statistics
            rho = [phi[-1] * 4]
            for jj in range(j):
                rho.append(rho[-1] // 2)

            #we include this iteration only if the last layer is valid
            if rho[-1] != 0: 
                # add the output layer -- constrainted to the n_classes
                # note that in the reegression task, n_classes = 1
                rho.append(config['n_classes']) 

                #update the overall structure
                phi_rho_layers.append((phi, rho))

    #========================== training phase ====================================
    best_val_loss = np.inf
    log = pd.DataFrame(None, columns = ['rho', 'phi', 'lr', 'val_loss', 'train_f1', 'val_f1', 'test_f1'])
    for x, y in phi_rho_layers:
        for lr in config['lr']:
            curr_val_loss, curr_train_m, curr_val_m, curr_test_m, curr_hist_train, curr_hist_val, curr_hist_test = train_deepset(config, 
                df, Xf, yf, train_id, val_id, test_id, x, y, lr)
        
            #save the best new setting, if founded
            if curr_val_loss < best_val_loss:
                best_val_loss = curr_val_loss
                best_test_measure = curr_test_m
                best_hist_train = curr_hist_train
                best_hist_val = curr_hist_val
                best_hist_test = curr_hist_test
            
            #save the log
            log.loc[len(log)] = [x, y, lr, curr_val_loss, curr_train_m, curr_val_m, curr_test_m]

    #========================== save settings ====================================
    log.to_csv(f'{config["log_dir"]}{config["level"]}__log_{config["attribute"]}_{exec_name}.csv')

    #build the loss plot
    loss_df = pd.DataFrame(None)
    loss_df['epoch'] = list(range(len(best_hist_train))) + list(range(len(best_hist_val))) + list(range(len(best_hist_test)))
    loss_df['loss'] = best_hist_train + best_hist_val + best_hist_test
    loss_df['type'] = ['train loss'] * len(best_hist_train) + ['val loss'] * len(best_hist_val) + ['test loss'] * len(best_hist_test) 
    plt.figure()
    sns.lineplot(data = loss_df, x = 'epoch', y = 'loss', hue = 'type')
    plt.savefig(f"{config['log_dir']}{config['level']}__{config['attribute']}_{exec_name}.png")        
    
    return best_test_measure #returns the optimal F1-score found 


def train_deepset(config, df, Xf, yf, train_id, val_id, test_id, phi_layers, rho_layers, lr):
    
    #extract training metadata
    is_classification = config['is_classification']
    max_epochs = config['max_epochs']
    patience = config['patience']
    data = torch.load('Datasets/data.pt')

    #define the loss function, based on the type of task
    if is_classification:
        loss_fn = nn.CrossEntropyLoss()
    else:
        loss_fn = nn.MSELoss()

    gcn_model=base_GC_net(conv=torch_geometric.nn.GCNConv, in_channels=data.x.shape[1], 
                          n_gc_hidden_units=config['n_units'], drop_prob=config['dropout'], n_layer = config['n_layers'],
                          conv_act= nn.ReLU(), device='cpu')
    
    
    #define the classifier
    model = Deepset(phi_layers= phi_layers, rho_layers= rho_layers, 
                    is_classification= is_classification)

    if config['verbose'] > 0:
        print(f"\n\n{phi_layers}")
        print(model.phi)
        print(f"\n\n{rho_layers}")
        print(model.rho)

    
    params = list(model.parameters()) + list(gcn_model.parameters())

    # #define the optimizer
    optimizer = optim.SGD(params, lr=lr, momentum=0.9)

    # ============= training phase ========================
    best_val_loss = np.inf
    no_updates_count = 0
    history_train, history_val, history_test = [], [], []

    for epoch in range(config['max_epochs']):

        # *** TRAINING SET ***
        train_loss = []
        y_true, y_pred = [], []

        #training mode
        gcn_model.train() 

        #todo:definire loader sopra


        model.train()

        # data = data.to(device)

        optimizer.zero_grad()
        loss=0.0

        h_g = gcn_model(data) 

        print(train_id[:10])
        print(len(data.id_users))
        print(h_g.shape)

        h_playlist = []
        y_train = []
        for cur_user in train_id:
            cur_user = hash(cur_user)
            cur_playlist = []
            for i, j in enumerate(data.id_users):
                print(data.id_users[i], cur_user)
                if data.id_users[i] == cur_user:
                    cur_playlist.append(h_g[i])
                    tmp = data.y_gender[i]
            
            y_train.append(tmp)
            
            h_playlist.append(cur_playlist)


        for i, (x, y) in enumerate(zip(h_playlist,y_train)):#TODO: occhio che non ci sono più le liste per utente e vanno ricreate in base a quello che fa pier
            #reset the gradient
            #optimizer.zero_grad()

            #forward 
            y_i = model(x)
            
            #save the ground truth and output variable
            y_true.append(y.item())
            y_pred.append(torch.argmax(y_i).item())

            #compute the loss
            loss += loss_fn(y_i, y)
            
        #optimize
        loss.backward()
        optimizer.step()
        train_loss.append(loss.item())

        #compute the metrics
        history_train.append(np.mean(train_loss))
        if is_classification:
            train_measure = f1_score(y_true= y_true, y_pred= y_pred, average= 'macro')
        else:
            train_measure = mean_squared_error(y_true=y_true, y_pred= y_pred)
        
        # *** VALIDATION SET ***
        #evaluation mode
        model.eval()

        y_true, y_pred = [], []
        with torch.no_grad(): #avoid the grad computation 
            val_loss = []
            for i, (_, x, _, y) in enumerate(dataloader['val']):
                #forward 
                y_i = model(x)

                #loss
                loss = loss_fn(y_i, y)

                #save the ground truth and output variable
                y_true.append(y.item())
                y_pred.append(torch.argmax(y_i).item())
                val_loss.append(loss.item())
            
            #compute the metrics
            history_val.append(np.mean(val_loss))
            if is_classification:
                val_measure = f1_score(y_true= y_true, y_pred= y_pred, average= 'macro')
            else:
                val_measure = mean_squared_error(y_true=y_true, y_pred= y_pred)

        # *** TEST SET ***
        #evaluation mode
        model.eval()

        y_true, y_pred = [], []
        with torch.no_grad(): #avoid the grad computation 
            test_loss = []
            for i, (_, x, _, y) in enumerate(dataloader['test']):
                #forward 
                y_i = model(x)

                #loss
                loss = loss_fn(y_i, y)

                #save the ground truth and output variable
                y_true.append(y.item())
                y_pred.append(torch.argmax(y_i).item())
                test_loss.append(loss.item())
            
            #compute the metrics
            history_test.append(np.mean(test_loss))
            if is_classification:
                test_measure = f1_score(y_true= y_true, y_pred= y_pred, average= 'macro')
            else:
                test_measure = mean_squared_error(y_true=y_true, y_pred= y_pred)

        # *** EARLY STOPPPING MECHANISM ***
        if history_val[-1] < best_val_loss: #improvement over the validation set
            best_val_loss = history_val[-1] #update best loss
            no_updates_count = 0 #reset the early stopping counter
            best_epoch = epoch #save the current epoch number

            #update the matrics set accordingly
            best_train_measure = train_measure
            best_val_measure = val_measure
            best_test_measure = test_measure 

        else: #no improvements
            no_updates_count += 1 #update the early stopping counter

        #print epoch stats if required
        if config['verbose'] > 0:
             print(f"\t---> epoch {epoch}\ttrain loss: {history_train[-1]:.8f}\tval loss: {history_val[-1]:.8f}\ttest loss: {history_test[-1]:.8f}\tpatience: {no_updates_count}")

        #early stopping mechanism checker
        if no_updates_count == config['patience']:
            #cut the histories
            history_train = history_train[:best_epoch]
            history_val = history_val[:best_epoch]
            history_test = history_test[:best_epoch]

            if config['verbose'] > 0:
                print(f"\tEARLY STOPPIN MECHANISM ACTIVATED\n\n")
            
            break
        
    return best_val_loss, best_train_measure, best_val_measure, best_test_measure, history_train, history_val, history_test

class CustomDataset(Dataset):
    def __init__(self, df, uid, Xf, yf, is_classification, partition,  
                 transform_X = None, transform_y = None):
        #save the settings
        self.df = df[df["id_owner"].isin(uid)] #maintain only the user ids
        self.uid = uid
        self.Xf = Xf
        self.yf = yf
        self.is_classification = is_classification
        self.partition = partition
        self.transform_X = transform_X
        self.transform_y = transform_y

        #if training partition, we fit a scaler
        if self.partition == 'train':
            self.transform_X = MinMaxScaler().fit(self.df[self.Xf])

            #if regression task, we scale y as well 
            if not self.is_classification:
                self.transform_y = MinMaxScaler().fit(self.df[self.yf])

        #construct the dataset
        self.data = [] # <user id, {samples}, # samples, y>
        for u in self.uid:
            #get the current user
            curr_df = self.df[self.df['id_owner'] == u]
            if len(curr_df) == 0:
                raise Exception(f"An error occured while extracting user: {u}")
            
            #extract X 
            X = self.transform_X.transform(curr_df[self.Xf])
            
            #extract y
            y = curr_df[self.yf]
            if not self.is_classification:
                y = self.transform_y.transform(y)[0]
            else:
                y = list(y)[0]

            self.data.append((u, X, len(X), y))

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        u, x, l, y = self.data[idx]
        return u, torch.tensor(x, dtype = torch.float32), l, torch.tensor(y, dtype = torch.long)

class base_GC_net(torch.nn.Module):

    def __init__(self, in_channels, n_gc_hidden_units,conv, n_layer=0, drop_prob=0, conv_act=lambda x: x,  device=None):
        super(base_GC_net, self).__init__()
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device

        #attributes init:
        self.in_channels = in_channels
        self.n_layer = n_layer
        self.out_channels = n_gc_hidden_units
        self.conv_act = conv_act
        self.dropout = torch.nn.Dropout(p=drop_prob)

        self.conv_layers = torch.nn.ModuleList()
        self.gc_layer_norm = torch.nn.ModuleList()


        #GC layers
        self.conv_layers.append(conv(self.in_channels, self.out_channels))
        self.gc_layer_norm.append(torch.nn.BatchNorm1d(self.out_channels))
        for _ in range(n_layer - 1):
            self.conv_layers.append(conv(self.out_channels, self.out_channels).to(self.device))
            self.gc_layer_norm.append(torch.nn.BatchNorm1d(self.out_channels).to(self.device))
        self.reset_parameters()

    def reset_parameters(self):
        for gc_layer, batch_norm in zip(self.conv_layers, self.gc_layer_norm):
            gc_layer.reset_parameters()
            batch_norm.reset_parameters()

    def forward(self, data):
        
        X = data.x
        edge_index = data.edge_index
        edge_attr = data.edge_attr #TODO:testare sta cosa

        # print(X.shape)
        # print(edge_index.shape)

        h = X
        for gc_layer, batch_norm in zip(self.conv_layers, self.gc_layer_norm):
            h = gc_layer(h, edge_index, edge_weight=edge_attr)#TODO:testare sta cosa che ho aggiunto edge_weight senza far girare il modello
            h = self.conv_act(h)
            h = batch_norm(h)
            h= self.dropout(h)

        return h

class Deepset(nn.Module):
    def __init__(self, phi_layers, rho_layers, is_classification):
        super(Deepset, self).__init__()
        #save the parameters
        self.phi_layers = phi_layers
        self.rho_layers = rho_layers
        self.is_classification = is_classification

        #define the network
        layers = [] #phi
        for i in range(len(self.phi_layers) - 1):
            #dense layer
            layers.append(nn.Linear(self.phi_layers[i], 
                                    self.phi_layers[i + 1]))

            #activation function
            layers.append(nn.ReLU())

        self.phi = nn.Sequential(*layers)

        layers2 = []
        for i in range(len(self.rho_layers) - 1):
            #dense layer
            layers2.append(nn.Linear(self.rho_layers[i], 
                                     self.rho_layers[i + 1]))
            
            #activation function for all but last layer
            if i != (len(self.rho_layers) - 2):
                layers2.append(nn.ReLU())
            elif not self.is_classification:
                #the last layers contains a sigmoid, if it is a regression
                #nothing otherwise
                layers2.append(nn.Sigmoid())

        self.rho = nn.Sequential(*layers2)
        
        #reset the parameters
        self.reset_parameters()
    
    def reset_parameters(self):
        for m in self.modules():
                if isinstance(m, nn.Linear):
                    m.weight.data = nn.init.uniform_(m.weight.data, 
                        a = -0.01, b = 0.01)

    def forward(self, x):
        #remove the batch size sinze the number of songs / playlists is our batch size
        x = x.squeeze(0)

        #forward phi
        x = self.phi(x)

        #compute summary statistics
        x_sum = torch.sum(x, dim = 0, keepdim = True)
        x_max = torch.max(x, dim = 0, keepdim = True)[0]
        x_avg = torch.mean(x, dim = 0, keepdim = True)
        x_med = torch.median(x, dim = 0, keepdim = True)[0]

        #concatenate the results
        x_stat = torch.zeros(1, 4 * x.shape[1])
        x_stat[:, : x.shape[1]] = x_sum
        x_stat[:, x.shape[1] : 2 * x.shape[1]] = x_max
        x_stat[:, 2 * x.shape[1] : 3 * x.shape[1]] = x_avg
        x_stat[:, 3 * x.shape[1] : 4 * x.shape[1]] = x_med

        #forward rho
        y = self.rho(x_stat)
        return y
    



