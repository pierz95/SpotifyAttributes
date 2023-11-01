import torch
from torch.utils.data import Dataset
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import f1_score, mean_squared_error

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

#set the seed
torch.manual_seed(0)

class CustomDataset(Dataset):
    def __init__(self, df, uid, Xf, yf, transform = None, 
                 transform2 = None, is_classification = True):
        #save the arguments
        self.df= df
        self.uid = uid
        self.Xf = Xf
        self.yf = yf
        self.transform = transform
        self.transform2 = transform2
        self.is_classification = is_classification

        #get the current dataset
        df_split = df[df["id_owner"].isin(uid)]

        #if transform is None, then we are using a training set
        # therefore, we train a custom Scaler
        if transform is None:
            self.transform = MinMaxScaler() #generate a new scaler
            self.transform.fit(df_split[self.Xf]) #fit the scaler
        
        if not is_classification:
            if transform2 is None:
                self.transform2 = MinMaxScaler()
                self.transform2.fit(np.array(df_split[self.yf]).reshape(-1, 1))
        
        #for each user id, we need to define a set of samples
        self.data = [] #format <uid, {samples}, #samples, y >
        for u in self.uid:
            #get the list of samples of the given user
            curr_df = self.df[self.df['id_owner'] == u]

            #extract the samples
            X = curr_df[self.Xf] #this is a pandas dataframe
            X = self.transform.transform(X) #this is a standardized 2D matrix

            Y = curr_df[self.yf]
            
            if self.transform2 is not None:
                Y = np.array(Y).reshape(-1, 1)
                Y = self.transform2.transform(Y)[0]
            else:
                Y = list(Y)
            
            #save the result
            self.data.append((u, X, len(X), Y[0]))
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        #extract the given item
        u, x, l, y = self.data[idx]
        return u, torch.tensor(x,  dtype=torch.float32), l, torch.tensor(y,  dtype=torch.float32)

class Deepset(nn.Module):
    def __init__(self, in_dim, phi_layers, rho_layers, is_classification):
        super(Deepset, self).__init__()
        #save the arguments
        self.in_dim = in_dim
        self.phi_layers = phi_layers
        self.rho_layers = rho_layers
        self.is_classification = is_classification


        #define the network
        layers = []
        for i in range(len(self.phi_layers) - 1):
            layers.append(nn.Linear(self.phi_layers[i], self.phi_layers[i+1]))
            layers.append(nn.ReLU()) #activation function

        self.phi = nn.Sequential(*layers)

        layers2 = []
        for i in range(len(self.rho_layers) - 1):
            layers2.append(nn.Linear(self.rho_layers[i], self.rho_layers[i+1]))
            if i == len(self.rho_layers) - 2 and self.is_classification: #output layer
                layers2.append(nn.Sigmoid()) #activation function
            else:
                layers2.append(nn.ReLU()) #activation function

        self.rho = nn.Sequential(*layers2)

        #reset parameters
        self.reset_parameters()


    def reset_parameters(self):
        for m in self.modules():
                if isinstance(m, nn.Linear):
                    m.weight.data = nn.init.uniform_(m.weight.data, 
                        a = -0.01, b = 0.01) 

    def forward(self, x):
        x = x.squeeze(0) #remove the batch size
        
        #forward phi
        x = self.phi(x)

        #summary statistics
        x_sum = torch.sum(x, dim = 0, keepdim = True)
        x_max = torch.max(x, dim = 0, keepdim = True)[0]
        x_avg = torch.mean(x, dim = 0, keepdim = True)
        x_med = torch.median(x, dim = 0, keepdim = True)[0]
        x_stat = torch.cat([x_sum, x_max, x_avg, x_med], dim = 1)

        #forward rho
        y = self.rho(x_stat)
        return y
    
def train_deepset(df, Xf, yf, train_id, val_id, test_id, phi_layers, 
                  rho_layers, is_classification, max_epochs = 20, patience = 5, 
                  lr = 0.0001, verbose = 0):
    #create the custom dataset - torch instance
    train_ds = CustomDataset(df=df, uid=train_id, Xf=Xf, yf=yf, is_classification= is_classification)
    val_ds = CustomDataset(df=df, uid=val_id, Xf=Xf, yf=yf, is_classification= is_classification,
        transform= train_ds.transform, transform2= train_ds.transform2)
    test_ds = CustomDataset(df=df, uid=test_id, Xf=Xf, yf=yf, is_classification= is_classification, 
        transform= train_ds.transform, transform2= train_ds.transform2)
    
    #define the dataloader
    dataloader = {
        'train': DataLoader(train_ds, batch_size = 1, shuffle = False),
        'val': DataLoader(val_ds, batch_size = 1, shuffle = False), 
        'test': DataLoader(test_ds, batch_size = 1, shuffle = False)
    }

    #define the classifier
    clf = Deepset(in_dim= len(Xf), phi_layers= phi_layers, 
                  rho_layers=rho_layers, is_classification = is_classification)
    # class_weights= compute_class_weight('balanced',np.unique(train_ds[yf]),np.array(train_ds[yf]))
    # class_weights=torch.tensor(class_weights,dtype=torch.float)
    if is_classification:
        loss_fn = nn.BCELoss()  # binary cross entropy
    else:
        loss_fn = nn.MSELoss()  # mean square error

    optimizer = optim.Adam(clf.parameters(), lr=0.01)

    #iterate over the epochs
    best_val_loss = np.inf
    best_test_f1 = 0
    no_updates_count = 0
    best_epoch = max_epochs
    history_train, history_val, history_test = [], [], []
    for epoch in range(max_epochs):
        #training mode
        clf.train()

        train_loss = []
        y_true, y_pred = [], []
        for i, (u, x, l, y) in enumerate(dataloader['train']):
            #forward 
            y_i= clf(x)
            y = y.unsqueeze(0)

            y_true.append(int(y.item()))
            if y_i >.5:
                y_pred.append(1)
            else:
                y_pred.append(0)
            
            #loss
            loss = loss_fn(y_i, y)
            
            #optimizer
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss.append(loss.item())
        
        if is_classification:
            train_f1 = f1_score(y_true, y_pred, average = 'macro')
        else:
            train_f1 = mean_squared_error(y_true, y_pred)
        history_train.append(np.mean(train_loss))

        #validation
        clf.eval()
        y_true, y_pred = [], []
        with torch.no_grad():
            val_loss = []
            for i, (u, x, l, y) in enumerate(dataloader['val']):
                #forward 
                y_i= clf(x)
                y = y.unsqueeze(0)

                #loss
                loss = loss_fn(y_i, y)
                
                val_loss.append(loss.item())

                y_true.append(y.item())
                if y_i >.5:
                    y_pred.append(1)
                else:
                    y_pred.append(0)

        if is_classification:
            val_f1 = f1_score(y_true, y_pred, average = 'macro')
        else:
            val_f1 = mean_squared_error(y_true, y_pred)
        history_val.append(np.mean(val_loss))
        
        #test loss
        y_true, y_pred = [], []
        with torch.no_grad():
            test_loss = []
            for i, (u, x, l, y) in enumerate(dataloader['test']):
                #forward 
                y_i= clf(x)
                y = y.unsqueeze(0)

                #loss
                loss = loss_fn(y_i, y)
                
                test_loss.append(loss.item())

                y_true.append(y.item())
                if y_i >.5:
                    y_pred.append(1)
                else:
                    y_pred.append(0)
        
        history_test.append(np.mean(test_loss))
        
        #improve the loss
        if np.mean(val_loss) < best_val_loss:
            best_val_loss = np.mean(val_loss)
            no_updates_count = 0
            best_epoch = epoch
            if is_classification:
                best_train_f1 = train_f1
                best_val_f1 = val_f1
                best_test_f1 = f1_score(y_true, y_pred, average = 'macro')
                
            else:
                best_test_f1 = mean_squared_error(y_true, y_pred)

        else: #patience mechianism
            no_updates_count +=1
    
        if verbose > 0:
            print(f"\t--->epoch {epoch}\ttrain loss: {np.mean(train_loss):.4f}\tval loss: {np.mean(val_loss):.4f}\ttest loss: {np.mean(test_loss):.4f}\tpatience: {no_updates_count}")
    
        # check if early stopping mechanism has been triggered
        if no_updates_count == patience:
            if verbose > 0:
                print(f"EARLY STOPPING ACTIVATED\n\n")
            break 
        
    return best_val_loss, best_train_f1, best_val_f1, best_test_f1, history_train[:best_epoch], history_val[:best_epoch], history_test[:best_epoch]

def exec_deepset(config, df, Xf, yf, train_id, val_id, test_id, 
                 max_epochs = 50, patience = 5, verbose = 0, 
                 log_dir = './tmp/', exec_name = 'tmp'):
    
    #define the list of layers combinations
    phi_rho_layers = [] #list of tuple

    max_phi_depth = config['max_phi_depth']
    max_rho_depth = config['max_rho_depth']

    for i in range(max_phi_depth):
        #construct phi
        phi = [len(Xf)] #constraint in the input size
        for ii in range(i + 1):
            if ii == 0:
                phi.append(phi[-1] * 2) #expand
            else:
                phi.append(phi[-1] // 2) #bottlneck structure

        for j in range(max_rho_depth):
            #construct rho
            rho = [4 * phi[-1]] #constraint to the last phi-layer
            for jj in range(j): 
                rho.append(rho[-1] // 2)
            rho.append(1) #output layer
        
            phi_rho_layers.append((phi, rho))

    #exec the deepset over the various parameters
    best_val_loss = None
    log = pd.DataFrame(None, columns = ['rho', 'phi', 'lr', 'val_loss', 'train_f1', 'val_f1', 'test_f1'])
    for x, y in phi_rho_layers:
        #iterate over the learning rate
        for lr in config['lr']:
            curr_val_loss, curr_train_f1, curr_val_f1, curr_test_f1, curr_hist_train, curr_hist_val, curr_hist_test = train_deepset(df, Xf, yf, 
                train_id, val_id, test_id, x, y, config['is_classification'], 
                max_epochs, patience, lr, verbose = config['verbose']) 

            if best_val_loss is None or best_val_loss > curr_val_loss: #init case or optimization
                best_val_loss = curr_val_loss
                best_test_f1 = curr_test_f1
                best_hist_train = curr_hist_train
                best_hist_val = curr_hist_val
                best_hist_test = curr_hist_test

            #save the log
            log.loc[len(log)] = [x, y, lr, curr_val_loss, curr_train_f1, curr_val_f1, curr_test_f1]


    #save the training info
    log.to_csv(f'{log_dir}log_{config["attribute"]}_{exec_name}.csv')

    #build the loss plot
    loss_df = pd.DataFrame(None)
    loss_df['epoch'] = list(range(len(best_hist_train))) + list(range(len(best_hist_val))) + list(range(len(best_hist_test)))
    loss_df['loss'] = best_hist_train + best_hist_val + best_hist_test
    loss_df['type'] = ['train loss'] * len(best_hist_train) + ['val loss'] * len(best_hist_val) + ['test loss'] * len(best_hist_test) 
    plt.figure()
    sns.lineplot(data = loss_df, x = 'epoch', y = 'loss', hue = 'type')
    plt.savefig(f"{log_dir}__{config['attribute']}_{exec_name}.png")        
    
    return best_test_f1 #returns the optimal F1-score found 