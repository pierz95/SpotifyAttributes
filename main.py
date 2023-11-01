import argparse
import yaml
from yaml.loader import SafeLoader

import numpy as np
import pandas as pd
import torch

#custom libraries
import data_handler
import baselines
import deepset
import GNNdeepset


parser = argparse.ArgumentParser(
    prog='Baseline Trainer [Playlists version]',
    description='This program allows you to train many baselines to predict private data.')

parser.add_argument('-c', '--config', type = str)

def run_split(config, df, Xf, yf, train_id, val_id, test_id, iter):
    split_results = {}

    if config['classifiers']['dummy']:
        split_results['dummy'] = baselines.exec_baseline(config, 'dummy', df, Xf, yf, train_id, val_id, test_id, iter)
    
    if config['classifiers']['linear']:
        split_results['linear'] = baselines.exec_baseline(config, 'linear', df, Xf, yf, train_id, val_id, test_id, iter)

    if config['classifiers']['decisiontree']:
        split_results['decisiontree'] = baselines.exec_baseline(config, 'decisiontree', df, Xf, yf, train_id, val_id, test_id, iter)
    
    if config['classifiers']['randomforest']:
        split_results['randomforest'] = baselines.exec_baseline(config, 'randomforest', df, Xf, yf, train_id, val_id, test_id, iter)
    
    if config['classifiers']['knn']:
        split_results['knn'] = baselines.exec_baseline(config, 'knn', df, Xf, yf, train_id, val_id, test_id, iter)
        
    if config['classifiers']['mlp']:
        split_results['mlp'] = baselines.exec_baseline(config, 'mlp', df, Xf, yf, train_id, val_id, test_id, iter)

    if config['classifiers']['deepset']:
        split_results['deepset'] = deepset.exec_deepset(config, df, Xf, yf, train_id, val_id, test_id, exec_name = iter, device = device)

    if config['classifiers']['graph_deepset']:
        split_results['graph_deepset'] = graph_deepset.exec_graph_deepset(config, df, Xf, yf, train_id, val_id, test_id, exec_name = iter)
    

    return split_results

if __name__ == "__main__":
    print("\t\t\t*** Spotify - Attribute Inference ***")
    args = parser.parse_args() #retrieve the arguments

    # Open the file and load the file
    with open(args.config) as f:
        config = yaml.load(f, Loader=SafeLoader)
        
    #check if we need to use the GPU for deep neural networks
    if config['classifiers']['deepset'] or config['classifiers']['graph_deepset']:
        if config['use_gpu']:
            global device
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            device = 'cpu'
        
        print('Using device:', device)
            
    #load the data
    df, Xf, yf = data_handler.load_data(config['attribute'])
    print(df[yf].value_counts())
    
    #generate a list of train, val, test partitions
    splits = data_handler.generate_splits(df, yf, n_splits=5)

    scores = []
    log = pd.DataFrame(None, columns=['Model', 'Iter', 'Train Score', 'Val Score', 'Test Score'])

    for iter in range(config['iter']):
        curr_res = run_split(config, df, Xf, yf, splits[iter][0], splits[iter][1], splits[iter][2], iter)
        scores.append(curr_res)
        for k in curr_res.keys():
            log.loc[len(log)] = [k, iter, curr_res[k][0], curr_res[k][1], curr_res[k][2]]
        print(f"iter {iter}:\t{curr_res}")
    
    #save the log
    log.to_csv(f'./results/{yf}.csv')

    #print average statistics
    print("\n\nAverage Statistics")
    for k in config['classifiers'].keys():
        if config['classifiers'][k]: #model enabled
            ii = [x[k][-1] for x in scores]
            print(f"\t--->{k}:\t\t{np.mean(ii):.4f} += {np.std(ii):.4f}")

