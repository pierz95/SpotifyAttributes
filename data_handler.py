import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
import random 

#define the mappers 
gt_mapper = {
    'gender': {'Male': 0, 'Female': 1, 'Other': 2},
    'country': {'United States': 0, 'Other': 1, 'United Kingdom' :2, 
                'Canada':3, 'Italy': 4, 'Brazil':5, 'Germany':6,  
                'Australia':7, 'Philippines': 8, 'India': 9},
    'marital_status': {'Single': 0, 'Relationship': 1},
    'live_with': {"I live with others": 0, "I live alone": 1},
    'occupation': {'Job': 0, 'No job': 1},
    "sport": {"No": 0, "Sometimes": 1, "Regularly": 2},
    "economic": {"Low": 0, "Medium": 1, "High": 2},
    'alchol': {'No': 0, 'Yes': 1},
    'smoke': {'No': 0, 'Yes': 1},
    'spotify_premium': {'No': 0, 'Yes': 1}
}


def load_data(attribute):
    #check the attribute to forecast. this list contains examples we tested
    if attribute not in ["gender", "age", "country", "marital_status", 
                         "live_with", "occupation", "sport", 
                         "economic", "alchol", "smoke", 
                         'openness', " conscientiousness", 
                         "extraversion", "agreeableness", "neuroticism", "spotify_premium"]:
        raise Exception("Invalid attribute to infer")
 
    #open playlist dataset and ground truth dataset
    playlist_df = pd.read_csv('./Datasets/df_playlists.csv')
    playlist_df['p_country'] = playlist_df['country']
    playlist_df.drop(['country'], axis = 1, inplace=True)
    gt_df = pd.read_csv('./Datasets/user_targets.csv')
    
    #remove from the playlists all those not belonging to any remaining user
    usersid = list(set(gt_df["id_owner"]))
    playlist_df = playlist_df[playlist_df['id_owner'].isin(usersid)]
    
    # #find nan columns
    # # print(playlist_df.columns[playlist_df.isna().any()].tolist())
    playlist_df['n_follower_playlist'] = playlist_df['n_follower_playlist'].fillna(0) 
    
    #extract the features
    X_features = playlist_df.columns[2:]
    
    #remove duplicate users
    duplicate_users = list(gt_df[gt_df['id_owner'].duplicated()]['id_owner'].unique())
    playlist_df = playlist_df[~playlist_df['id_owner'].isin(duplicate_users)]        
    gt_df = gt_df[~gt_df['id_owner'].isin(duplicate_users)]    

    #merge the two dataframes
    playlist_df['id_owner'] = playlist_df['id_owner'].astype(str)
    gt_df['id_owner'] = gt_df['id_owner'].astype(str)
    df_overall = pd.merge(playlist_df, gt_df, how='left', on='id_owner')
    
    #convert the attribute to numerical values
    if attribute in gt_mapper.keys():
        df_overall[attribute] = df_overall[attribute].map(gt_mapper[attribute])
     
    return df_overall, X_features, attribute

def generate_splits(df, yf, n_splits = 1, seed = 123):
    """ Generate a list of train, val, test splits """
    #get users id for the CV
    uid = list(df["id_owner"].unique())
    random.seed(seed)
    
    #obtain the attribute distribution 
    y = []
    for u in uid:
        y.append(df[df['id_owner'] == u][yf].tolist()[0])
    
    assert len(uid) == len(y)
    
    sets = []
    sss = StratifiedShuffleSplit(n_splits=n_splits, test_size=0.2, random_state=seed)
    for i, (train_val_index, test_index) in enumerate(sss.split(uid, y)):
        #update the info
        uid_train_val = [uid[i] for i in train_val_index]
        # uid_train_val = uid[train_val_index]
        y_train_val = [y[i] for i in train_val_index]
        # y_train_val = y[train_val_index]
        uid_test = [uid[i] for i in test_index]
        # uid_test = uid[test_index]
        
        sss1 = StratifiedShuffleSplit(n_splits=1, test_size=0.125, random_state=seed)
        for i, (train_index, val_index) in enumerate(sss.split(uid_train_val, y_train_val)):
            # uid_train = uid[train_index]
            # uid_val = uid[val_index]
            uid_train = [uid_train_val[i] for i in train_index]
            uid_val = [uid_train_val[i] for i in val_index]
        
        sets.append((uid_train, uid_val, uid_test))

    ## === old fashioned
    # for i in range(n_splits):        
        
        # #shuffle
        # random.shuffle(uid)

        #split into train, val and test sets
        # tr_id, val_id, test_id = uid[:int(0.6*(len(uid)))], uid[int(0.6*(len(uid))):int(0.8*(len(uid)))], uid[int(0.8*(len(uid))):]
        # sets.append((tr_id, val_id, test_id))
        
    return sets
