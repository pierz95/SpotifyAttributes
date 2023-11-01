# %%
import pandas as pd
import numpy as np
import math
import networkx as nx
from tqdm import tqdm
from IPython.display import display
import torch
from torch_geometric.data import Data


df_link = pd.read_csv("../Datasets/playlist_songs_link_FIXED.csv")

df_playlist = pd.read_csv("../Datasets/df_playlists_FIXED.csv")
df_users = pd.read_csv("../Datasets/user_targets_FIXED.csv")

df_users.drop_duplicates(['id_owner'],keep='last',inplace=True)

print(len(df_users))
print(len(df_playlist))


df_playlist_uniq = df_link.drop_duplicates(['id_playlist_belong'], keep="first")

dict_pl_user = pd.Series(df_playlist_uniq['id_owner'].values,index=df_playlist_uniq['id_playlist_belong']).to_dict()

# print(dict_pl_user)



print(len(df_link['id_playlist_belong'].unique()))

k = df_link.groupby('id_playlist_belong').agg(list).reset_index()

print(k.head())


playlists = k['id_playlist_belong']
songs_per_playlist = k['id_song']



playlist_dict = {playlists[i]: songs_per_playlist[i] for i in range(len(playlists))}

def common_member(a, b):
    a_set = set(a)
    b_set = set(b)
    return len(a_set & b_set) / len(a_set | b_set)


# G = nx.Graph()
# G.add_nodes_from(playlist_dict.keys())
# G.number_of_nodes()

pl_list = list(playlists)

#pl_list = pl_list[:50]


#add X, Y, user, pl_id

_gender_mapper = {
    'Female': 0, 
    'Male': 1,
    'Other': 2
}

_country_mapper = {
    'Italy':0, 
    'Brazil':1,
    'United Kingdom': 2, 
    'Philippines': 3, 
    'United States': 4,
    'India': 5,
    'Australia': 6,
    'Germany': 7,
    'Canada': 8,
    'Other': 9
}

_marital_mapper = {
    'Single':0, 
    'Relationship':1
}

_live_mapper = {
    'I live with others':0, 
    'I live alone':1
}

_occupation_mapper = {
    'No job': 0,
    'Job': 1
}

_sport_mapper = {
    'No': 0,
    'Sometimes': 1,
    'Regularly': 2
}

_economic_mapper = {
    'Low': 0,
    'Medium': 1,
    'High': 2
}

_smoke_mapper = {
    'No': 0,
    'Yes': 1
}

_alchol_mapper = {
    'No': 0,
    'Yes': 1
}

_premium_mapper = {
    'No': 0,
    'Yes': 1
}



x = []

y_age = []
y_gender = []
y_country = []
y_marital = []
y_live = []
y_occupation = []
y_sport = []
y_economic = []
y_smoke = []
y_alchol = []
y_premium = []
y_extra = []
y_agreeable = []
y_consc = []
y_neurot = []
y_open = []

id_users = []
id_playlists = []

df_playlist.set_index('id_playlist',inplace=True)
df_users.set_index('id_owner',inplace=True)
id_playlists_dict = dict()


for i, pl in enumerate(pl_list):
    attributes = df_playlist.loc[pl].values[2:]
    id_user = df_playlist.loc[pl]['id_owner']
    age = df_users.loc[id_user]['age']
    gender = _gender_mapper[df_users.loc[id_user]['gender']]
    country = _country_mapper[df_users.loc[id_user]['country']]
    marital = _marital_mapper[df_users.loc[id_user]['marital_status']]
    live = _live_mapper[df_users.loc[id_user]['live_with']]
    occupation = _occupation_mapper[df_users.loc[id_user]['occupation']]
    sport = _sport_mapper[df_users.loc[id_user]['sport']]
    economic = _economic_mapper[df_users.loc[id_user]['economic']]
    smoke = _smoke_mapper[df_users.loc[id_user]['smoke']]
    alchol = _alchol_mapper[df_users.loc[id_user]['alchol']]
    premium = _premium_mapper[df_users.loc[id_user]['spotify_premium']]
    extra = df_users.loc[id_user]['extraversion']
    agreeable = df_users.loc[id_user]['agreeableness']
    consc = df_users.loc[id_user][' conscientiousness']
    neurot = df_users.loc[id_user]['neuroticism']
    open = df_users.loc[id_user]['openness']

    
    id_user = hash(id_user)

    x.append([attributes])
    y_age.append([age])
    y_gender.append([gender])
    y_country.append([country])
    y_marital.append([marital])
    y_live.append([live])
    y_occupation.append([occupation])
    y_sport.append([sport])
    y_economic.append([economic])
    y_smoke.append([smoke])
    y_alchol.append([alchol])
    y_premium.append([premium])
    y_extra.append([extra])
    y_agreeable.append([agreeable])
    y_consc.append([consc])
    y_neurot.append([neurot])
    y_open.append([open])
    id_users.append([id_user])
    id_playlists.append([i])


y_age = torch.tensor(y_age, dtype = torch.float)
y_gender = torch.tensor(y_gender, dtype = torch.float)
y_country = torch.tensor(y_country, dtype = torch.float)
y_marital = torch.tensor(y_marital, dtype = torch.float)
y_live = torch.tensor(y_live, dtype = torch.float)
y_occupation = torch.tensor(y_occupation, dtype = torch.float)
y_sport = torch.tensor(y_sport, dtype = torch.float)
y_economic = torch.tensor(y_economic, dtype = torch.float)
y_smoke = torch.tensor(y_smoke, dtype = torch.float)
y_alchol = torch.tensor(y_alchol, dtype = torch.float)
y_premium = torch.tensor(y_premium, dtype = torch.float)
y_extra = torch.tensor(y_extra, dtype = torch.float)
y_agreeable = torch.tensor(y_agreeable, dtype = torch.float)
y_consc = torch.tensor(y_consc, dtype = torch.float)
y_neurot = torch.tensor(y_neurot, dtype = torch.float)
y_open = torch.tensor(y_open, dtype = torch.float)

# add edges 
# edge_index = None
# edge_attr = None 
edge_index = []
edge_attr = []

# df_res_similar = pd.DataFrame(columns = ["owner1", "owner2", "id_pl1", "id_pl2", "similarity"])

print("starting number2")

rows_list = []


# pl_list = pl_list[:40]

counter = 0

for i in tqdm(range(len(pl_list)-1)):
    for j in range(i+1,len(pl_list)):
        node1 = pl_list[i]
        node2 = pl_list[j]
        weight = common_member(playlist_dict[node1], playlist_dict[node2])

        own1 = dict_pl_user[node1]
        own2 = dict_pl_user[node2]

        rows_list.append({
            "owner1": own1,
            "owner2": own2, 
            "id_pl1": node1, 
            "id_pl2": node2,
            "similarity": weight
            })

        if weight != 0:
                counter +=1
                edge_index.append([i, j])
                edge_index.append([j, i])
                edge_attr.append([weight])
            # if edge_index == None:
            #     edge_index = torch.tensor([i,j],dtype = torch.long)
            #     edge_index = torch.stack((edge_index,torch.tensor([j,i],dtype=torch.long)), dim = 1)
            #     edge_attr = torch.tensor([weight], dtype=torch.long)
            #     edge_attr = torch.concat((edge_attr, torch.tensor([weight],dtype=torch.long)), dim = 0)
            # else:
            #     edge_index = torch.stack((edge_index,torch.tensor([i,j],dtype=torch.long)), dim = 1)
            #     edge_index = torch.stack((edge_index,torch.tensor([j,i],dtype=torch.long)), dim = 1)
            #     edge_attr = torch.stack((edge_attr, torch.tensor([weight], dtype = torch.long)), dim = 1)
            #     edge_attr = torch.stack((edge_attr, torch.tensor([weight], dtype = torch.long)), dim = 1)
print(counter)

df_res_similar = pd.DataFrame(rows_list)
df_res_similar.to_csv("df_res_similar.csv",index = False)


edge_index = torch.tensor(edge_index, dtype = torch.long)
edge_attr = torch.tensor(edge_attr, dtype = torch.float)

print(edge_attr)
print(edge_index)

data = Data(x=x, edge_index=edge_index.t().contiguous(), edge_attr = edge_attr)
data.y_age = y_age
data.y_gender = y_gender

data.y_country = y_country
data.y_marital = y_marital
data.y_live = y_live
data.y_occupation = y_occupation
data.y_sport = y_sport
data.y_economic = y_economic
data.y_smoke = y_smoke
data.y_alchol = y_alchol
data.y_premium = y_premium
data.y_extra = y_extra
data.y_agreeable = y_agreeable
data.y_consc = y_consc
data.y_neurot = y_neurot
data.y_open = y_open


data.id_users = id_users
data.id_playlists = id_playlists

torch.save(data, "data.pt")