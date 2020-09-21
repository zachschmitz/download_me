
###     Making the Dataset         ###
import torch
from torch_geometric.data import InMemoryDataset
from torch_geometric.data import Data
import os.path as osp
import pandas as pd

class DevelopabilityDataset(InMemoryDataset): 

  #root='/Users/zachschmitz/Documents/GitHub/pytorch_geometric/August_Trial/MessagePassingDatasets' #where I want to save my dataset
  url='https://github.com/zachschmitz/download_me.git'

  def __init__(self,root,transform=None,pre_transform=None):
    super(DevelopabilityDataset,self).__init__(root,transform,pre_transform)
    self.data,self.slices=torch.load(self.processed_paths[0])

  @property
  def raw_file_names(self):
    print('reached raw_file_names')
    return ['short_graph_info.csv']

  @property
  def processed_file_names(self):
    pass
    #return ['data.pt']

  def download(self):
    #pass
    print('reached download')
    path=-download_url(self.url,self.raw_dir)
    extract_zip(path,self.raw_dir)
    os.unlink(path)

  def process(self):
    data_list=[]
    #df=pd.read_csv('raw_graph_info.csv')

    ##process by row index
    ##Initializing each graph Data object:
    for index, row in df.iterrows():
      x=df.loc[row,'Feature']
      edge_index=df.loc[row,'Connectivity']
      y=list(df.loc[row,'Sort1Score'],df.loc[row,'Sort8Score'],df.loc[row,'Sort10Score'])

      graph=Data(x=x,edge_index=edge_index.t().contiguous(),y=y)
      data_list.append(graph)

    print("I made it past the iteration")
    data,slices=self.collate(data_list)
    torch.save((data,slices),self.processed_paths[0])

a=DevelopabilityDataset(root='check')
print(a)