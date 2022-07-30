import torch
from torch.utils.data import Dataset
from torchvision.transforms import transforms

from PIL import Image
from glob import glob
import numpy as np
import os
import random



# # If Omniglot dataset is not loaded yet:
# import torchvision
# dataset = torchvision.datasets.Omniglot(
#     root="", download=True, transform=torchvision.transforms.ToTensor()
# )



"""
Data folder tree:
omniglot-py -> alpha1   -> char1 -> 0.png, 1.png, ..., 20.png
                           char1 -> 0.png, 1.png, ..., 20.png
                           ...
            -> alpha2 -> char1 -> ...
            ...

"""

#############################################################################################################
# Preprocess DATASET:
#############################################################################################################
def meta_preprocess(root):
    dictTrain = {}
    dataset_indx = 0
    datasets_dir = glob(os.path.join(root, '*'))
    datasets_dir.sort()
    
    print('All datasets are from: {}'.format(datasets_dir))
    for dir in datasets_dir :
        if (os.path.isdir(dir)):
            sub_dirs = glob(os.path.join( dir, '*'))
            for sub_dir in sub_dirs:
                if (os.path.isdir(sub_dir)):
                    train_paths = glob(os.path.join( sub_dir, '*'))
                    np.random.shuffle(train_paths)

                    for filename in train_paths:
                        if dataset_indx in dictTrain.keys():
                            dictTrain[dataset_indx].append(filename)
                        else:
                            dictTrain[dataset_indx] = [filename]

                    dataset_indx = dataset_indx + 1


    dataset_num = dataset_indx
    print('Finished preprocessing the datasets...')
    print('Overall dataset number : {}'.format(dataset_num))
    return dictTrain, dataset_num



#############################################################################################################
# Meta DataSet:
#############################################################################################################
class MetaDataloader(Dataset):
    def __init__(self, k_shot, k_query, resize, dataset_num, test_indx_list, dictTrain):

        self.k_shot = k_shot  
        self.k_query = k_query 
        self.resize = resize  
        self.dataset_num = dataset_num
        self.dictTrain = dictTrain

        print('shuffle %d-shot, %d-query, resize:%d' % ( k_shot, k_query, resize))

        self.transform = transforms.Compose([lambda x: Image.open(x).convert('RGB'),
                                             transforms.Resize( (self.resize, self.resize)),
                                             transforms.ToTensor(),
                                             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                            ])
        
        dataset_num_list = list(range(self.dataset_num))
        for id in test_indx_list:
            dataset_num_list.pop( id)
        self.create_batch( dataset_num_list )
            
    def create_batch(self, dataset_num_list ):
        """
        create batch for meta-learning.
        """
        length = self.k_shot + self.k_query
        self.dict_data = {}

        cntr = 0
        for n in dataset_num_list:
            Ts = self.dictTrain[n]
            random.shuffle( Ts)
            Nbr = len(Ts)//length
            for i in range(Nbr):
                self.dict_data[cntr] = {'spt': Ts[ i*length:i*length+self.k_shot],
                                        'qry': Ts[ i*length+self.k_shot:(i+1)*length]}
                cntr += 1

        self.size = cntr
        print('Finished create batches of the datasets...')



    def __getitem__(self, index):
        # [setsz, 3, resize, resize]
        support = torch.FloatTensor(self.k_shot, 3, self.resize, self.resize)
        query = torch.FloatTensor(self.k_query, 3, self.resize, self.resize)

        flatten_support = self.dict_data[index]['spt']
        flatten_query = self.dict_data[index]['qry']


        for i, path in enumerate(flatten_support):
            support[i] = self.transform(path)
        for i, path in enumerate(flatten_query):
            query[i] = self.transform(path)

        return support, query

    def __len__(self):
        return self.size



#############################################################################################################
# FineTuning DataSet:
#############################################################################################################
class FineTuningDataloader(Dataset):
    def __init__(self, k_shot, k_query, resize, dataset_num, test_dataset_indx, dictTrain):

        self.k_shot = k_shot  
        self.k_query = k_query  
        self.resize = resize  
        self.dataset_num = dataset_num
        self.dictTrain = dictTrain

        print('shuffle %d-shot, %d-query, resize:%d' % ( k_shot, k_query, resize))

        self.transform = transforms.Compose([lambda x: Image.open(x).convert('RGB'),
                                             transforms.Resize( (self.resize, self.resize)),
                                             transforms.ToTensor(),
                                             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                            ])
       
        self.create_batch( test_dataset_indx )

    
            
    def create_batch(self, test_dataset_indx ):
        """
        create batch for meta-learning.
        """
        length = self.k_shot + self.k_query
        self.dict_data = {}

        cntr = 0
        Ts = self.dictTrain[test_dataset_indx]
        random.shuffle( Ts)
        Nbr = len(Ts)//length 
        for i in range(Nbr):
            self.dict_data[cntr] = {'spt': Ts[ i*length:i*length+self.k_shot],
                                    'qry': Ts[ i*length+self.k_shot:(i+1)*length]}
            cntr += 1

        self.size = cntr
        print('Finished create batches of the datasets...')



    def __getitem__(self, index):

        # [setsz, 3, resize, resize]
        support = torch.FloatTensor(self.k_shot, 3, self.resize, self.resize)
        query = torch.FloatTensor(self.k_query, 3, self.resize, self.resize)

        flatten_support = self.dict_data[index]['spt']
        flatten_query = self.dict_data[index]['qry']

        for i, path in enumerate(flatten_support):
            support[i] = self.transform(path)
        for i, path in enumerate(flatten_query):
            query[i] = self.transform(path)

        return support, query


    def __len__(self):
        return self.size

