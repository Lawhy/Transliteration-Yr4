"""
The code in this file is largely based on the book: 
   ##########################################################################
   # Natural Language Processing with PyTorch, by Delip Rao & Brian McMahan #
   ##########################################################################
I modified the code to fit the purpose of the Transliteration project.
"""

from utils.vectorizer import *
import numpy as np
import pandas as pd
import json
from torch.utils.data import Dataset, DataLoader

# #####################################################################
# ############################ Dataset ################################
# #####################################################################

class NMTDataset(Dataset):
    def __init__(self, vectorizer, tra_df, val_df, tst_df):
        """
        Args:
            dataset_df (pandas.DataFrame): the dataset
            vectorizer (NMTVectorizer): vectorizer instatiated from dataset
        """
        self._vectorizer = vectorizer

        self.train_df = tra_df
        self.train_size = len(self.train_df)

        self.val_df = val_df
        self.validation_size = len(self.val_df)

        self.test_df = tst_df
        self.test_size = len(self.test_df)

        self._lookup_dict = {'train': (self.train_df, self.train_size),
                             'val': (self.val_df, self.validation_size),
                             'test': (self.test_df, self.test_size)}

        self.set_split('train')  # by-default, the split is the trining set

    @classmethod
    def load_dataset_and_make_vectorizer(cls, tra_src, tra_tgt, val_src, val_tgt, tst_src, tst_tgt):
        """Load dataset and make a new vectorizer from scratch
        
        Args:
            paths to each dataset file
        Returns:
            an instance of NMTDataset
        """
        
        tra_df = NMTVectorizer.text_to_df(tra_src, tra_tgt)
        val_df = NMTVectorizer.text_to_df(val_src, val_tgt)
        tst_df = NMTVectorizer.text_to_df(tst_src, tst_tgt)
        
        return cls(NMTVectorizer.from_text(tra_src, tra_tgt), tra_df, val_df, tst_df)

    @classmethod
    def load_dataset_and_load_vectorizer(cls, tra_src, tra_tgt, val_src, val_tgt, tst_src, tst_tgt, vectorizer_filepath):
        """Load dataset and the corresponding vectorizer. 
        Used in the case in the vectorizer has been cached for re-use
        
        Args:
            tra/val/tst_src/tgt (str): paths to each dataset file
            vectorizer_filepath (str): location of the saved vectorizer
        Returns:
            an instance of NMTDataset
        """
        vectorizer = cls.load_vectorizer_only(vectorizer_filepath)
        
        tra_df = NMTVectorizer.text_to_df(tra_src, tra_tgt)
        val_df = NMTVectorizer.text_to_df(val_src, val_tgt)
        tst_df = NMTVectorizer.text_to_df(tst_src, tst_tgt)
        
        return cls(vectorizer, tra_df, val_df, tst_df)

    @staticmethod
    def load_vectorizer_only(vectorizer_filepath):
        """a static method for loading the vectorizer from file
        
        Args:
            vectorizer_filepath (str): the location of the serialized vectorizer
        Returns:
            an instance of NMTVectorizer
        """
        with open(vectorizer_filepath) as fp:
            return NMTVectorizer.from_serializable(json.load(fp))

    def save_vectorizer(self, vectorizer_filepath):
        """saves the vectorizer to disk using json
        
        Args:
            vectorizer_filepath (str): the location to save the vectorizer
        """
        with open(vectorizer_filepath, "w+") as fp:
            json.dump(self._vectorizer.to_serializable(), fp)

    def get_vectorizer(self):
        """ returns the vectorizer """
        return self._vectorizer

    def set_split(self, split="train"):
        self._target_split = split
        self._target_df, self._target_size = self._lookup_dict[split]

    def __len__(self):
        return self._target_size

    def __getitem__(self, index):
        """the primary entry point method for PyTorch datasets (must be implemented)
        
        Args:
            index (int): the index to the data point 
        Returns:
            a dictionary holding the data point: (x_data, y_target, class_index)
        """
        row = self._target_df.iloc[index]

        vector_dict = self._vectorizer.vectorize(row.source_language, row.target_language)

        return {"x_source": vector_dict["source_vector"], 
                "x_target": vector_dict["target_x_vector"],
                "y_target": vector_dict["target_y_vector"], 
                "x_source_length": vector_dict["source_length"]}
        
    def get_num_batches(self, batch_size):
        """Given a batch size, return the number of batches in the dataset
        
        Args:
            batch_size (int)
        Returns:
            number of batches in the dataset
        """
        return len(self) // batch_size

