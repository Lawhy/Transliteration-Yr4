# +
import os
from argparse import Namespace
from argparse import ArgumentParser
from collections import Counter
import json
import re
import string

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.data import Dataset, DataLoader

from nltk.translate import bleu_score
import seaborn as sns
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET

# my own module
from utils.batch import *
from nmt.model import NMTModel
from utils.dataset import NMTDataset

chencherry = bleu_score.SmoothingFunction()


# -

class Translator:
    
    def __init__(self, args_path):
        args = Namespace()
        with open(args_path, 'r') as f:
            args.__dict__ = json.load(f)
        self.args = args      
        self.seed = args.seed
        self.cuda = args.cuda
        self.exp_num = args.exp_num
        
        
    def set_dataset_and_vectorizer(self, 
                               tra_src, tra_tgt, 
                               val_src, val_tgt, 
                               tst_src, tst_tgt, 
                               vectorizer_file):
        
        self.dataset = NMTDataset.load_dataset_and_load_vectorizer(tra_src, tra_tgt, 
                                                                   val_src, val_tgt, 
                                                                   tst_src, args.tst_tgt, 
                                                                   vectorizer_file)
        self.vectorizer = self.dataset.get_vectorizer()
        
    def set_model(self, 
         source_vocab_size, source_embedding_size, 
         target_vocab_size, target_embedding_size, 
         encoding_size, decoding_size,
         encoder_layers, decoder_layers,
         encoder_dropout, decoder_dropout,
         target_bos_index,
         # parameters for loading existing models
         reload_from_files=True, model_state_file=None):
        
        if self.vectorizer: 
            self.model = NMTModel(source_vocab_size=source_vocab_size, source_embedding_size=source_embedding_size, 
                                 target_vocab_size=target_vocab_size, target_embedding_size=target_embedding_size, 
                                 encoding_size=encoding_size, decoding_size=decoding_size,
                                 encoder_layers=encoder_layers, decoder_layers=decoder_layers,
                                 encoder_dropout=encoder_dropout, decoder_dropout = decoder_dropout,
                                 target_bos_index=target_bos_index)
        else:
            print("Warning! Please load the dataset first!")

        if reload_from_files and os.path.exists(model_state_file):
            self.model.load_state_dict(torch.load(model_state_file))
            print("Reloaded model")
        else:
            self.model = None
            print("No existing model to be loaded!")
            
    def set_device(self):
        # Check CUDA
        if not torch.cuda.is_available():
            self.cuda = False

        self.device = torch.device("cuda" if self.cuda else "cpu")
    
        print("Using CUDA: {}".format(self.cuda))
    
    def set_seed_everywhere(self):
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if self.cuda:
            torch.cuda.manual_seed_all(self.seed)
    
    @staticmethod
    def sentence_from_indices(indices, vocab, strict=True, return_string=True):
        ignore_indices = set([vocab.mask_index, vocab.begin_seq_index, vocab.end_seq_index])
        out = []
        for index in indices:
            if index == vocab.begin_seq_index and strict:
                continue
            elif index == vocab.end_seq_index and strict:
                break
            else:
                out.append(vocab.lookup_index(index))
        if return_string:
            return " ".join(out)
        else:
            return out
    
    def apply_to_batch(self, batch_dict):
        self._last_batch = batch_dict
        y_pred = self.model(x_source=batch_dict['x_source'], 
                            x_source_lengths=batch_dict['x_source_length'], 
                            target_sequence=batch_dict['x_target'])
        self._last_batch['y_pred'] = y_pred
        
        attention_batched = np.stack(self.model.decoder._cached_p_attn).transpose(1, 0, 2)
        self._last_batch['attention'] = attention_batched
        
    def _get_source_sentence(self, index, return_string=True):
        indices = self._last_batch['x_source'][index].cpu().detach().numpy()
        vocab = self.vectorizer.source_vocab
        return self.sentence_from_indices(indices, vocab, return_string=return_string)

    def _get_reference_sentence(self, index, return_string=True):
        indices = self._last_batch['y_target'][index].cpu().detach().numpy()
        vocab = self.vectorizer.target_vocab
        return self.sentence_from_indices(indices, vocab, return_string=return_string)
    
    def _get_sampled_sentence(self, index, return_string=True):
        _, all_indices = torch.max(self._last_batch['y_pred'], dim=2)
        sentence_indices = all_indices[index].cpu().detach().numpy()
        vocab = self.vectorizer.target_vocab
        return self.sentence_from_indices(sentence_indices, vocab, return_string=return_string)

    
    # that's for bleu-score
    def get_ith_item(self, index, return_string=True):
        output = {"source": self._get_source_sentence(index, return_string=return_string), 
                  "reference": self._get_reference_sentence(index, return_string=return_string), 
                  "sampled": self._get_sampled_sentence(index, return_string=return_string),
                  "attention": self._last_batch['attention'][index]}
        
        reference = output['reference']
        hypothesis = output['sampled']
        
        if not return_string:
            reference = " ".join(reference)
            hypothesis = " ".join(hypothesis)
        
        output['bleu-4'] = bleu_score.sentence_bleu(references=[reference],
                                                    hypothesis=hypothesis,
                                                    smoothing_function=chencherry.method1)
        
        return output
    
    # dealing with testing data of multiple answers, specifically for the NEWS data
    def get_test_multi_ans(self, test_xml):
        tree = ET.parse(test_xml)
        root = tree.getroot()
        multi_ans = pd.DataFrame(columns=['src', 'tgt'])
        j=0
        for child in root:
            src = child.find('SourceName').text
            names = child.findall('TargetName')
            if len(names) > 1:
                tgt = [names[i].text for i in range(len(names))]
                multi_ans.loc[j] = [' '.join(src), tgt]
                j+=1
        self.test_multi_ans = multi_ans
        
        
    def translate(self, split='test', batch_size=None):
        
        self.model = self.model.eval().to(self.device)

        self.dataset.set_split(split)
        
        if not batch_size:
            batch_size = self.dataset.__len__()
        
        batch_generator = generate_nmt_batches(self.dataset, 
                                               batch_size=batch_size, 
                                               device=self.device)

        test_results = pd.DataFrame(columns=['src', 'tgt', 'pred'])
        for batch_dict in batch_generator:
            self.apply_to_batch(batch_dict)
            for i in range(batch_size):
                source = self._get_source_sentence(i).lower()
                target = self._get_reference_sentence(i)
                pred = self._get_sampled_sentence(i)
                print('[Source]:', source)
                print('[Target]:', target)
                print('[Pred]:', pred)
                print('----------------------')
                test_results.loc[i] = [source, target, pred]
                
        self.test_results = test_results
        
        
    def compute_acc(self, multi=True):
        
        ori_acc = np.sum(self.test_results['tgt'] == self.test_results['pred']) / len(self.test_results)
        
        # for watching acc other than test set
        if not multi:
            return ori_acc
        
        # init correct with those with single answers
        correct = np.sum(self.test_results[~self.test_results['src'].isin(self.test_multi_ans.src)]['tgt'] 
                         == self.test_results[~self.test_results['src'].isin(self.test_multi_ans.src)]['pred'])
        
        extra = 0
        for ind, dp in self.test_results[self.test_results['src'].isin(self.test_multi_ans.src)].iterrows():
            src = dp['src']
            pred = dp['pred']
            if pred.replace(' ', '') in list(self.test_multi_ans[self.test_multi_ans['src'] == src]['tgt'])[0]:
                extra += 1
                
        correct += extra
        acc = correct / len(self.test_results)
        
        return ori_acc, acc

translator = Translator('experiments/exp1/args.json')
translator.set_device()
# Set seed for reproducibility
translator.set_seed_everywhere()

args = translator.args
translator.set_dataset_and_vectorizer(args.tra_src, args.tra_tgt, 
                                      args.val_src, args.val_tgt, 
                                      args.tst_src, args.tst_tgt, 
                                      args.vectorizer_file)

translator.set_model(source_vocab_size=len(translator.vectorizer.source_vocab), 
                     source_embedding_size=args.source_embedding_size, 
                     target_vocab_size=len(translator.vectorizer.target_vocab),
                     target_embedding_size=args.target_embedding_size, 
                     encoding_size=args.encoding_size,
                     decoding_size=args.decoding_size,
                     encoder_layers=args.encoder_layers,
                     decoder_layers=args.decoder_layers,
                     encoder_dropout=args.encoder_dropout,
                     decoder_dropout=args.decoder_dropout,
                     target_bos_index=translator.vectorizer.target_vocab.begin_seq_index,
                     reload_from_files=args.reload_from_files,
                     model_state_file=args.model_state_file)

translator.translate(split='test')

translator.get_test_multi_ans('data/en2ch.dev.xml')
translator.compute_acc()

translator.translate(split='train', batch_size=12)

translator.compute_acc(multi=False)
