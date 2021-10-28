'''
Prepare the Twitter Emotions dataset as tokenized torch
ids tensor and attention mask
'''

import torch
import torch.nn as nn
from transformers import ElectraTokenizer, BertTokenizer, RobertaTokenizer
from datasets import load_dataset

def get_data(arch):
    dataset = load_dataset('ag_news')
    print(dataset[0])

def get_train(arch, filepath='../data/train.txt'):
    return get_data(filepath, arch)

def get_test(arch, filepath='../data/test.txt'):
    return get_data(filepath, arch)