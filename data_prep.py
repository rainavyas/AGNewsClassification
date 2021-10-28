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
    train = dataset['train']
    test = dataset['test']
    print(train[:5])

def get_train(arch):
    return get_data(arch)

def get_test(arch):
    return get_data(arch)