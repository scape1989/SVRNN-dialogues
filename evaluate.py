from __future__ import print_function

import random
import os
import time
import argparse
import sys
import inspect

import pickle as pkl
import torch
from torch import nn, optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from beeprint import pp

from models.linear_vrnn import LinearVRNN
from data_apis.data_utils import SWDADataLoader
from data_apis.SWDADialogCorpus import SWDADialogCorpus
from utils.loss import print_loss
import params


def main():
    data = pkl.load(open("data/cambridge_data/cambridge_data.pkl", "rb"))
    d = pkl.load(open("data/simdial/bus-CleanSpec-2000.pkl", "rb"))
    c = pkl.load(open("data/cambridge_data/data_DSTC2_with_label.pkl", "rb"))

    print("ts")
    # data = pkl.load(open("data/weather_data/simdial_weather_data.pkl", "rb"))
    # api = SWDADialogCorpus("data/cambridge_data/cambridge_data.pkl",
    #                        labeled=True)
    # dial_corpus = api.get_dialog_corpus()

    # labeled_dial = dial_corpus.get("labeled")

    # labeled_dial_labels = api.get_state_corpus(params.max_dialog_len)['labeled']
    # print(labeled_dial_labels)
    # labeled_dial_labels = api.get_state_corpus(params.max_dialog_len)['labeled']
    # dial = api.get_dialog_corpus()

    # train_loader = SWDADataLoader("Train",
    #                             train_dial,
    #                             params.max_utt_len,
    #                             params.max_dialog_len,
    #                             device='cpu')

    # # convert to numeric input outputs
    # labeled_loader = SWDADataLoader("Labeled",
    #                                 labeled_dial,
    #                                 params.max_utt_len,
    #                                 params.max_dialog_len,
    #                                 device="cpu")

    # labeled_loader.epoch_init(len(labeled_loader), shuffle=False)

    # while True:
    #     labeled_batch = labeled_loader.next_batch()
    #     if labeled_batch is None:
    #         break
    #     labeled_usr_input_sent, labeled_sys_input_sent, labeled_dialog_len_mask, labeled_usr_full_mask, labeled_sys_full_mask = labeled_batch


if __name__ == "__main__":
    main()