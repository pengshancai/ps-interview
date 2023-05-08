import numpy as np
import csv

path = '../ps-interview_/data/raw/'


for split in ['train', 'valid', 'test']:
    with open(path + 'tweet_summ_%s.csv' % split) as f_in:
        lines = [line for line in csv.reader(f_in)][1:]
    ls_src, ls_tgt = [], []
    for line in lines:
        ls_src.append(len(line[1].split(' ')))
        for tgt in line[2].split('<sep>'):
            ls_tgt.append(len(tgt.split(' ')))
    print(split)
    print(len(ls_src))
    print(len(ls_tgt))
    print(np.mean(ls_src))
    print(np.mean(ls_tgt))
