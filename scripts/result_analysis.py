import json
import numpy as np

dir = '../ps-interview_/results/'
folders = ['bart-large_1.2/epoch_0/', 'bart-large_1.3/epoch_2/', 'bart-large_1.5/epoch_1/', 'llama_1.1/attempt1/', 'text-davinci-003/']
ls = []
results = {}
for folder in folders:
    with open(dir + folder + 'results.json') as f:
        lines = json.load(f)
    for line in lines['pred']:
        ls.append(len(line.split(' ')))
    print('%s:\t%s' % (folder, np.mean(ls)))
    results[folder.split('/')[0]] = lines


def compare_instance(idx):
    print('Src:\t%s' % results['bart-large_1.2']['src'][idx])
    print('Gold:\t%s' % results['bart-large_1.2']['golds'][idx])
    for key, result in results.items():
        print('%s:\t%s' % (key, result['pred'][idx]))


