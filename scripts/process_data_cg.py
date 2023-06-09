import os
import csv
import re
import jsonlines
from transformers import LlamaTokenizer

tokenizer = LlamaTokenizer.from_pretrained("decapoda-research/llama-7b-hf")

input_dir = '../ps-interview_/data/raw/'
output_dir = '../ps-interview_/data/processed_cg/'
if not os.path.exists(output_dir):
    os.mkdir(output_dir)


def process_line(line):
    pat_trash = re.compile('@[0-9]+')
    src_str = line[1].replace("\"", "'")
    tgt_str = line[2].replace("\"", "'")
    trashs = list(set(pat_trash.findall(src_str)))
    for trash in trashs:
        src_str = src_str.replace(trash, '')
    tgts = tgt_str.split('<sep>')
    return src_str, tgts


fnames = os.listdir(input_dir)
for fname in fnames:
    split = fname[11:-4]
    if split == 'test':
        continue
    with open(input_dir + fname, mode='r') as f_in:
        lines = [line for line in csv.reader(f_in)][1:]
    with jsonlines.open(output_dir + '%s.json' % split, 'w') as f:
        for line in lines:
            src, tgts = process_line(line)
            for tgt in tgts:
                _ = f.write({
                    'text': src,
                    'summary': tgt
                })


with open(input_dir + 'tweet_summ_test.csv', mode='r') as f_in:
    lines = [line for line in csv.reader(f_in)][1:]

with jsonlines.open(output_dir + 'test.json', 'w') as f:
    for line in lines:
        src, tgts = process_line(line)
        _ = f.write({
            'text': src,
            'golds': tgts
        })

