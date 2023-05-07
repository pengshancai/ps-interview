import json
import numpy as np
from sumeval.metrics.rouge import RougeCalculator
from bert_score import score as bscore
from tqdm import tqdm

# result_dir = '../ps-interview_/results/text-davinci-003/'
result_dir = '../ps-interview_/results/gpt2_1.0/'
with open(result_dir + 'results.json', 'r') as f:
    results = json.load(f)

# Traditional Metrics
rouge = RougeCalculator(stopwords=True, lang="en")
rouge1s, rouge2s, rougels = [], [], []
for i, pred in enumerate(results['pred']):
    if pred == '':
        pred = 'none'
    refs = results['golds'][i]
    rouge1s.append(rouge.rouge_n(summary=pred, references=refs, n=1))
    rouge2s.append(rouge.rouge_n(summary=pred, references=refs, n=2))
    rougels.append(rouge.rouge_l(summary=pred, references=refs))

with open(result_dir + 'score.txt', 'a') as f:
    f.write('Rouge 1:\t%s\n' % np.mean(rouge1s))
    f.write('Rouge 2:\t%s\n' % np.mean(rouge2s))
    f.write('Rouge L:\t%s\n' % np.mean(rougels))

# BART Score
b_f1s = []
for i, pred in tqdm(enumerate(results['pred'])):
    print(i)
    if pred == '':
        pred = 'none'
    src = results['src'][i]
    refs = results['golds'][i]
    cands = [pred for _ in refs]
    _, _, b_f1 = bscore(cands, refs, lang='en', verbose=False)
    b_f1s.append(max(b_f1))

with open(result_dir + 'score.txt', 'a') as f:
    _ = f.write('BERT Score F1:\t%s\n' % np.mean(b_f1s))

