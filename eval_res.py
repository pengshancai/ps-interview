import json
import numpy as np
import argparse
from tqdm import tqdm
from sumeval.metrics.rouge import RougeCalculator
from bert_score import score as bscore
from utils.unieval_utils import get_evaluator, convert_to_json
from collections import defaultdict



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_dir", type=str, default='../ps-interview_/results/')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    with open(args.result_dir + 'results.json', 'r') as f:
        results = json.load(f)

    # ROUGE Scores
    rouge = RougeCalculator(stopwords=True, lang="en")
    rouge1s, rouge2s, rougels = [], [], []
    for i, pred in enumerate(results['pred']):
        if pred == '':
            pred = 'none'
        pred = pred.replace('<s>', '')
        pred = pred.replace('</s>', '')
        refs = results['golds'][i]
        rouge1s.append(rouge.rouge_n(summary=pred, references=refs, n=1))
        rouge2s.append(rouge.rouge_n(summary=pred, references=refs, n=2))
        rougels.append(rouge.rouge_l(summary=pred, references=refs))

    with open(args.result_dir + 'score.txt', 'w') as f:
        _ = f.write('Rouge 1:\t%s\n' % np.mean(rouge1s))
        _ = f.write('Rouge 2:\t%s\n' % np.mean(rouge2s))
        _ = f.write('Rouge L:\t%s\n' % np.mean(rougels))

    # BERT Score
    b_ps, b_rs, b_f1s = [], [], []
    for i, pred in tqdm(enumerate(results['pred'])):
        print(i)
        if pred == '':
            pred = 'none'
        pred = pred.replace('<s>', '')
        pred = pred.replace('</s>', '')
        src = results['src'][i]
        refs = results['golds'][i]
        cands = [pred for _ in refs]
        p, r, f1 = bscore(cands, refs, lang='en', verbose=False)
        b_ps.append(max(p))
        b_rs.append(max(r))
        b_f1s.append(max(f1))

    with open(args.result_dir + 'score.txt', 'a') as f:
        _ = f.write('BERT Score Precision:\t%s\n' % np.mean(b_ps))
        _ = f.write('BERT Score Recall:\t%s\n' % np.mean(b_rs))
        _ = f.write('BERT Score F1:\t%s\n' % np.mean(b_f1s))

    # UniEval
    task = 'summarization'
    evaluator = get_evaluator(task)

    uni_scores = defaultdict(list)
    for i, pred in enumerate(results['pred']):
        print(i)
        if pred == '':
            pred = 'none'
        pred = pred.replace('<s>', '')
        pred = pred.replace('</s>', '')
        src_list = [results['src'][i]]
        ref_list = results['golds'][i]
        output_list = [pred]
        data = convert_to_json(output_list=output_list,
                               src_list=src_list,
                               ref_list=ref_list)
        scores = evaluator.evaluate(data)
        for metric, score in scores[0].items():
            uni_scores[metric].append(score)

    with open(args.result_dir + 'score.txt', 'a') as f:
        for metric, scores in uni_scores.items():
            _ = f.write('UniEval %s:\t%s\n' % (metric, np.mean(scores)))


if __name__ == "__main__":
    main()
