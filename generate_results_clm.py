import json
import re
from pathlib import Path
import argparse
import torch
from datasets import load_dataset
from transformers import (
    MODEL_MAPPING,
    AutoModelForCausalLM,
)
from transformers import AutoTokenizer, LlamaForCausalLM


MODEL_CONFIG_CLASSES = list(MODEL_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dump_name", type=str, required=True)
    parser.add_argument("--dump_dir", type=str, default='../ps-interview_/dump/')
    parser.add_argument("--result_dir", type=str, default='../ps-interview_/results/')
    parser.add_argument("--data_path", type=str, default='../ps-interview_/data/processed_clm/test.json')
    args = parser.parse_args()
    args.dump_dir = args.dump_dir + args.dump_name + '/'
    args.result_dir = args.result_dir + args.dump_name + '/'
    return args


def cut_results(src, pred):
    start_point = len(src + "Summarize the customer\'s issue in the above dialog in one sentence.")
    if 'END' in pred:
        end_point = pred.find('END')
        return pred[start_point: end_point].replace("\"", "").strip()
    else:
        pat_sum = re.compile("Summarize the customer\'s issue in the above dialog in one sentence\. \"(.+)\"")
        se = pat_sum.search(pred[start_point:])
        if se:
            return se.group(1)
        else:
            return ""


def main():
    args = parse_args()
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    # Load the pre-trained model
    model = AutoModelForCausalLM.from_pretrained(args.dump_dir)
    tokenizer = AutoTokenizer.from_pretrained(args.dump_dir)
    model.to(device)
    # Load test set
    data_files = {'test': args.data_path}
    extension = args.data_path.split(".")[-1]
    test_set = load_dataset(extension, data_files=data_files)['test']
    results = {
        'src': [],
        'pred': [],
        'golds': [],
    }
    for rec in test_set:
        src = rec['text']
        tgts = rec['golds']
        inputs = tokenizer(src + "Summarize the customer\'s issue in the above dialog in one sentence.", return_tensors='pt').to(device)
        outputs = model.generate(**inputs, max_length=1024)
        pred = tokenizer.decode(outputs.cpu().numpy()[0])
        pred = cut_results(src, pred)
        results['src'].append(src)
        results['pred'].append(pred)
        results['golds'].append(tgts)
    path = Path(args.result_dir)
    path.mkdir(parents=True)
    with open(args.result_dir + 'results.txt', 'w') as f:
        json.dump(results, f)


if __name__ == "__main__":
    main()

