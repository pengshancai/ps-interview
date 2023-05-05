import os
import openai
from pathlib import Path
from tqdm import tqdm
from datasets import load_dataset
import time
import json

api_key_path = '../openai/api_key.txt'
output_path = "../ps-interview_/results/gpt3/"
with open(api_key_path) as f:
    openai.api_key = f.read().strip()

model_name = "text-davinci-003"
result_dir = '../ps-interview_/results/%s/' % model_name
data_path = '../ps-interview_/data/processed_clm/test.json'

data_files = {'test': data_path}
extension = data_path.split(".")[-1]
test_set = load_dataset(extension, data_files=data_files)['test']


def generate_response(src):
    try:
        prompt_template = "\"%s\" Summarize the customer\'s issue in the above dialog in one sentence."
        prompt = prompt_template % src
        response = openai.Completion.create(model=model_name, prompt=prompt, temperature=0, max_tokens=32)
        pred = response["choices"][0]['text'].strip().replace('\n', ' ')
        return pred
    except:
        print('Server error, restart in 5s')
        time.sleep(5)
        return generate_response(src)


results = {
    'src': [],
    'pred': [],
    'golds': [],
}
for rec in tqdm(test_set):
    src = rec['text']
    tgts = rec['golds']
    pred = generate_response(src)
    results['src'].append(src)
    results['pred'].append(pred)
    results['golds'].append(tgts)

path = Path(result_dir)
path.mkdir(parents=True)
with open(result_dir + 'results.json', 'w') as f:
    json.dump(results, f)
