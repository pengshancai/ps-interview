# CUDA_VISIBLE_DEVICES=0 python gen_res.py \
#     --dump_name llama-7b-hf \
#     --dump_dir decapoda-research/llama-7b-hf \
#     --peft_dir ../ps-interview_/dump/llama_1.0/ \
#     --result_dir ../ps-interview_/results/llama-7b-hf/ \
#     --data_path ../ps-interview_/data/processed_clm/test.json

CUDA_VISIBLE_DEVICES=0 python gen_res.py \
    --dump_dir decapoda-research/llama-7b-hf \
    --peft_dir ../ps-interview_/dump/llama_1.1/epoch_0 \
    --result_dir ../ps-interview_/results/llama_1.1/epoch_0 \
    --data_path ../ps-interview_/data/processed_clm/test.json

CUDA_VISIBLE_DEVICES=0 python gen_res.py \
    --dump_dir decapoda-research/llama-7b-hf \
    --peft_dir ../ps-interview_/dump/llama_1.1/epoch_1 \
    --result_dir ../ps-interview_/results/llama_1.1/epoch_1 \
    --data_path ../ps-interview_/data/processed_clm/test.json

CUDA_VISIBLE_DEVICES=0 python gen_res.py \
    --dump_dir decapoda-research/llama-7b-hf \
    --peft_dir ../ps-interview_/dump/llama_1.1/epoch_2 \
    --result_dir ../ps-interview_/results/llama_1.1/epoch_2 \
    --data_path ../ps-interview_/data/processed_clm/test.json

CUDA_VISIBLE_DEVICES=0 python gen_res.py \
    --dump_dir decapoda-research/llama-7b-hf \
    --peft_dir ../ps-interview_/dump/llama_1.1/epoch_3 \
    --result_dir ../ps-interview_/results/llama_1.1/epoch_3 \
    --data_path ../ps-interview_/data/processed_clm/test.json

CUDA_VISIBLE_DEVICES=0 python gen_res.py \
    --dump_dir decapoda-research/llama-7b-hf \
    --peft_dir ../ps-interview_/dump/llama_1.1/epoch_4 \
    --result_dir ../ps-interview_/results/llama_1.1/epoch_4 \
    --data_path ../ps-interview_/data/processed_clm/test.json

CUDA_VISIBLE_DEVICES=0 python gen_res.py \
    --dump_dir decapoda-research/llama-7b-hf \
    --peft_dir ../ps-interview_/dump/llama_1.1/epoch_5 \
    --result_dir ../ps-interview_/results/llama_1.1/epoch_5 \
    --data_path ../ps-interview_/data/processed_clm/test.json

CUDA_VISIBLE_DEVICES=0 python gen_res.py \
    --dump_dir decapoda-research/llama-7b-hf \
    --peft_dir ../ps-interview_/dump/llama_1.1/epoch_6 \
    --result_dir ../ps-interview_/results/llama_1.1/epoch_6 \
    --data_path ../ps-interview_/data/processed_clm/test.json

CUDA_VISIBLE_DEVICES=0 python gen_res.py \
    --dump_dir decapoda-research/llama-7b-hf \
    --peft_dir ../ps-interview_/dump/llama_1.1/epoch_7 \
    --result_dir ../ps-interview_/results/llama_1.1/epoch_7 \
    --data_path ../ps-interview_/data/processed_clm/test.json

CUDA_VISIBLE_DEVICES=0 python gen_res.py \
    --dump_dir decapoda-research/llama-7b-hf \
    --peft_dir ../ps-interview_/dump/llama_1.1/epoch_8 \
    --result_dir ../ps-interview_/results/llama_1.1/epoch_8 \
    --data_path ../ps-interview_/data/processed_clm/test.json

CUDA_VISIBLE_DEVICES=0 python gen_res.py \
    --dump_dir decapoda-research/llama-7b-hf \
    --peft_dir ../ps-interview_/dump/llama_1.1/epoch_9 \
    --result_dir ../ps-interview_/results/llama_1.1/epoch_9 \
    --data_path ../ps-interview_/data/processed_clm/test.json

CUDA_VISIBLE_DEVICES=0 python gen_res.py \
    --dump_dir decapoda-research/llama-7b-hf \
    --peft_dir ../ps-interview_/dump/llama_1.1/epoch_10 \
    --result_dir ../ps-interview_/results/llama_1.1/epoch_10 \
    --data_path ../ps-interview_/data/processed_clm/test.json

CUDA_VISIBLE_DEVICES=0 python gen_res.py \
    --dump_dir decapoda-research/llama-7b-hf \
    --peft_dir ../ps-interview_/dump/llama_1.1/epoch_11 \
    --result_dir ../ps-interview_/results/llama_1.1/epoch_11 \
    --data_path ../ps-interview_/data/processed_clm/test.json

CUDA_VISIBLE_DEVICES=0 python gen_res.py \
    --dump_dir decapoda-research/llama-7b-hf \
    --peft_dir ../ps-interview_/dump/llama_1.1/epoch_3 \
    --result_dir ../ps-interview_/results/llama_1.1/epoch_3/attempt1/ \
    --data_path ../ps-interview_/data/processed_clm/test.json

CUDA_VISIBLE_DEVICES=0 python eval_res.py \
    --result_dir ../ps-interview_/results/llama_1.1/epoch_3/attempt1/

CUDA_VISIBLE_DEVICES=0 python gen_res.py \
    --dump_dir decapoda-research/llama-7b-hf \
    --peft_dir ../ps-interview_/dump/llama_1.1/epoch_3 \
    --result_dir ../ps-interview_/results/llama_1.1/epoch_3/attempt2/ \
    --data_path ../ps-interview_/data/processed_clm/test.json

CUDA_VISIBLE_DEVICES=0 python eval_res.py \
    --result_dir ../ps-interview_/results/llama_1.1/epoch_3/attempt2/

CUDA_VISIBLE_DEVICES=0 python gen_res.py \
    --dump_dir decapoda-research/llama-7b-hf \
    --peft_dir ../ps-interview_/dump/llama_1.1/epoch_3 \
    --result_dir ../ps-interview_/results/llama_1.1/epoch_3/attempt3/ \
    --data_path ../ps-interview_/data/processed_clm/test.json

CUDA_VISIBLE_DEVICES=0 python eval_res.py \
    --result_dir ../ps-interview_/results/llama_1.1/epoch_3/attempt3/

CUDA_VISIBLE_DEVICES=0 python gen_res.py \
    --dump_dir decapoda-research/llama-7b-hf \
    --peft_dir ../ps-interview_/dump/llama_1.1/epoch_3 \
    --result_dir ../ps-interview_/results/llama_1.1/epoch_3/attempt4/ \
    --data_path ../ps-interview_/data/processed_clm/test.json

CUDA_VISIBLE_DEVICES=0 python eval_res.py \
    --result_dir ../ps-interview_/results/llama_1.1/epoch_3/attempt4/
