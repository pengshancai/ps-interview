CUDA_VISIBLE_DEVICES=6 python gen_res.py \
    --dump_dir ../ps-interview_/dump/bart-large_1.5/epoch_0/ \
    --result_dir ../ps-interview_/results/bart_large_1.5/epoch_0/ \
    --data_path ../ps-interview_/data/processed_cg/test.json

CUDA_VISIBLE_DEVICES=6 python eval_res.py \
    --result_dir ../ps-interview_/results/bart-large_1.5/epoch_0/

CUDA_VISIBLE_DEVICES=6 python gen_res.py \
    --dump_dir ../ps-interview_/dump/bart-large_1.5/epoch_1/ \
    --result_dir ../ps-interview_/results/bart_large_1.5/epoch_1/ \
    --data_path ../ps-interview_/data/processed_cg/test.json

CUDA_VISIBLE_DEVICES=6 python eval_res.py \
    --result_dir ../ps-interview_/results/bart-large_1.5/epoch_1/

CUDA_VISIBLE_DEVICES=6 python gen_res.py \
    --dump_dir ../ps-interview_/dump/bart-large_1.5/epoch_2/ \
    --result_dir ../ps-interview_/results/bart_large_1.5/epoch_2/ \
    --data_path ../ps-interview_/data/processed_cg/test.json

CUDA_VISIBLE_DEVICES=6 python eval_res.py \
    --result_dir ../ps-interview_/results/bart-large_1.5/epoch_2/

CUDA_VISIBLE_DEVICES=6 python gen_res.py \
    --dump_dir ../ps-interview_/dump/bart-large_1.5/epoch_3/ \
    --result_dir ../ps-interview_/results/bart_large_1.5/epoch_3/ \
    --data_path ../ps-interview_/data/processed_cg/test.json

CUDA_VISIBLE_DEVICES=6 python eval_res.py \
    --result_dir ../ps-interview_/results/bart-large_1.5/epoch_3/

CUDA_VISIBLE_DEVICES=6 python gen_res.py \
    --dump_dir ../ps-interview_/dump/bart-large_1.5/epoch_4/ \
    --result_dir ../ps-interview_/results/bart_large_1.5/epoch_4/ \
    --data_path ../ps-interview_/data/processed_cg/test.json

CUDA_VISIBLE_DEVICES=6 python eval_res.py \
    --result_dir ../ps-interview_/results/bart-large_1.5/epoch_4/
