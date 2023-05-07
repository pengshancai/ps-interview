CUDA_VISIBLE_DEVICES=0 python run_gen.py \
    --dump_name llama-7b-hf \
    --dump_dir decapoda-research/llama-7b-hf \
    --peft_dir ../ps-interview_/dump/llama_1.0/ \
    --result_dir ../ps-interview_/results/llama-7b-hf/ \
    --data_path ../ps-interview_/data/processed_clm/test.json
