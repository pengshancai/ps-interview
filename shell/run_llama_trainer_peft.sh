CUDA_VISIBLE_DEVICES=0 python run_clm_trainer.py \
    --model_name_or_path decapoda-research/llama-7b-hf \
    --train_file ../ps-interview_/data/processed_clm/train.json \
    --validation_file ../ps-interview_/data/processed_clm/valid.json \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --do_train \
    --do_eval \
    --output_dir ../ps-interview_/dump/llama_1.0/ \
    --num_train_epochs 5 \
    --do_lora \
    --max_length 100 \
    --overwrite_output_dir \
    --fp16

