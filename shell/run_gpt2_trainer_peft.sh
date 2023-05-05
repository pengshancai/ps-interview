CUDA_VISIBLE_DEVICES=7 python run_clm_trainer.py \
    --model_name_or_path gpt2 \
    --train_file ../ps-interview_/data/processed_clm/train.json \
    --validation_file ../ps-interview_/data/processed_clm/valid.json \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --do_train \
    --do_eval \
    --output_dir ../ps-interview_/dump/gpt2_1.3/ \
    --num_train_epochs 5 \
    --do_lora \
    --max_length 100 \
    --fp16

