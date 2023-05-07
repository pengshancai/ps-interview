CUDA_VISIBLE_DEVICES=6 python run_sum.py \
    --model_name_or_path facebook/bart-large \
    --train_file ../ps-interview_/data/processed_clm/train.json \
    --validation_file ../ps-interview_/data/processed_clm/valid.json \
    --output_dir ../ps-interview_/dump/bart-large/ \
    --text_column text \
    --summary_column summary \
    --num_train_epochs 5 \


