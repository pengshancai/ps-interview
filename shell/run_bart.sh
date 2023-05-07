CUDA_VISIBLE_DEVICES=6 python run_cg.py \
    --model_name_or_path facebook/bart-large \
    --train_file ../ps-interview_/data/processed_cg/train.json \
    --validation_file ../ps-interview_/data/processed_cg/valid.json \
    --output_dir ../ps-interview_/dump/bart-large/ \
    --text_column text \
    --summary_column summary \
    --num_train_epochs 5 \
    --num_beams 3


CUDA_VISIBLE_DEVICES=6 python run_cg.py \
    --model_name_or_path facebook/bart-large \
    --train_file ../ps-interview_/data/processed_cg/train.json \
    --validation_file ../ps-interview_/data/processed_cg/valid.json \
    --output_dir ../ps-interview_/dump/bart-large_1.1/ \
    --text_column text \
    --summary_column summary \
    --num_train_epochs 3 \
    --num_beams 3


