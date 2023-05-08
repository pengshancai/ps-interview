# Exploration with TWEETSUMM

## Reformat Datasets
Please refer to the following python files:
```
python scripts/process_data_cg.py # Process data for conditional generation
python scripts/process_data_clm.py # Process data for causal language modeling
```

## Training BART
```
python run_cg.py \
    --model_name_or_path facebook/bart-large \
    --train_file path_to_train.json \
    --validation_file path_to_valid.json \
    --output_dir path_to_output_dir \
    --text_column text \
    --summary_column summary \
    --num_train_epochs 5 \
    --num_beams 3
```

## Training LLaMA
```
python run_clm.py \
    --train_file path_to_train.json \
    --validation_file path_to_valid.json \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --model_name_or_path decapoda-research/llama-7b-hf \
    --learning_rate 1e-5 \
    --output_dir path_to_output_dir \
    --num_train_epochs 12 \
    --do_lora \
    --max_length 512 \
    --gradient_accumulation_steps 4 \
    --overwrite_cache
```

## Run GPT3
Please first prepare your OpenAI key and replace it in the code, then do
```
python run_gpt3.py
```
