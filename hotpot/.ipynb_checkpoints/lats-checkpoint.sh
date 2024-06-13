python run.py \
    --backend gpt-3.5-turbo \
    --task_start_index 0 \
    --task_end_index 100 \
    --n_generate_sample 1 \
    --n_evaluate_sample 1 \
    --prompt_sample cot \
    --temperature 1.0 \
    --iterations 2 \
    --log logs/tot_10k.log \
    --algorithm lats
    ${@}
