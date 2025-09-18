# Collect all task files
all_tasks=()
for task in "${tasks[@]}"; do
    all_tasks+=("src/tasks/${task}.py")
done

# Run all tasks for each model with parallelization
for model in "${models[@]}"; do
    case "${model}" in
        tiny) encoder="example/ced/tiny_ced.py" ;;
        mini) encoder="example/ced/mini_ced.py" ;;
        small) encoder="example/ced/small_ced.py" ;;
        base) encoder="example/ced/base_ced.py" ;;
    esac

    LOG_FILE="/data1/repos/EAT_projs/logfiles/xares_main_run/ced_${model}.log"
    OUTPUT_DIR="/data1/repos/EAT_projs/xares-main/outputs/ced_${model}"
    
    mkdir -p "${OUTPUT_DIR}"
    
    echo "==================== Running CED ${model} ====================" | tee -a "${LOG_FILE}"
    
    # This will now use --max-jobs 8 effectively
    python -m xares.run --max-jobs 8 --from-stage 1 --to-stage 2 \
        --output_dir "${OUTPUT_DIR}" "${encoder}" "${all_tasks[@]}" 2>&1 | tee -a "${LOG_FILE}"
done