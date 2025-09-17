#!/usr/local/bin/zsh
# there are 39 tasks.
set -e
set -o pipefail

source /data1/miniforge3/etc/profile.d/conda.sh
source /data1/root/.zsh_utils/dl_utils/dl_utils.zsh

conda activate py310


cd /data1/repos/EAT_projs/xares-main

export EAT_FRAMEWORK="fairseq"
export EAT_MODE="pretrain"

#---------kkuhn-block------------------------------
tasks=(
    "2023_Gree_Motor_task"
    "2023_Steering_Column_task"
    "2023_Xinjie_Pump_task"
    "2024_Shanghai_Yanfeng_Slider_task"
    "2025_Changshu_Foya_task"
    "2025_Dongguan_Anwen_Fan_task"
    "2025_Endi_Fan_task"
    "2025_Kuka_Robot_task"
    "2025_Mianyang_Fulin_Vibration_task"
    "2025_Wuxi_Shengwei_Dehua_Auto_Parts_task"
    "changshu_motor2023"
)

models=("tiny" "mini" "small" "base")
COMMENT=""
#---------kkuhn-block------------------------------

for model in "${models[@]}"; do
    case "${model}" in
        tiny) encoder="example/ced/tiny_ced.py" ;;
        mini) encoder="example/ced/mini_ced.py" ;;
        small) encoder="example/ced/small_ced.py" ;;
        base) encoder="example/ced/base_ced.py" ;;
        *) echo "Unknown model: ${model}" ; exit 1 ;;
    esac

    LOG_FILE="/data1/repos/EAT_projs/logfiles/xares_main_run/ced_${model}.log"
    OUTPUT_DIR="/data1/repos/EAT_projs/xares-main/outputs/ced_${model}"

    mkdir -p "${OUTPUT_DIR}"

    add_log_header COMMENT >> "${LOG_FILE}"
    echo "==================== Running CED ${model} ====================" | tee -a "${LOG_FILE}"

    for task in "${tasks[@]}"; do
        echo "Running task: ${task} (model: ${model})" | tee -a "${LOG_FILE}"
        python -m xares.run --from-stage 1 --to-stage 2 --output_dir "${OUTPUT_DIR}" "${encoder}" \
            src/tasks/${task}.py 2>&1 | tee -a "${LOG_FILE}"
    done
done