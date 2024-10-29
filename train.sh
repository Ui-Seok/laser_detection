#!/bin/bash

# train_multi_stage.sh
set -e  # 에러 발생 시 스크립트 중단

# 환경 설정
MODEL_NAME="laser_detector_multi"
CHECKPOINT="yolov11s-face.pt"
DATA_YAML="datasets/data.yaml"

# 필요한 파일 체크
echo "Checking required files..."
for file in "transfer_train.py" "${CHECKPOINT}" "${DATA_YAML}"; do
    if [ ! -f "$file" ]; then
        echo "Error: Required file $file not found"
        exit 1
    fi
done

# 시작 시간 기록
START_TIME=$(date +"%Y-%m-%d %H:%M:%S")
echo "Training started at: ${START_TIME}"

# Stage 1: 초기 특징 추출층은 동결하고 학습
echo "Starting Stage 1 Training..."
python multi_stage_train.py \
    --checkpoint ${CHECKPOINT} \
    --epochs 150 \
    --img-size 640 \
    --batch 8 \
    --optimizer AdamW \
    --name "${MODEL_NAME}_stage1" \
    --stage 1 \
    --freeze 6

# Stage 1 완료 체크
STAGE1_WEIGHTS="runs/detect/${MODEL_NAME}_stage1/weights/best.pt"
if [ ! -f "${STAGE1_WEIGHTS}" ]; then
    echo "Error: Stage 1 weights not found at ${STAGE1_WEIGHTS}"
    exit 1
fi

echo "Stage 1 completed. Starting Stage 2..."

# Stage 2: 전체 네트워크 미세 조정
python multi_stage_train.py \
    --checkpoint ${STAGE1_WEIGHTS} \
    --epochs 100 \
    --img-size 640 \
    --batch 8 \
    --optimizer AdamW \
    --name "${MODEL_NAME}_stage2" \
    --stage 2 \
    --freeze 0

# 종료 시간 기록
END_TIME=$(date +"%Y-%m-%d %H:%M:%S")

# 최종 결과 확인
FINAL_WEIGHTS="runs/detect/${MODEL_NAME}_stage2/weights/best.pt"
if [ -f "${FINAL_WEIGHTS}" ]; then
    echo "Training successful!"
    echo "Final weights saved at: ${FINAL_WEIGHTS}"
else
    echo "Warning: Final weights file not found at expected location"
fi

# 학습 요약 출력
echo -e "\nTraining Summary:"
echo "Start time: ${START_TIME}"
echo "End time: ${END_TIME}"
echo "Model name: ${MODEL_NAME}"
echo "Initial checkpoint: ${CHECKPOINT}"
echo "Stage 1 weights: ${STAGE1_WEIGHTS}"
echo "Final weights: ${FINAL_WEIGHTS}"

# 학습 결과 복사 (선택사항)
RESULTS_DIR="training_results_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${RESULTS_DIR}"
cp -r "runs/train/${MODEL_NAME}_stage1" "${RESULTS_DIR}/"
cp -r "runs/train/${MODEL_NAME}_stage2" "${RESULTS_DIR}/"
echo "Results copied to: ${RESULTS_DIR}"