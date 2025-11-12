#!/bin/bash
# pope_eval_batch.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_SCRIPT="${SCRIPT_DIR}/pope_eval.sh"

# 설정
SEEDS=(4 6 8 10)  # llava의 경우
TYPES=("random" "popular" "adversarial")
MODEL="llava"

RESULTS_DIR="./pope_results_summary"
mkdir -p ${RESULTS_DIR}
SUMMARY_FILE="${RESULTS_DIR}/summary_$(date +%Y%m%d_%H%M%S).txt"

echo "Starting batch evaluation..."
echo "Seeds: ${SEEDS[@]}"
echo "Types: ${TYPES[@]}"
echo ""

# 각 조합 실행
for seed in "${SEEDS[@]}"; do
    for type in "${TYPES[@]}"; do
        echo "Running: seed=${seed}, type=${type}"
        
        # 임시 스크립트 생성
        TEMP_SCRIPT="${RESULTS_DIR}/pope_eval_${seed}_${type}.sh"
        cp ${BASE_SCRIPT} ${TEMP_SCRIPT}
        
        # seed와 type 수정
        sed -i "s/^seed=.*/seed=${seed}/" ${TEMP_SCRIPT}
        sed -i "s/^type=.*/type=\"${type}\"/" ${TEMP_SCRIPT}
        
        # 실행
        bash ${TEMP_SCRIPT} 2>&1 | tee ${RESULTS_DIR}/log_${seed}_${type}.txt
        
        # 정리
        rm -f ${TEMP_SCRIPT}
        echo ""
    done
done

# 결과 수집
echo "==========================================" > ${SUMMARY_FILE}
echo "POPE Results Summary" >> ${SUMMARY_FILE}
echo "==========================================" >> ${SUMMARY_FILE}
printf "%-12s %-10s %-8s %-10s %-10s %-10s %-10s\n" \
    "Type" "Seed" "Acc" "Precision" "Recall" "F1" "Yes_Ratio" >> ${SUMMARY_FILE}
echo "------------------------------------------------------------" >> ${SUMMARY_FILE}

for seed in "${SEEDS[@]}"; do
    for type in "${TYPES[@]}"; do
        LOG_FILE="${RESULTS_DIR}/log_${seed}_${type}.txt"
        if [ -f "${LOG_FILE}" ]; then
            RESULT=$(grep -E "acc:.*precision:.*recall:.*f1:" ${LOG_FILE} | tail -1)
            if [ -n "${RESULT}" ]; then
                ACC=$(echo ${RESULT} | grep -oP 'acc: \K[0-9.]+')
                PREC=$(echo ${RESULT} | grep -oP 'precision: \K[0-9.]+')
                REC=$(echo ${RESULT} | grep -oP 'recall: \K[0-9.]+')
                F1=$(echo ${RESULT} | grep -oP 'f1: \K[0-9.]+')
                YES=$(echo ${RESULT} | grep -oP 'yes_ratio: \K[0-9.]+')
                printf "%-12s %-10s %-8s %-10s %-10s %-10s %-10s\n" \
                    "${type}" "${seed}" "${ACC}" "${PREC}" "${REC}" "${F1}" "${YES}" >> ${SUMMARY_FILE}
            fi
        fi
    done
done

cat ${SUMMARY_FILE}
echo ""
echo "Summary saved to: ${SUMMARY_FILE}"