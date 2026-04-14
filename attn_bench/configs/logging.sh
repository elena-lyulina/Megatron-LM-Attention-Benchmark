#!/usr/bin/env bash
# Logging configuration.
# Requires: EXP_DIR, DEBUG_DIR, REPO_ROOT, PROJECT_NAME, EXP_NAME set before sourcing.

### LOGGING ###
: ${LOGGING_DIR:=$EXP_DIR/logging}
: ${TENSORBOARD_DIR:=$LOGGING_DIR/tensorboard}

: ${GPU_MEM_LOGGING_FILE:=$DEBUG_DIR/memory_logging.txt}

mkdir -p $LOGGING_DIR $TENSORBOARD_DIR
srun -l bash -c 'echo $(hostname) $(nvidia-smi | grep -o "|\s*[0-9]*MiB")' > $GPU_MEM_LOGGING_FILE

LOGGING_ARGS=(
    --log-throughput
    --log-progress
    --log-interval 1
    --tensorboard-dir $TENSORBOARD_DIR
    --log-timers-to-tensorboard
    --no-log-loss-scale-to-tensorboard
    --log-memory-to-tensorboard
)

### LOG W&B ###

# uses / modifies $TRAINING_CMD, so run as a function after it is set
setup_wandb() {
    LOG_WANDB="${LOG_WANDB:-true}"
    if [ "${LOG_WANDB}" = "true" ]; then
        echo "[$(date)] W&B logging enabled."
        if [ -d "$LOGGING_DIR/wandb/latest-run" ]; then
            wandb sync "$LOGGING_DIR/wandb/latest-run"
        fi
        TRAINING_CMD="$TRAINING_CMD \
            --wandb-save-dir $LOGGING_DIR \
            --wandb-project $PROJECT_NAME \
            --wandb-exp-name $EXP_NAME-$SLURM_JOB_ID"
    else
        export WANDB_MODE=disabled
        echo "[$(date)] W&B logging disabled."
    fi
}

### LOG COMPUTE ENVIRONMENT ###
: ${COMPUTE_ENVIRONMENT_FILE:=$DEBUG_DIR/compute_environment.txt}

# uses / modifies $TRAINING_CMD, so run as a function after it is set
log_compute_env() {
    SEP=$(printf '=%.0s' {1..100})
    {
        echo -e "$(date)"
        echo -e "${SEP}"
        echo -e "\nCMD: ${CMD_PREFIX} ${TRAINING_CMD}"
        echo -e "${SEP}"
        echo -e "\nSlurm file: $0\n"
        cat "$0"
        echo -e "${SEP}"
        echo -e "\nTOML file: ${SLURM_SPANK__SLURM_SPANK_OPTION_pyxis_environment}\n"
        cat "${SLURM_SPANK__SLURM_SPANK_OPTION_pyxis_environment}"
        echo -e "${SEP}"
        echo -e "\nNODES: $(scontrol show hostnames ${SLURM_JOB_NODELIST})"
        echo -e "${SEP}"
        echo -e "\nRepo: ${REPO_ROOT} ($(git -C ${REPO_ROOT} rev-parse --verify HEAD))"
        echo -e "${SEP}"
        echo -e "\n$(pip list)"
        echo -e "${SEP}"
        echo -e "\nPython sys.path:\n$(python -c 'import sys; print(chr(10).join(sys.path))')"
        echo -e "${SEP}"
        echo -e "\nmegatron location: $(python -c 'import megatron; print(megatron.__file__)' 2>&1)"
        echo -e "${SEP}"
        echo -e "\nmegatron.__path__: $(python -c 'import megatron; print(list(megatron.__path__))' 2>&1)"
        echo -e "${SEP}"
        echo -e "\nmegatron/training exists: $(ls ${REPO_ROOT}/megatron/training 2>&1 | head -5)"
        echo -e "${SEP}"
        echo -e "\n$(nvidia-smi)"
        echo -e "${SEP}"
        echo -e "\nEnvironment Variables:\n\n$(printenv)"
        echo -e "${SEP}"
    } > "${COMPUTE_ENVIRONMENT_FILE}"
}