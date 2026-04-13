#!/bin/bash

# ---- CONFIG ----
SESSION_NAME="job_$(date +%Y%m%d_%H%M%S)"
LOG_DIR="logs"
SCRIPT="src/main.py"
CONDA_ENV="/venv/main"
CONDA_SH="/opt/miniforge3/etc/profile.d/conda.sh"
START_TIME="$(date '+%Y-%m-%d %H:%M:%S')"
ENABLE_EMAIL=true

# ---- ARGUMENTS PASSED TO THIS SCRIPT ----
ARGS="$@"

# ---- SETUP ----
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/$(date +%Y-%m-%d_%H-%M-%S).log"

# ---- START TMUX SESSION ----
tmux new-session -d -s "$SESSION_NAME" "

source $CONDA_SH
conda activate $CONDA_ENV

python $SCRIPT $ARGS 2>&1 | tee $LOG_FILE

status=\${PIPESTATUS[0]}
end_time=\$(date '+%Y-%m-%d %H:%M:%S')

if [ "$ENABLE_EMAIL" = true ]; then
python scripts/send_job_email.py \
  --session-name \"$SESSION_NAME\" \
  --args-text \"$ARGS\" \
  --exit-code \"\$status\" \
  --start-time \"$START_TIME\" \
  --end-time \"\$end_time\" \
  --log-file \"$LOG_FILE\"
  email_status=\$?
  if [ \$email_status -eq 0 ]; then
    printf 'Email notification: SENT\n' | tee -a $LOG_FILE
  else
    printf 'Email notification: FAILED (sender exit code: %s)\n' \"\$email_status\" | tee -a $LOG_FILE
  fi
fi

if [ \$status -eq 0 ]; then
  printf '\n===== JOB FINISHED =====\nStart time: %s\nEnd time: %s\nSession: %s\nArgs: %s\nExit code: %s\nStatus: SUCCESS\n' \
    \"$START_TIME\" \"\$end_time\" \"$SESSION_NAME\" \"$ARGS\" \"\$status\" | tee -a $LOG_FILE
else
  printf '\n===== JOB FINISHED =====\nStart time: %s\nEnd time: %s\nSession: %s\nArgs: %s\nExit code: %s\nStatus: FAILURE\n' \
    \"$START_TIME\" \"\$end_time\" \"$SESSION_NAME\" \"$ARGS\" \"\$status\" | tee -a $LOG_FILE
fi
exit \$status
"

# ---- OUTPUT INFO ----
echo "🚀 Job submitted!"
echo "🧠 tmux session: $SESSION_NAME"
echo "📄 log file: $LOG_FILE"
echo "⚙️ args: $ARGS"
echo "🔍 attach with: tmux attach -t $SESSION_NAME"