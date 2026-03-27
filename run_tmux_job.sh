#!/bin/bash

# ---- CONFIG ----
SESSION_NAME="job_$(date +%Y%m%d_%H%M%S)"
LOG_DIR="logs"
SCRIPT="src/main.py"
CONDA_ENV="main"

# ---- ARGUMENTS PASSED TO THIS SCRIPT ----
ARGS="$@"

# ---- SETUP ----
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/$(date +%Y-%m-%d_%H-%M-%S).log"

# ---- START TMUX SESSION ----
tmux new-session -d -s "$SESSION_NAME" "

source ~/miniconda3/etc/profile.d/conda.sh
conda activate $CONDA_ENV

python $SCRIPT $ARGS 2>&1 | tee $LOG_FILE

"

# ---- OUTPUT INFO ----
echo "🚀 Job submitted!"
echo "🧠 tmux session: $SESSION_NAME"
echo "📄 log file: $LOG_FILE"
echo "⚙️ args: $ARGS"
echo "🔍 attach with: tmux attach -t $SESSION_NAME"