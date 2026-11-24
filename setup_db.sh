#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
INSTALL_DIR="${CNN_INSTALL_DIR:-$ROOT_DIR/install}"
DATASET_DIR="${CNN_DATASET_DIR:-$ROOT_DIR/datasets/oxford-pets-64}"
DB_FARM_PATH="${CNN_DB_FARM:-$ROOT_DIR/monetdb_farm}"
DB_NAME="${CNN_DB_NAME:-cnn_train_db}"
CMD_MCLIENT="${CNN_MCLIENT:-$INSTALL_DIR/bin/mclient}"
MODEL_ID="${CNN_MODEL_ID:-cat_dog_oxford_cnn}"
OPTIMIZER="${CNN_OPTIMIZER:-adam}"
EPOCHS="${CNN_EPOCHS:-20}"
BATCH_SIZE="${CNN_BATCH_SIZE:-64}"
LEARNING_RATE="${CNN_LEARNING_RATE:-0.001}"
SEED="${CNN_SEED:-42}"

fail() {
    printf '%s\n' "$*" >&2
    exit 1
}

[[ -x "$CMD_MCLIENT" ]] || fail 'mclient is missing. Build and install the current MonetDB source with bash build.sh.'
command -v python3 >/dev/null || fail 'Python 3 is required to load the prepared images.'
[[ -r "$DATASET_DIR/manifest.csv" && -d "$DATASET_DIR/pixels" ]] ||
    fail "Prepared images are missing in $DATASET_DIR. Run scripts/prepare_oxford_pets.py first."
[[ "$DB_NAME" =~ ^[A-Za-z0-9_]+$ ]] || fail 'CNN_DB_NAME must contain only letters, digits or underscores.'
[[ "$MODEL_ID" =~ ^[A-Za-z0-9_-]{1,128}$ ]] || fail 'CNN_MODEL_ID must contain 1–128 letters, digits, underscores or hyphens.'
[[ "$OPTIMIZER" == adam || "$OPTIMIZER" == sgd ]] || fail 'CNN_OPTIMIZER must be adam or sgd.'
[[ "$EPOCHS" =~ ^[1-9][0-9]{0,9}$ && "$BATCH_SIZE" =~ ^[1-9][0-9]{0,9}$ ]] ||
    fail 'CNN_EPOCHS and CNN_BATCH_SIZE must be positive integers.'
[[ "$SEED" =~ ^(0|[1-9][0-9]{0,18})$ ]] || fail 'CNN_SEED must be a non-negative BIGINT.'
[[ "$LEARNING_RATE" =~ ^([0-9]+(\.[0-9]*)?|\.[0-9]+)([eE][+-]?[0-9]+)?$ ]] ||
    fail 'CNN_LEARNING_RATE must be a positive number.'
python3 - "$EPOCHS" "$BATCH_SIZE" "$SEED" "$LEARNING_RATE" <<'PY'
import math
import sys

epochs, batch_size, seed = map(int, sys.argv[1:4])
rate = float(sys.argv[4])
if epochs > 2**31 - 2:
    sys.exit('CNN_EPOCHS exceeds the model version range.')
if batch_size > 2**31 - 1:
    sys.exit('CNN_BATCH_SIZE exceeds the SQL INTEGER range.')
if seed > 2**63 - 1:
    sys.exit('CNN_SEED exceeds the SQL BIGINT range.')
if not (math.isfinite(rate) and rate > 0):
    sys.exit('CNN_LEARNING_RATE must be finite and positive.')
PY

export DOTMONETDBFILE="${DOTMONETDBFILE:-$ROOT_DIR/.monetdb_auth}"
if [[ ! -e "$DOTMONETDBFILE" ]]; then
    (umask 077; printf 'user=monetdb\npassword=monetdb\nlanguage=sql\n' > "$DOTMONETDBFILE")
fi

if [[ -n "${CNN_MAPI_URI:-}" ]]; then
    # Connect to an already-running database, for example an isolated test server.
    SQL_CONNECTION=(-d "$CNN_MAPI_URI")
else
    CMD_MONETDBD="$INSTALL_DIR/bin/monetdbd"
    CMD_MONETDB="$INSTALL_DIR/bin/monetdb"
    [[ -x "$CMD_MONETDBD" && -x "$CMD_MONETDB" ]] || fail 'MonetDB daemon tools are missing. Run bash build.sh.'
    export LD_LIBRARY_PATH="$INSTALL_DIR/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    export DYLD_LIBRARY_PATH="$INSTALL_DIR/lib${DYLD_LIBRARY_PATH:+:$DYLD_LIBRARY_PATH}"

    if [[ ! -d "$DB_FARM_PATH" ]]; then
        "$CMD_MONETDBD" create "$DB_FARM_PATH"
    fi
    farm_info="$("$CMD_MONETDBD" get status,port,sockdir "$DB_FARM_PATH")"
    if [[ "$farm_info" == *'no monetdbd is serving this dbfarm'* ]]; then
        "$CMD_MONETDBD" start "$DB_FARM_PATH"
    fi
    port="$(awk '$1 == "port" {print $2}' <<< "$farm_info")"
    sockdir="$(awk '$1 == "sockdir" {sub(/^[[:space:]]*sockdir[[:space:]]+/, ""); print}' <<< "$farm_info")"
    [[ "$port" =~ ^[0-9]+$ && "$sockdir" == /* ]] || fail 'Could not determine the MonetDB farm socket.'
    CONTROL_CONNECTION=(-h "$sockdir" -p "$port")
    if [[ ! -d "$DB_FARM_PATH/$DB_NAME" ]]; then
        "$CMD_MONETDB" "${CONTROL_CONNECTION[@]}" create "$DB_NAME"
        "$CMD_MONETDB" "${CONTROL_CONNECTION[@]}" release "$DB_NAME"
    fi
    SQL_CONNECTION=(-h "$sockdir" -p "$port" -d "$DB_NAME")
fi

cnn_sql() {
    "$CMD_MCLIENT" "${SQL_CONNECTION[@]}" -E UTF-8 "$@"
}

cnn_tables="$(cnn_sql -f csv-noquote -s "
SELECT COUNT(*) FROM sys.tables t JOIN sys.schemas s ON s.id = t.schema_id
WHERE s.name = 'cnn' AND t.type = 0 AND t.name IN (
    'model', 'modeltensor', 'optimizertensor', 'trainingmetric', 'trainingimage', 'imagechannel'
);")"
[[ "$cnn_tables" == 6 ]] || fail 'This database does not have the CNN schema. Install the current build and initialize a new training database.'

# The previous build has the same six tables. Probe the query validation without
# creating model state or importing data, so using a stale server fails early.
if diagnostic="$(cnn_sql -f csv-noquote -s "CALL sys.cnn_train('cnn_script_preflight', 'SELECT 1', 1, 1, 'sgd', CAST(0.01 AS DOUBLE));" 2>&1)"; then
    fail 'Unexpected result from CNN preflight. Install the current MonetDB build before training.'
fi
[[ "$diagnostic" == *'Supported data_query'* ]] ||
    fail "The server did not recognize the current CNN training entry. Install the current build before training.
$diagnostic"

model_exists="$(cnn_sql -f csv-noquote -s "SELECT COUNT(*) FROM cnn.Model WHERE ModelId = '$MODEL_ID';")"
[[ "$model_exists" == 0 ]] || fail "Model '$MODEL_ID' already exists. Choose a new CNN_MODEL_ID to start a fresh training run."

# COPY, writes and the full comparison all share one transaction.
python3 "$ROOT_DIR/scripts/import_cnn_images.py" "$DATASET_DIR" | cnn_sql

printf 'Training %s with %s for %s epoch(s), batch size %s, learning rate %s, seed %s.\n' \
    "$MODEL_ID" "$OPTIMIZER" "$EPOCHS" "$BATCH_SIZE" "$LEARNING_RATE" "$SEED"
cnn_sql <<SQL
START TRANSACTION;
CALL sys.cnn_create('$MODEL_ID', '$OPTIMIZER', CAST($SEED AS BIGINT));
CALL sys.cnn_train(
    '$MODEL_ID',
    'SELECT ImageId FROM cnn.TrainingImage WHERE Split = ''train'' ORDER BY ImageId',
    CAST($EPOCHS AS INTEGER), CAST($BATCH_SIZE AS INTEGER),
    '$OPTIMIZER', CAST($LEARNING_RATE AS DOUBLE)
);
COMMIT;

SELECT ModelId, Version, Status, Epoch, Step
FROM cnn.Model
WHERE ModelId = '$MODEL_ID'
ORDER BY Version DESC LIMIT 1;

SELECT Epoch, Split, Loss, Accuracy, F1, Runtime AS Seconds
FROM cnn.TrainingMetric
WHERE ModelId = '$MODEL_ID'
ORDER BY Epoch, CASE Split WHEN 'train' THEN 1 WHEN 'validation' THEN 2 ELSE 3 END;
SQL
