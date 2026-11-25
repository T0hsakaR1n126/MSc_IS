#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
INSTALL_DIR="${CNN_INSTALL_DIR:-$ROOT_DIR/install}"
DB_FARM_PATH="${CNN_DB_FARM:-$ROOT_DIR/monetdb_farm}"
DB_NAME="${CNN_DB_NAME:-cnn_train_db}"
CMD_MCLIENT="${CNN_MCLIENT:-$INSTALL_DIR/bin/mclient}"
MODEL_ID="${CNN_MODEL_ID:-cat_dog_oxford_cnn}"
IMAGE_SPLIT="${CNN_SPLIT:-test}"
IMAGE_LIMIT="${CNN_LIMIT:-20}"
PREDICT_SQL="$ROOT_DIR/MonetDB/sql/scripts/90_cnn_predict.sql"

fail() {
    printf '%s\n' "$*" >&2
    exit 1
}

[[ "$DB_NAME" =~ ^[A-Za-z0-9_]+$ ]] || fail 'CNN_DB_NAME must contain only letters, digits or underscores.'
[[ "$MODEL_ID" =~ ^[A-Za-z0-9_-]{1,128}$ ]] || fail 'CNN_MODEL_ID must contain 1–128 letters, digits, underscores or hyphens.'
case "$IMAGE_SPLIT" in
    train|validation|test) ;;
    *) fail 'CNN_SPLIT must be train, validation or test.' ;;
esac
[[ "$IMAGE_LIMIT" =~ ^[1-9][0-9]{0,9}$ ]] && (( IMAGE_LIMIT <= 2147483647 )) ||
    fail 'CNN_LIMIT must be an integer from 1 to 2147483647.'
[[ -x "$CMD_MCLIENT" ]] || fail 'mclient is missing. Build and install MonetDB with bash build.sh.'

export DOTMONETDBFILE="${DOTMONETDBFILE:-$ROOT_DIR/.monetdb_auth}"
export LD_LIBRARY_PATH="$INSTALL_DIR/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export DYLD_LIBRARY_PATH="$INSTALL_DIR/lib${DYLD_LIBRARY_PATH:+:$DYLD_LIBRARY_PATH}"

if [[ -n "${CNN_MAPI_URI:-}" ]]; then
    SQL_CONNECTION=(-d "$CNN_MAPI_URI")
else
    CMD_MONETDBD="$INSTALL_DIR/bin/monetdbd"
    [[ -x "$CMD_MONETDBD" ]] || fail 'monetdbd is missing. Build and install MonetDB with bash build.sh.'
    [[ -d "$DB_FARM_PATH/$DB_NAME" ]] || fail "Database '$DB_NAME' does not exist in $DB_FARM_PATH."
    farm_info="$("$CMD_MONETDBD" get status,port,sockdir "$DB_FARM_PATH")"
    port="$(awk '$1 == "port" {print $2}' <<< "$farm_info")"
    sockdir="$(awk '$1 == "sockdir" {sub(/^[[:space:]]*sockdir[[:space:]]+/, ""); print}' <<< "$farm_info")"
    [[ "$port" =~ ^[0-9]+$ && "$sockdir" == /* ]] || fail 'Could not determine the MonetDB farm socket.'
    if [[ "$farm_info" == *'no monetdbd is serving this dbfarm'* ]]; then
        "$CMD_MONETDBD" start "$DB_FARM_PATH"
    fi
    SQL_CONNECTION=(-h "$sockdir" -p "$port" -d "$DB_NAME")
fi

cnn_sql() {
    "$CMD_MCLIENT" "${SQL_CONNECTION[@]}" -E UTF-8 "$@"
}

predict_registered() {
    cnn_sql -f csv-noquote -s "
SELECT COUNT(*)
FROM sys.functions f JOIN sys.schemas s ON s.id = f.schema_id
WHERE s.name = 'sys' AND f.name = 'cnn_predict' AND f.type = 5
  AND f.language = 1 AND f.mod = 'sql';"
}

report_prediction_error() {
    if [[ "$1" == *'external name sql.cnn_predict not bound'* ]]; then
        fail "The connected server has no native CNN predictor. Build and install the current MonetDB source with bash build.sh on that server, then restart its MonetDB service.
$1"
    fi
    fail "$1"
}

trained_version="$(cnn_sql -f csv-noquote -s "
SELECT COALESCE(MAX(Version), 0) FROM cnn.Model
WHERE ModelId = '$MODEL_ID' AND Status = 'trained' AND Epoch > 0;")"
[[ "$trained_version" != 0 ]] || fail "No trained version was found for model '$MODEL_ID'."

registered="$(predict_registered)"
if [[ "$registered" == 0 ]]; then
    [[ -r "$PREDICT_SQL" ]] || fail "CNN prediction SQL is missing at $PREDICT_SQL."
    if ! diagnostic="$(cnn_sql -f csv-noquote < "$PREDICT_SQL" 2>&1)"; then
        # Another invocation may have registered the function since the check.
        if ! registered="$(predict_registered)" || [[ "$registered" == 0 ]]; then
            report_prediction_error "$diagnostic"
        fi
    fi
fi

if ! prediction="$(cnn_sql 2>&1 <<SQL
SELECT m.ModelId, p.ModelVersion, m.Epoch, m.Step, p.ImageId,
       CASE p.Label WHEN 0 THEN 'cat' WHEN 1 THEN 'dog' END AS ActualClass,
       CASE p.PredictedLabel WHEN 0 THEN 'cat' WHEN 1 THEN 'dog' END AS PredictedClass,
       p.CatProbability, p.DogProbability
FROM sys.cnn_predict('$MODEL_ID', '$IMAGE_SPLIT', CAST($IMAGE_LIMIT AS INTEGER)) AS p
JOIN cnn.Model AS m ON m.ModelId = '$MODEL_ID' AND m.Version = p.ModelVersion
ORDER BY p.ImageId;
SQL
)"; then
    report_prediction_error "$prediction"
fi
printf '%s\n' "$prediction"
