#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
INSTALL_DIR="${CNN_INSTALL_DIR:-$ROOT_DIR/install}"
DB_FARM_PATH="${CNN_DB_FARM:-$ROOT_DIR/monetdb_farm}"
DB_NAME="${CNN_DB_NAME:-cnn_train_db}"
CMD_MCLIENT="${CNN_MCLIENT:-$INSTALL_DIR/bin/mclient}"
MODEL_ID="${CNN_MODEL_ID:-}"

fail() {
    printf '%s\n' "$*" >&2
    exit 1
}

FORMAT=sql
MODEL_COLUMNS='ModelId, Version, Status, Epoch, Step, "seed", InputShape, InputLayout, DType, ClassLabels'
PARAMETER_COLUMNS='ModelId, Version, LayerId, TensorName, TensorRole, Shape, Layout, DType, OCTET_LENGTH(TensorValue) AS TensorBytes'
OPTIMIZER_COLUMNS='ModelId, Version, LayerId, TensorName, StateName, Shape, DType, OCTET_LENGTH(TensorValue) AS TensorBytes'

if [[ $# == 1 && "$1" == --raw ]]; then
    FORMAT='csv+,'
    MODEL_COLUMNS='*'
    PARAMETER_COLUMNS='*'
    OPTIMIZER_COLUMNS='*'
elif [[ $# != 0 ]]; then
    fail 'Usage: showmodel.sh [--raw]'
fi

[[ "$DB_NAME" =~ ^[A-Za-z0-9_]+$ ]] || fail 'CNN_DB_NAME must contain only letters, digits or underscores.'
[[ -x "$CMD_MCLIENT" ]] || fail 'mclient is missing. Build and install MonetDB with bash build.sh.'

MODEL_FILTER=
if [[ -n "$MODEL_ID" ]]; then
    [[ "$MODEL_ID" =~ ^[A-Za-z0-9_-]{1,128}$ ]] || fail 'CNN_MODEL_ID must contain 1–128 letters, digits, underscores or hyphens.'
    MODEL_FILTER="WHERE ModelId = '$MODEL_ID'"
fi

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

# Share one transaction so all four queries see the same model versions.
"$CMD_MCLIENT" "${SQL_CONNECTION[@]}" -E UTF-8 -f "$FORMAT" <<SQL
START TRANSACTION;

SELECT $MODEL_COLUMNS
FROM cnn.Model
$MODEL_FILTER
ORDER BY ModelId, Version;

SELECT $PARAMETER_COLUMNS
FROM cnn.ModelTensor
$MODEL_FILTER
ORDER BY ModelId, Version, LayerId, TensorName;

SELECT $OPTIMIZER_COLUMNS
FROM cnn.OptimizerTensor
$MODEL_FILTER
ORDER BY ModelId, Version, LayerId, TensorName, StateName;

SELECT *
FROM cnn.TrainingMetric
$MODEL_FILTER
ORDER BY ModelId, Version, Epoch, Split;

COMMIT;
SQL
