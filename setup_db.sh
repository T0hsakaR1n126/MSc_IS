#!/bin/bash
set -e

ROOT_DIR=$(pwd)
INSTALL_DIR="$ROOT_DIR/install"
BIN_DIR="$INSTALL_DIR/bin"
LIB_DIR="$INSTALL_DIR/lib"
CMD_MONETDBD="$BIN_DIR/monetdbd"
CMD_MONETDB="$BIN_DIR/monetdb"
CMD_MCLIENT="$BIN_DIR/mclient"

DB_FARM_PATH="$ROOT_DIR/monetdb_farm"
DB_NAME="test_db"

AUTH_FILE="$ROOT_DIR/.monetdb_auth"

if [ ! -f "$CMD_MONETDBD" ]; then
    echo "Error: $CMD_MONETDBD not found, please run build.sh first..."
    exit 1
fi

export LD_LIBRARY_PATH="$LIB_DIR:$LD_LIBRARY_PATH"
export DYLD_LIBRARY_PATH="$LIB_DIR:$DYLD_LIBRARY_PATH"

cat > "$AUTH_FILE" <<EOF
user=monetdb
password=monetdb
language=sql
save_history=true
EOF

export DOTMONETDBFILE="$AUTH_FILE"

echo "Cleaning old daemons..."
if [ -d "$DB_FARM_PATH" ]; then
    "$CMD_MONETDBD" stop "$DB_FARM_PATH" || true
    sleep 1
fi

echo "Starting MonetDB daemon process..."
rm -rf "$DB_FARM_PATH"
if [ ! -d "$DB_FARM_PATH" ]; then
    "$CMD_MONETDBD" create "$DB_FARM_PATH" || { echo "failed to create farm"; exit 1; }
fi

"$CMD_MONETDBD" start "$DB_FARM_PATH"

echo "Initializing database..."

if "$CMD_MONETDB" status | grep -q "$DB_NAME"; then
    echo "   '$DB_NAME' exists"
else
    echo "   creating '$DB_NAME'..."
    "$CMD_MONETDB" create "$DB_NAME" || { echo "failed to create"; exit 1; }
fi

# "$CMD_MONETDB" set gdk_nr_threads=1 "$DB_NAME"
"$CMD_MONETDB" release "$DB_NAME"
sleep 2

echo "Initializing data..."

"$CMD_MCLIENT" -d "$DB_NAME" <<EOF
-- 1. 查看自动初始化的 CNN 存储表，预期四张。
SELECT s.name AS schema_name, t.name AS table_name
FROM sys.tables t JOIN sys.schemas s ON s.id = t.schema_id
WHERE s.name = 'cnn'
ORDER BY t.name;

-- 2. 查看过程的模块、实现语言和副作用标记；language_id = 1 表示 MAL。
-- sys.functions.func 对此类过程保存完整 SQL 定义，不是单独的 MAL 函数名。
SELECT s.name AS schema_name, f.name, f.mod, f.language AS language_id, f.side_effect
FROM sys.functions f JOIN sys.schemas s ON s.id = f.schema_id
WHERE s.name = 'sys' AND f.name = 'cnn_train' AND f.type = 2;

-- 3. 查看六个输入参数的顺序、名称和 SQL 类型。
SELECT a.number, a.name, a.type
FROM sys.args a JOIN sys.functions f ON f.id = a.func_id
JOIN sys.schemas s ON s.id = f.schema_id
WHERE s.name = 'sys' AND f.name = 'cnn_train' AND f.type = 2 AND a.inout = 1
ORDER BY a.number;

-- 4. 当前只定义了存储结构，初始行数应为零。
SELECT 'Model' AS table_name, COUNT(*) AS row_count FROM cnn.Model
UNION ALL SELECT 'ModelTensor', COUNT(*) FROM cnn.ModelTensor
UNION ALL SELECT 'OptimizerTensor', COUNT(*) FROM cnn.OptimizerTensor
UNION ALL SELECT 'TrainingMetric', COUNT(*) FROM cnn.TrainingMetric
ORDER BY 1;
EOF

# 自动核对四张表、MAL 过程元数据及六个预期输入参数是否存在。
# 后面的 CALL 及其精确错误消息进一步验证是否进入了预期的 C 实现。
cnn_metadata="$("$CMD_MCLIENT" -d "$DB_NAME" -f csv-noquote -s "
SELECT
    (SELECT COUNT(*) FROM sys.tables t JOIN sys.schemas s ON s.id = t.schema_id
     WHERE s.name = 'cnn' AND t.type = 0
       AND t.name IN ('model', 'modeltensor', 'optimizertensor', 'trainingmetric')),
    (SELECT COUNT(*) FROM sys.functions f JOIN sys.schemas s ON s.id = f.schema_id
     WHERE s.name = 'sys' AND f.name = 'cnn_train' AND f.type = 2
       AND f.mod = 'sql' AND f.language = 1 AND f.side_effect = true),
    (SELECT COUNT(*) FROM sys.args a JOIN sys.functions f ON f.id = a.func_id
     JOIN sys.schemas s ON s.id = f.schema_id
     WHERE s.name = 'sys' AND f.name = 'cnn_train' AND f.type = 2 AND a.inout = 1
       AND a.name IN ('model_id', 'data_query', 'epochs', 'batch_size', 'optimizer', 'learning_rate'));")"
if [[ "$cnn_metadata" != '4,1,6' ]]; then
    echo "failed: need 4,1,6, but is $cnn_metadata" >&2
    exit 1
fi
echo "passed: four tables, MAL procedure metadata and six input parameters are registered."

# 每条预期报错的 CALL 单独执行，在 if 中捕获，避免 set -e 提前结束。
# csv-noquote 格式会保留错误输出中的 CODE = SQLSTATE。
expect_cnn_error() {
    local state="$1" message="$2" query="$3"
    local output line state_seen=false
    if output="$("$CMD_MCLIENT" -d "$DB_NAME" -f csv-noquote -s "$query" 2>&1)"; then
        echo "failed: call succeeded unexpectedly, expected SQLSTATE $state." >&2
        exit 1
    fi
    while IFS= read -r line; do
        if [[ "$line" =~ ^CODE[[:space:]]*=[[:space:]]*${state}[[:space:]]*$ ]]; then
            state_seen=true
        fi
    done <<< "$output"
    if [[ "$state_seen" != true || "$output" != *"$message"* ]]; then
        printf 'failed: SQLSTATE %s, %s\n' "$state" "$message" >&2
        exit 1
    fi
    printf 'passed: SQLSTATE %s, %s\n' "$state" "$message"
}

expect_cnn_error '22023' 'epochs must be a positive integer' \
    "CALL sys.cnn_train('cnn_smoke', 'SELECT 1', 0, 2, 'sgd', CAST(0.01 AS DOUBLE));"
expect_cnn_error '0A000' 'CNN training is not implemented yet' \
    "CALL cnn_train('cnn_smoke', 'SELECT 1', 1, 2, 'sgd', CAST(0.01 AS DOUBLE));"
expect_cnn_error '0A000' 'CNN training is not implemented yet' \
    "CALL sys.cnn_train('cnn_smoke', 'SELECT 1', 1, 2, 'adam', CAST(0.01 AS DOUBLE));"

# 原前段创建全新的数据库，当前骨架验收后四张表应仍然为空。
cnn_counts="$("$CMD_MCLIENT" -d "$DB_NAME" -f csv-noquote -s "
SELECT COUNT(*) FROM cnn.Model
UNION ALL SELECT COUNT(*) FROM cnn.ModelTensor
UNION ALL SELECT COUNT(*) FROM cnn.OptimizerTensor
UNION ALL SELECT COUNT(*) FROM cnn.TrainingMetric;")"
if [[ "$cnn_counts" != $'0\n0\n0\n0' ]]; then
    printf 'failed: expected all tables to be empty, but got counts:\n%s\n' "$cnn_counts" >&2
    exit 1
fi
echo "passed: all four tables are empty after test calls."

echo "Done!"
echo "Please execute the following command to test your changes:"
echo "    export DOTMONETDBFILE=\"$AUTH_FILE\""
echo "    export PATH=\"$BIN_DIR:\$PATH\""
echo "    mclient -d $DB_NAME"
