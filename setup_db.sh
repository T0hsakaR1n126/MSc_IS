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

"$CMD_MCLIENT" -d "$DB_NAME" -t performance <<EOF
DROP TABLE IF EXISTS model;
DROP TABLE IF EXISTS uw;
DROP TABLE IF EXISTS mw;

CREATE TABLE uw (U VARCHAR(10), F CLOB);
CREATE TABLE mw (M VARCHAR(10), G CLOB);

COPY INTO uw FROM '$ROOT_DIR/ml-latest/uw.csv' USING DELIMITERS ',', '\n', '"';
COPY INTO mw FROM '$ROOT_DIR/ml-latest/mw.csv' USING DELIMITERS ',', '\n', '"';

--Baseline
DROP TABLE IF EXISTS mo; 
SELECT M FROM uw, mw WHERE U='u2' ORDER BY dot(F,G) DESC LIMIT 10;

CREATE TABLE mo(R CLOB);
INSERT INTO mo SELECT pcatrain(F, CAST(64 AS INTEGER)) 
FROM (SELECT F FROM uw ORDER BY rand() LIMIT 900) AS subq;

--Filter: Top-50
DROP TABLE IF EXISTS candidates;
CREATE TABLE candidates AS SELECT M FROM uw, mw WHERE U='u2' ORDER BY dot(F,G) DESC LIMIT 50;

--Refine: Tap-10
DROP TABLE mo;
SELECT c.M FROM candidates c, uw, mw WHERE uw.U='u2' AND mw.M = c.M ORDER BY dot(uw.F, mw.G) DESC LIMIT 10;


EOF

echo "Done!"
echo "Please execute the following command to test your changes:"
echo "    export DOTMONETDBFILE=\"$AUTH_FILE\""
echo "    export PATH=\"$BIN_DIR:\$PATH\""
echo "    mclient -d $DB_NAME"
