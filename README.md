# Project: Integrating Vector Similarity Join Computations into MonetDB

## Authors

Ziqi Yang

Zheyuan Fu

## Build

Please run `build.sh` to build the project and `setup_db.sh` to establish a test database farm.

```bash
./build.sh
./setup_db.sh
```

Then a command for custom SQL queries in `mclient` will be provided:

```bash
export DOTMONETDBFILE="/Users/yzq/projects/monetdb_similarity_join/.monetdb_auth"
    export PATH="/Users/yzq/projects/monetdb_similarity_join/install/bin:$PATH"
    mclient -d test_db
```

## Dataset

We use MovieLens as a test dataset: https://grouplens.org/datasets/movielens/latest/
