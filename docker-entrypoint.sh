#!/usr/bin/env bash

cmd="$1"

case "$cmd" in
    "train")
        python train.py
        ;;
    "train_hpo")
        python train.py --do-hpo
        ;;
    "test_deployment")
         pytest -vv -m deployment
        ;;
    "serve_dev")
        uvicorn corelib.entrypoints.api:get_app() --host 0.0.0.0 --port 8000 --reload
        ;;
    *)
        exec "$@"
        ;;
esac