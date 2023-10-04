#!/usr/bin/env bash

cmd="$1"

case "$cmd" in
    "train")
        python train.py
        ;;
    "train_hpo")
        python train.py --do-hpo
        ;;
    *)
        exec "$@"
        ;;
esac