#!/bin/bash

models=(lstm att att_mh vq vq_mh)

# run `python -m src.model.{}`
for model in "${models[@]}"; do
    python -m src.model."$model"
done