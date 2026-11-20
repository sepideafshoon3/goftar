#!/bin/sh
# Commands that load a model: download it on the first start (the only time network is
# needed), then run fully offline. Any other command (version, doctor, models ...) runs as is.
set -e
case "$1" in
  ui|serve|transcribe)
    model="${GOFTAR_MODEL:-large-v3-turbo}"
    dir="/data/models/$(echo "$model" | sed 's#/#--#g')"
    if [ ! -f "$dir/model.bin" ]; then
      echo "goftar: model $model is not in /data/models yet; downloading it once..."
      goftar models download "$model"
    fi
    exec goftar "$@" --model "$model" --offline
    ;;
  *)
    exec goftar "$@"
    ;;
esac
