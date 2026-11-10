#!/bin/sh
# Rebuild the 4 setup folders from the official sample agent.
# Usage: sh setups/make_setups.sh <path to the competition's sample_submission folder>
# (Kaggle: Gemma 4 Developer Agent competition, Data tab.)
set -e
SAMPLE=${1:?usage: make_setups.sh <sample_submission folder>}
cd "$(dirname "$0")"
for d in *.diff; do
    s=${d%.diff}
    rm -rf "$s" && mkdir "$s"
    (cd "$SAMPLE" && tar cf - agent.yaml eval_config.yaml configs prompts sub_agents) | (cd "$s" && tar xf -)
    patch -s -p1 -d "$s" < "$d"
    echo "built setups/$s"
done
