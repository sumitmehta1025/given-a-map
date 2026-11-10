#!/bin/sh
# Bundle what the Kaggle notebook (or the Google Cloud VM) needs into build/bundle/:
# the run script, both setups and the selected task IDs (26). Upload it as a private
# Kaggle dataset (-r zip keeps the setups/ folders; without it Kaggle skips folders):
#   first time:  .venv/bin/kaggle datasets create  -p build/bundle -r zip
#   updates:     .venv/bin/kaggle datasets version -p build/bundle -r zip -m "<what changed>"
set -e
cd "$(dirname "$0")/.."
rm -rf build/bundle && mkdir -p build/bundle
cp experiment/run_experiment.py analysis/output/selected_tasks.txt build/bundle/
cp -R setups build/bundle/setups
cp experiment/kaggle/dataset-metadata.json build/bundle/
# Earlier sessions of a run, so a new Kaggle session can continue it (--resume-from)
# Only the latest session of each run: it already holds every earlier attempt, and an older
# session's pending.json would be read as a second hang.
for run in $(ls -d runs/*-run*-session*/results/run* 2>/dev/null | sed 's#.*/##' | sort -u); do
    d=$(ls -d runs/*-run*-session*/results/$run | sort | tail -1)
    mkdir -p build/bundle/resume && cp -R "$d" build/bundle/resume/
    echo "resume/$run <- $d"
done
cp experiment/validate_tasks.py build/bundle/
echo "Commit: $(git rev-parse --short HEAD)$(git diff --quiet || echo ' (uncommitted changes!)')" > build/bundle/BUNDLE_VERSION.txt
cat build/bundle/BUNDLE_VERSION.txt
find build/bundle -type f | sort
