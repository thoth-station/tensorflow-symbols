#!/usr/bin/sh

pipenv install --dev

for tf_version in "2.2.0", "2.3.0", "2.4.0", "2.5.0", "2.6.0", "2.7.0", "2.8.0"
do
    pipenv run pip install "tensorflow===${tf_version}"
    pipenv run python3 app.py gather
done

pipenv run python3 app.py merge --no-patch > api.json
