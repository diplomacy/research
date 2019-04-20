#!/usr/bin/env bash

# Expand environment variables in requirements.txt
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd $DIR/..
sed -i 's@git+ssh://git@git+https://'"$GITHUB_TOKEN"':x-oauth-basic@' requirements.txt

# Uninstall diplomacy from cache
pip uninstall -qy diplomacy
exit 0
