#!/bin/bash
# Syntax: ./run_tests               -- Run tests in parallel across CPUs
#         ./run_tests <nb_cores>    -- Run tests in parallel across this number of CPUs
#         ./run_tests 0             -- Only runs the pylint tests
export PYTHONIOENCODING=utf-8
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
FAILED=0

# Switching to correct directory and removing cache
cd $DIR
rm -f $HOME/.cache/diplomacy/*saved*.pb*

# Running pytest
if [ "${1:-auto}" != "0" ]; then
    pytest -v --forked -n "${1:-auto}" diplomacy_research || FAILED=1
fi

# Running pylint
find diplomacy_research -name "*.py" ! -name 'zzz_*.py' ! -name '_*.py' ! -name '*_pb2*.py' -exec pylint '{}' + && \
find . -path ./env -prune -o -name "*.py" ! -name 'zzz_*.py' ! -name '_*.py' ! -name '*_pb2*.py' -exec pylint -E -d E1129 -s y '{}' + || FAILED=1

# Exiting
if [[ "$FAILED" -eq 1 ]]; then
    echo "*** TESTS FAILED ***"
    exit 1
else
    echo "All tests passed."
    exit 0
fi
