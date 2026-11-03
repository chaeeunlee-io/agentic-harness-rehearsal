#!/bin/bash
set -u
cd /opt/appworld
python3 /tests/evaluate.py
echo "Verifier exit status: $?"
