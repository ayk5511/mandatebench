#!/usr/bin/env bash
# Build Paper 8. Regenerates every number, table and figure from the raw runs
# first, so the PDF can never show a number the data do not produce.
set -euo pipefail
VERSION="v0.1"
OUT_NAME="Khan_2026_agent_compliance_${VERSION}.pdf"
cd "$(dirname "$0")"
python3 ../scripts/analyze.py
tectonic main.tex
mkdir -p submission-ssrn
cp main.pdf "submission-ssrn/${OUT_NAME}"
echo "Built:           paper/main.pdf"
echo "Versioned copy:  paper/submission-ssrn/${OUT_NAME}"
