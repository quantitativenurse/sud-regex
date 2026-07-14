#!/bin/bash
# Environment for running sudregex's Spark tests/integration on ACCRE.
# Pinned versions: arrow/24.0.0 (Python 3.13 build), java/17.0.6 (min required by pyspark 4.x).
# If ACCRE bumps default arrow/java versions, this may need updating — check with
# `module avail arrow` / `module avail java` and adjust the paths below.
module load gcc arrow/24.0.0 java/17.0.6
export PYTHONPATH="/cvmfs/soft.computecanada.ca/easybuild/software/2023/x86-64-v3/Compiler/gcccore/arrow/24.0.0/lib/python3.13/site-packages:$PYTHONPATH"
source .venv313/bin/activate
