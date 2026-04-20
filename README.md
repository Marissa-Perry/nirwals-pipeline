# NIRWALS Pipeline

Science data reduction pipeline for reducing NIRWALS data from SALT.

## Structure

- `salt_pkgs/` - Science data reduction pipeline code from SALT
- `wrappers/` - Wrapper scripts for running the pipeline
- `YYYYMMDD/` - Data directories (not tracked in git)

## Setup

[eventually add a list of dependencies in .txt file]

## Usage
```bash
python -m nirwals_pipeline.wrappers.run_nirwals_workflow nirwals_pipeline/salt_pkgs/workflows/nirwals/workflow_science_nirwals.json YYYYMMDD nirwals_pipeline/salt_pkgs/params nirwals_pipeline/salt_pkgs/configs nirwals_pipeline/YYYYMMDD
```
