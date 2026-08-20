# NIRWALS Pipeline

Science data reduction pipeline for reducing NIRWALS data from SALT.

## Structure

- `salt_pkgs/` - Science data reduction pipeline code from SALT
- `wrappers/` - Wrapper scripts for running the pipeline
- `YYYYMMDD/` - Data directories (not tracked in git)

## Setup

Clone this repository, then create and activate the conda environment with all dependencies:

```bash
git clone https://github.com/Marissa-Perry/nirwals-pipeline.git
cd nirwals-pipeline
conda env create -f nirwals_drp_env.yml
conda activate nirwals_drp_env
```

## Usage
```bash
python -m nirwals_pipeline.wrappers.run_nirwals_workflow YYYYMMDD
```
where YYYYMMDD is the date of the observation to be reduced.
