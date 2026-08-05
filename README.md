# MFMFA-VI

## Requirements

* Git
* Anaconda or Miniconda

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/tjbucco/MFMFA-VI.git
   cd MFMFA-VI
   ```

2. Create the Conda environment:

   ```bash
   conda env create -f environment.yml
   ```

3. Activate the environment:

   ```bash
   conda activate mixtureenv
   ```

## Running the Program

Run the Python script in the mixtureenv environment with:

```bash
python mfmfa_v2.py
```

## Updating the Environment

If `environment.yml` is updated after you have already created the environment, run:

```bash
conda env update -f environment.yml --prune
```
