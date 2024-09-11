# README - Frontier Data Scientist Interview Task: Domestic Energy Usage

This repository contains the code and analysis output of the domestic energy usage in England and Wales. Follow the steps below to reproduce the analysis.

## Installation

1. **Ensure you have Anaconda (or Miniconda) installed**.
2. **Open a terminal**, navigate to your project repository, and run the following commands:
    ```sh
    conda env create -f environment.yml
    conda activate frontier-deu
    ```

## Usage

#### Run the analysis
```sh
python -m src.analysis
```

## Project Organization

```
├── README.md          <- The top-level README.
├── data
│   ├── processed      <- The final data sets for analysis.
│   └── raw            <- The original, immutable data dump.
│            
├── notebooks          <- Jupyter notebooks.
│
├── output             <- Generated analysis.
│   ├── eda            <- Exploratory data analysis.
│   └── analysis       <- Modeling exercise.
│
├── environment.yml    <- The requirements file for reproducing the analysis environment using conda.
│    
└── src                <- Source code for use in this project.
```

---