# Are DAOs dead? An Empirical Model for Determining the Activity of a Decentralized Autonomous Organization

This repo contains all source code used to conduct the empirical experiment for my bachelors thesis.

#### Abstract

> The advent of smart-contract-enabled blockchains has paved the way for a new form of decentralized organization that can operate under autonomous and predictable rules encoded in transparent smart contracts. Since these Decentralized Autonomous Organizations (DAOs) are inexpensive to create and can effectively exist forever on the blockchain, it is vital to determine what distinguishes an “active” DAO from an “inactive” one. To address this, we developed a simplified theoretical framework grounded in financial, social, and development indicators. We then examined the 12 largest public blockchains to construct a novel dataset of 1,332 deployed DAOs, each manually classified according to these criteria, and made the dataset publicly available for future research. Using this dataset, we trained an Long Short-term Memory (LSTM) model that classifies DAO activity with 75% accuracy, outperforming any individual heuristic.

The repo contains multiple parts of the thesis:

1. `model/` is the LSTM model used to predict the activity state of a DAO. It also contains some helper scripts and the code for the simple heuristics.
2. `preprocessing/` contains all scripts regarding data aggregation and data preparation for the LSTM model and the heuristics.
3. There is also a _Manual Classification Utility Program_ Which exists in the form of a simple express server in the root of this repository.

## Check that the dataset is complete

> **Note:** Because the dataset files were too large for git to handle, they are exempted from the repo.

However, the `preprocessing/master.mjs` file is supposted to be an automagic "manual" describing which files you need to place in which directories. Run it using: `cd preprocessing; bun run master.mjs`

If it exits without an error, you can be sure that your dataset is at the right locations and error free!

## Manual Classification Utility Program

The program spawns a server reading all coins from `preprocessing/normalizedAllSantimentProjects.json`.
You can then view all of them in a list and click on the slug of the project you would like to see the time series data of. After you click on the time when the project became inactive, on the Price USD chart, it will write the classification to the file: `classifiedProjects.json` and continue to load the next project.

### To run:

```
node server.mjs
```

Then navigate to: `http://localhost:3000/index.html`

## Models

No matter what you do, first make sure to `cd model/`.
It is also advised to use a virtual environment.

There are two types of models: ML model and Heuristics.

### LSTM Model

To **train** the LSTM:

1. Change the parameters of the model in `lstm.py` (and the name).
2. Run: `python lstm.py`

Since the model works with large timeseries it can sometimes crash on `mps`. Therefore it has a failsafe built in. So rerunning the script after it crashed makes sure that it starts from the same position it was last at.

To **test** the LSTM there are multiple ways the best way is to run `prc.py` which automatically invoces `test.py`.

### Heuristics

```
python heuristics.py
```
