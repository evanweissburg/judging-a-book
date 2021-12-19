# Judging a book by its cover: Predicting the marginal impact of title on Reddit post popularity

Evan Weissburg, Arya Kumar, Paramveer Dhillon

Paper forthcoming at ICWSM 2022

## Installation

Make sure that you have python3.8 or above installed on your machine.

### Create a Virtual Environment:

```bash
$ python -m venv venv
$ source venv/bin/activate
$ pip install -r requirements.txt
```

### Set CUDA devices:

Edit `src/helpers/constants.py` to set `DEVICES` to the list of CUDA devices you want to use.

### Download Dataset & Preprocess Data

If you would like to operate on the default list of subreddits we used in our paper, run:

```bash
$ python -m src.preprocess
```

If you would instead like to pass a custom list of subreddits, use the following:

```bash
$ python -m src.preprocess <SUBREDDIT> ... <SUBREDDIT>
```

## Usage

### Freeform Training

To train some set of models, tuning all parameters, use the `freeform.py` script. All model configs
are checked for validity before training. For example:

```python
for model_folds in scheduler.train(
    [
        OursModelConfig(
            subreddit="depression",
            training_scheme=TrainAll(),
            epochs=25,
        ),
    ]
):
    model, _ = model_folds[0]
    model.write()
```

### Table and Figure Replication

All tables and figures from the paper can easily be replicated using the scripts in the `src/tables`
and `src/figures` directories. To make generation efficient, the scripts expect model files for
previously trained models using our `freeform` script.

For example, to generate table 1 in the paper, we run:

```bash
python3 -m src.tables.subreddit_pairs --output_name subreddit_pairs
```

To generate an Attention Directed Graph (ADG) of sports, we would train our model by running freeform
with the following config:

```python
OursModelConfig(
    subreddit="sports",
    training_scheme=TrainAll(),
    epochs=25,
)
```

Then, we would generate the ADG by running:

```bash
python3 -m src.figures.attention_directed_graph \
    --subreddit sports \
    --model_name ours_sports
    --output_name adg_sports
```
