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

### Download Dataset & Preprocess Data

If you would like to operate on the default list of subreddits we used in our paper, run:

```bash
$ python src/preprocess.py
```

If you would instead like to pass a custom list of subreddits, use the following:

```bash
$ python src/preprocess.py <SUBREDDIT> ... <SUBREDDIT>
```

## Usage

### Freeform Training

TODO

### Table Replication

TODO

### Figure Replication

TODO
