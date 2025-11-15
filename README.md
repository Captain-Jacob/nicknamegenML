## Overview
This project is a character-level LSTM (CharLSTM) that suggests new creative names,
based on a scoring system and several seed datasets.  
It is good at generating fresh, fantasy-style names, but it is still an experiment
— so don’t expect perfect results every time.

---

## Installation

1. Install PyTorch (CPU or GPU)  

   Go to: https://pytorch.org/get-started/locally/  
   Choose the correct command for your OS, Python version, and **CPU/GPU**.  
   It will look similar to:

       pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/XXXX

   Replace `XXXX` with the variant suggested by the PyTorch website.

2. Install NumPy:

       pip install numpy

---

## Project structure

The repository is expected to look like this:

    project_root/
    │
    ├── seeds/
    │   ├── dnd_feminine.txt
    │   ├── goddess_names.txt
    │   ├── japanese_female.txt
    │   ├── jp_name.txt
    │   └── female_names.txt     # optional: include only if you want this dataset
    │
    ├── finalsuggestion.txt      # running top suggestions (best-scoring names)
    ├── generated_long.txt       # log of generated long names
    ├── generated_short.txt      # log of generated short names
    ├── model.pt                 # saved PyTorch model weights
    ├── nameforge_ml.py          # main CharLSTM training + generation script
    └── test.py                  # small test / demo script

---

## Basic usage (example)

After training or using the provided model:

    python nameforge_ml.py --train_epochs 10 --generate 500 --temperature 0.8

- `--train_epochs` : number of training epochs
- `--generate`     : how many names to generate
- `--temperature`  : controls randomness (lower = safer, higher = wilder)
