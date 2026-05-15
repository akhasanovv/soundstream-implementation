# soundstream implementation

## installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## data preparation

folder structure:

```text
data/librispeech/
- train-clean-100/
- test-clean/
```

`.flac` and `.wav` audio files are also supported. 
for train: random crops of len `0.5s` wiht frequency `16kHz` (short audio are padded). 
for val/test: entire audiofile

## comet.ml logging

authorize before running:

```bash
comet login
```

or set api key in dotenv:

```bash
export COMET_API_KEY=...
```

> you can replace comet.ml (used by default) with wandb in `src/configs/soundstream.yaml`

## training

```bash
python3 train.py -cn=soundstream
```

config used by default:

1. `sample_rate=16000`
2. `batch_size=12`
3. `45000` training steps
4. constant LR 

use hydra to change training settings, for example:

```bash
python3 train.py -cn=soundstream trainer.device=cuda
```

## inference / evaluation

```bash
python3 inference.py -cn=inference_soundstream
```

`inference_soundstream` computes `STOI` and `NISQA` metrics on `test-clean` part of the dataset.

---

based on [Petr Grinberg's template](https://github.com/Blinorot/pytorch_project_template)