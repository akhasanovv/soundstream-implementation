# soundstream implementation

this repository contains the implementation of [soundstream](https://arxiv.org/abs/2107.03312) audio codec in pytorch. you can download the trained model on [hf](https://huggingface.co/akhasanovv/soundstream-implementation)

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
4. `learning_rate=1e-4`, `Adam(beta1=0.5, beta2=0.9)` (as proposed in [seanet](https://arxiv.org/pdf/2009.02095))

use hydra to change training settings, for example:

```bash
python3 train.py -cn=soundstream trainer.device=cuda
```

in case you want to train this monster on kaggle, use:

```bash
python3 train.py -cn=soundstream \
  datasets.train.data_dir=PATH_TO_DATASET \
  datasets.val.data_dir=PATH_TO_DATASET \
  datasets.test.data_dir=PATH_TO_DATASET \
  trainer.n_epochs=1 \
  trainer.epoch_len=45000 \
  dataloader.batch_size=12 \
  writer.mode=online \
  trainer.override=true
```

## inference / evaluation

```bash
python3 inference.py -cn=inference_soundstream
```

`inference_soundstream` computes `STOI` and `NISQA` metrics on `test-clean` part of the dataset.

## credits

based on [Petr Grinberg's template](https://github.com/Blinorot/pytorch_project_template)