# Voice Anti-spoofing with LCNN on ASVspoof2019 LA

This project implements a voice anti-spoofing countermeasure for the Logical Access (LA) part of the ASVspoof2019 dataset.  
The goal is to distinguish genuine human speech from spoofed speech produced by text-to-speech or voice conversion systems.

The model takes an audio recording as input and outputs a score: higher scores correspond to bona fide speech, while lower scores correspond to spoofed speech.

The implementation is based on a Light CNN (LCNN) architecture with Max-Feature-Map activations. The network follows the LCNN countermeasure described in:

- Lavrentyeva et al., *STC Antispoofing Systems for the ASVspoof2019 Challenge*
- Wang & Yamagishi, *A Comparative Study on Recent Neural Spoofing Countermeasures*

The project is built using the [PyTorch Project Template](https://github.com/Blinorot/pytorch_project_template).

---

## Results

| Front end | Loss | dev EER, % | eval EER, % |
| --------- | ---- | ---------- | ----------- |
| STFT | weighted cross entropy | TODO | TODO |

Training curves and logs are not included yet.  
TODO: add the final W&B report link.

---

## Approach

### Features

I use a log power magnitude spectrum as the input representation.

The feature extraction setup follows the configuration used in the STC ASVspoof2019 system:

- Blackman window of 1724 samples
- hop size of 130 samples, which is about 0.0081 seconds at 16 kHz
- 863 frequency bins
- fixed input length of 600 frames

Recordings longer than 600 frames are cropped. During training the crop starts from a random position, while during evaluation it starts from the beginning.  
Shorter recordings are repeated until they are long enough.

I do not use voice activity detection or feature normalisation, following the setup described in the reference papers.

### Model

The model is implemented in:

```text
src/model/asvspoof_lcnn.py
```

It is a Light CNN with:

- convolutional blocks
- 1x1 Network-in-Network layers
- Max-Feature-Map activations
- max-pooling
- batch normalisation
- one hidden fully connected layer
- dropout
- final two-class output layer

The two output classes are:

```text
bona fide
spoof
```

The model has around 10 million parameters. Most of them are in the first fully connected layer.

### Loss

The model is trained with weighted cross entropy.

The ASVspoof2019 LA training set is strongly imbalanced: there are many more spoofed examples than bona fide examples. To compensate for this, the bona fide class is given a larger weight during training.

I use standard weighted cross entropy rather than margin-based losses, since previous work reports that cross entropy works competitively for LCNN-based spoofing countermeasures.

### Scoring

For each audio file, the final score is computed as the difference between the two logits:

```text
score = bona_fide_logit - spoof_logit
```

This means that:

- higher score = more likely bona fide
- lower score = more likely spoof

This matches the expected scoring convention for ASVspoof-style evaluation.

### Metric

The main metric is Equal Error Rate (EER).

EER is computed over the whole partition, not separately per mini-batch.  
The implementation uses the `compute_eer` function provided with the assignment:

```text
src/metrics/calculate_eer.py
```

The best checkpoint is selected using the development set EER.

---

## Installation

First, create and activate a Python environment. For example:

```bash
conda create -n antispoofing python=3.11
conda activate antispoofing
```

Then install the required packages:

```bash
pip install -r requirements.txt
```

---

## Data

Download the LA partition of ASVspoof2019.

One possible source is the Kaggle mirror:

```text
https://www.kaggle.com/datasets/awsaf49/asvpoof-2019-dataset
```

The data directory should have the following structure:

```text
LA/
├── ASVspoof2019_LA_cm_protocols/
│   ├── ASVspoof2019.LA.cm.train.trn.txt
│   ├── ASVspoof2019.LA.cm.dev.trl.txt
│   └── ASVspoof2019.LA.cm.eval.trl.txt
├── ASVspoof2019_LA_train/
│   └── flac/
├── ASVspoof2019_LA_dev/
│   └── flac/
└── ASVspoof2019_LA_eval/
    └── flac/
```

---

## Training

To train the model, run:

```bash
python3 train.py -cn=asvspoof data_dir=/path/to/LA
```

The `-cn=asvspoof` argument is important because the base template uses a different default config.

You can override config values from the command line. For example:

```bash
python3 train.py -cn=asvspoof data_dir=/path/to/LA trainer.n_epochs=20 dataloader.batch_size=64
```

Some useful overrides are:

```text
trainer.n_epochs=20
dataloader.batch_size=64
trainer.seed=10
trainer.device=cuda
```

Checkpoints are saved to:

```text
saved/${writer.run_name}/
```

The best model is stored as:

```text
model_best.pth
```

Training can be resumed from a checkpoint using:

```bash
python3 train.py -cn=asvspoof data_dir=/path/to/LA trainer.resume_from=checkpoint-epochN.pth
```

---

## Making a submission

To score a partition and write a submission file:

```bash
python3 make_submission.py \
    --checkpoint saved/fft-lcnn/model_best.pth \
    --part test \
    --out <your_university_login>.csv
```

The script writes a CSV file with two columns and no header, which is the expected format for `grading.py`.

Before scoring the test set, it is useful to run the script on the validation set:

```bash
python3 make_submission.py \
    --checkpoint saved/fft-lcnn/model_best.pth \
    --part val \
    --out val_scores.csv
```

The printed EER should be close to the validation EER logged during training for the same checkpoint.

---

## Repository structure

```text
src/
├── configs/           # Hydra configs
├── datasets/          # dataset loading and protocol parsing
├── loss/              # weighted cross entropy
├── metrics/           # EER calculation
├── model/             # LCNN and MFM layers
├── trainer/           # training and validation loop
└── logger/            # experiment writers

train.py               # training entry point
make_submission.py     # scoring and CSV generation
inference.py           # original template inference script
```

---

## Credits

This repository is based on the [PyTorch Project Template](https://github.com/Blinorot/pytorch_project_template) by Petr Grinberg.

The EER implementation in `src/metrics/calculate_eer.py` was provided with the assignment.

Main references:

- Lavrentyeva et al., *STC Antispoofing Systems for the ASVspoof2019 Challenge*
- Wang & Yamagishi, *A Comparative Study on Recent Neural Spoofing Countermeasures*
- Wu et al., *A Light CNN for Deep Face Representation with Noisy Labels*

---

## License

This project uses the MIT license. See `LICENSE` for details.
