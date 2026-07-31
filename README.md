# Voice Anti-spoofing with LCNN on ASVspoof2019 LA

This project implements a countermeasure (CM) system for the Logical Access (LA) partition of the ASVspoof2019 dataset. The task is to tell genuine human speech apart from speech produced by text-to-speech and voice conversion systems.

The model takes an audio recording and returns a single real-valued score. High scores correspond to bona fide speech and low scores to spoofing attacks, which is the convention required by the ASVspoof2019 evaluation plan.

The network is a Light CNN (LCNN) with Max-Feature-Map activations. It follows:

- Lavrentyeva et al., *STC Antispoofing Systems for the ASVspoof2019 Challenge* (arXiv:1904.05576) — the architecture itself;
- Wang & Yamagishi, *A Comparative Study on Recent Neural Spoofing Countermeasures for Synthetic Speech Detection* (arXiv:2103.11326) — the front end, the training recipe and the choice of loss function;
- Wu et al., *A Light CNN for Deep Face Representation with Noisy Labels* (arXiv:1511.02683) — the original Max-Feature-Map operation.

The repository is built on the [PyTorch Project Template](https://github.com/Blinorot/pytorch_project_template).

---

## Results

| Front end | Loss                   | dev EER, % | eval EER, % |
| --------- | ---------------------- | ---------- | ----------- |
| LFCC      | weighted cross entropy | 0.67       | 5.46        |

The model was trained for 4 epochs. The checkpoint was selected by the lowest EER on the development set, and the eval EER above is the value returned by the grading script for the predictions produced by that checkpoint.

The result is close to what the literature reports for a comparable system: Lavrentyeva et al. obtain 5.06% eval EER for their single LFCC-LCNN on the same partition.

Training curves and logs: https://api.wandb.ai/links/mariia-stolbova2005-hse-university/eosx2xw7

---

## Approach

### Features

The final system uses an LFCC front end configured as in Wang & Yamagishi (Sec. 3.1): a 20 ms window, a 10 ms shift, a 512-point FFT and a linearly spaced triangular filter bank of 20 channels. Delta and delta-delta coefficients are appended, which gives 60 dimensions per frame, and the zeroth cepstral coefficient is replaced by the log spectral energy of the frame. That study finds LFCC to outperform both the linear filter bank and the spectrogram for LCNN back ends, and the difference there is statistically significant.

A second front end is implemented as well, in `src/model/asvspoof_lcnn.py`. It reproduces the FFT-LCNN configuration of Lavrentyeva et al.: a raw log power magnitude spectrum computed with a Blackman window of 1724 samples and a hop of 130 samples, which is about 0.0081 s at 16 kHz, giving 863 frequency bins. The convolutional part of the network is the same in both models; only the input representation differs.

Every trial is brought to a fixed length of 77870 samples, roughly 4.9 seconds. Longer recordings are cropped: during training the crop starts at a random offset, during evaluation it always starts from the beginning. Shorter recordings are repeated until they are long enough. Repetition is used instead of zero padding because a block of silence produces a constant band in the log spectrum that the network can pick up as an artefact of the padding rather than of the speech.

Neither voice activity detection nor feature normalisation is applied. Both papers report that these steps hurt performance on this database: non-speech regions carry cues that are specific to the recording or synthesis pipeline, and removing them throws away useful information.

### Model

The architecture is the one given in Table 1 of Lavrentyeva et al. and is implemented in `src/model/asvspoof_lcnn.py`. It consists of a 5x5 convolution, four blocks that alternate 1x1 Network-in-Network layers with 3x3 convolutions, and max-pooling and batch normalisation placed exactly as in the paper. Every convolution is followed by a Max-Feature-Map 2/1 activation, which splits the channels in half and takes the element-wise maximum, so that each block outputs half as many channels as its convolution produces.

The classifier is a fully connected layer of 160 units followed by MFM, dropout, batch normalisation and a final two-class layer. Dropout is placed before the batch normalisation and its rate is 0.75, as in the paper. Weights are initialised with normal Kaiming initialisation.

The convolutional part accounts for about 158K parameters, which matches the per-layer counts in the paper. The total depends on the front end, since the flattening layer scales with the size of the feature map: about 619K parameters for the LFCC model and about 10.2M for the STFT one.

The two output classes are `spoof` (index 0) and `bona fide` (index 1).

### Loss

The model is trained with weighted cross entropy. The LA training partition is strongly imbalanced — 2580 bona fide utterances against 22800 spoofed ones — so the bona fide class is given a weight of 9, which is close to the actual ratio between the classes.

Margin-based losses were not used. The A-softmax of Lavrentyeva et al. is an obvious candidate, but the comparative study of Wang & Yamagishi shows that a plain sigmoid or cross entropy performs on par with margin-based softmax for LCNN back ends, while the latter requires tuning a margin hyper-parameter. Cross entropy was therefore preferred as the simpler option with no measurable loss in quality.

### Scoring

For each file the score is the difference between the two logits:

```text
score = bona_fide_logit - spoof_logit
```

so that a higher score means a more confident bona fide decision, as the evaluation plan requires.

### Metric

The metric is the Equal Error Rate. It is computed over the whole partition at once rather than averaged over mini-batches, since EER is a property of the pooled score distribution and batch-wise averaging would give a different and meaningless number. The computation uses the `compute_eer` routine supplied with the assignment, kept in `src/metrics/calculate_eer.py`.

EER is logged separately for the development and the evaluation partitions after every epoch, so that the training dynamics can be followed. The best checkpoint is chosen by the development EER, which is why the final eval EER is higher than the development one.

---

## Installation

Create and activate an environment:

```bash
conda create -n antispoofing python=3.11
conda activate antispoofing
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

---

## Data

Download the LA partition of ASVspoof2019. The [Kaggle mirror](https://www.kaggle.com/datasets/awsaf49/asvpoof-2019-dataset) is usually the fastest option.

The data directory is expected to look like this:

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

The final model is reproduced with:

```bash
python3 train.py -cn=asvspoof model=asvspoof_lcnn_lfcc \
    data_dir=/path/to/LA \
    trainer.n_epochs=4 \
    writer.run_name=lfcc-lcnn
```

Dropping `model=asvspoof_lcnn_lfcc` trains the STFT variant instead, which is the default in `src/configs/asvspoof.yaml`. The `-cn=asvspoof` flag is required in either case, since the template ships with a different default config.

The optimiser and the schedule follow Wang & Yamagishi, Sec. 3.2: Adam with a learning rate of 3e-4, betas of 0.9 and 0.999, eps of 1e-8, a batch size of 8, and the learning rate halved every ten epochs. The scheduler steps once per iteration, so `step_size` is set to ten epochs' worth of iterations rather than to ten.

Any config value can be overridden from the command line, for example:

```bash
python3 train.py -cn=asvspoof data_dir=/path/to/LA trainer.n_epochs=20 dataloader.batch_size=64 trainer.seed=10
```

Checkpoints are written to `saved/${writer.run_name}/`, with the best one saved as `model_best.pth`. Training can be resumed with `trainer.resume_from=checkpoint-epochN.pth`.

---

## Making a submission

To score the evaluation partition and write the submission file:

```bash
python3 make_submission.py \
    --checkpoint saved/lfcc-lcnn/model_best.pth \
    --data-dir /path/to/LA \
    --part test \
    --out <your_university_login>.csv
```

The output has two comma-separated columns, the trial key and the score, and no header row, which is what `grading.py` expects. The script also checks that the number of scored trials matches the protocol, that there are no duplicate keys and no NaNs, and that the scores are soft rather than binary.

It is worth running the same command with `--part val` first. It prints the pooled EER of the development partition, which has to agree with the value logged during training for the same checkpoint. If the two disagree, the preprocessing at inference time differs from the one used in training.

---

## Repository structure

```text
src/
├── configs/           # Hydra configs
├── datasets/          # protocol parsing, length handling
├── loss/              # weighted cross entropy
├── metrics/           # EER
├── model/             # LCNN, MFM, both front ends
├── trainer/           # training loop and per-epoch evaluation
└── logger/            # experiment writers

train.py               # training entry point
make_submission.py     # scoring and CSV generation
inference.py           # inference script inherited from the template
```

---

## Credits

The repository is based on the [PyTorch Project Template](https://github.com/Blinorot/pytorch_project_template) by Petr Grinberg. The EER implementation in `src/metrics/calculate_eer.py` was provided with the assignment. Everything else was written from scratch.

References:

- G. Lavrentyeva, S. Novoselov, A. Tseren, M. Volkova, A. Gorlanov, A. Kozlov. STC Antispoofing Systems for the ASVspoof2019 Challenge. Interspeech, 2019.
- X. Wang, J. Yamagishi. A Comparative Study on Recent Neural Spoofing Countermeasures for Synthetic Speech Detection. Interspeech, 2021.
- X. Wu, R. He, Z. Sun, T. Tan. A Light CNN for Deep Face Representation with Noisy Labels. IEEE TIFS, 2018.

---

## License

MIT. See `LICENSE`.
