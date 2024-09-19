# Diff-Phoneme: Diffusion Model-Based Neural Speech Decoding from EEG Signals

Submitted to ICASSP 2025


Diff-Phoneme is a diffusion model-based neural speech decoding framework that predicts phoneme sequences from electroencephalography (EEG) signals, enabling the generation of unseen words using a restricted word corpus. This method is designed to enhance the capability of brain-computer interfaces (BCIs) to decode and reconstruct speech from non-invasive EEG signals.

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Architecture](#architecture)
- [Data Preprocessing](#data-preprocessing)
- [Training](#training)
- [Evaluation](#evaluation)
- [Results](#results)
- [License](#license)

## Overview

Brain-computer interfaces (BCIs) have the potential to revolutionize communication by enabling individuals to generate speech or text directly from brain signals. Diff-Phoneme focuses on decoding EEG signals into phoneme sequences, using a diffusion model and a sequence-to-sequence (Seq2Seq) architecture. This allows the reconstruction of speech or text, even for unseen words that were not present in the training data.

The framework integrates a Conditional Autoencoder (CAE), Denoising Diffusion Probabilistic Model (DDPM), and Seq2Seq model to accurately decode phoneme sequences from EEG signals.

## Features

- **Phoneme Sequence Prediction:** Learns phoneme sequences from EEG signals using a restricted word corpus and reconstructs unseen words.
- **Diffusion Model:** Incorporates a Denoising Diffusion Probabilistic Model (DDPM) to generate latent EEG representations for robust speech decoding.
- **Trie-based Word Matching:** Maps predicted phoneme sequences to the closest matching words in a predefined word corpus.
- **Supports Unseen Word Generation:** Capable of generating words composed of phonemes from a restricted corpus, even if those words were not in the training set.

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/yorgoon/Diff-Phoneme.git
   cd Diff-Phoneme

2. Install the required dependencies:

  ```bash
  pip install -r requirements.txt

## Usage

### Data Preparation

1. **Data Acquisition:**  
   EEG data should be recorded using a 64-channel EEG system and preprocessed using the standard protocol (e.g., EEGLAB for filtering and segmentation).

2. **Signal Preprocessing:**
   - Resample signals to 500 Hz.
   - Apply baseline correction.
   - Rereference EEG signals using the common average method.

### Training

To train the model on your dataset, run:

```bash
python train.py

This will train the model using the provided EEG signals and corresponding phoneme sequences. The model will learn to predict phoneme sequences from EEG signals during training and can be used to generate unseen words from familiar phonemes.

### Testing
Once the model is trained, you can test it on unseen EEG signals to decode phoneme sequences and reconstruct words:

```bash
python test.py
