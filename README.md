# Diff-Phoneme: Diffusion Model-Based Neural Speech Decoding from EEG Signals

*Submitted to ICASSP 2025*

Diff-Phoneme is a diffusion model-based neural speech decoding framework that predicts phoneme sequences from electroencephalography (EEG) signals. This framework enables the generation of unseen words using a restricted word corpus, enhancing brain-computer interfaces (BCIs) by decoding and reconstructing speech from non-invasive EEG signals.

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
  - [Data Preparation](#data-preparation)
  - [Training](#training)
  - [Testing](#testing)
- [Architecture](#architecture)
- [Data Preprocessing](#data-preprocessing)
- [Training](#training)
- [Evaluation](#evaluation)
- [Results](#results)
- [License](#license)

## Overview

Brain-computer interfaces (BCIs) can revolutionize communication by enabling individuals to generate speech or text directly from brain signals. Diff-Phoneme focuses on decoding EEG signals into phoneme sequences by leveraging a diffusion model and a sequence-to-sequence (Seq2Seq) architecture. This process allows the reconstruction of speech or text, including unseen words not present in the training data.

The framework integrates the following components:
- **Conditional Autoencoder (CAE)**
- **Denoising Diffusion Probabilistic Model (DDPM)**
- **Seq2Seq model**  
These components work together to accurately decode phoneme sequences from EEG signals.

## Features

- **Phoneme Sequence Prediction:** Learns to predict phoneme sequences from EEG signals using a restricted word corpus and reconstructs unseen words.
- **Diffusion Model:** Incorporates a Denoising Diffusion Probabilistic Model (DDPM) to generate latent EEG representations, ensuring robust speech decoding.
- **Trie-based Word Matching:** Maps predicted phoneme sequences to the closest matching words from a predefined word corpus.
- **Supports Unseen Word Generation:** Generates words composed of phonemes from a restricted corpus, even if those words were not in the training set.

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
1. Data Acquisition:
EEG data should be recorded using a 64-channel EEG system and preprocessed following a standard protocol (e.g., using EEGLAB for filtering and segmentation).

2. Signal Preprocessing:
Resample signals to 500 Hz.
Apply baseline correction.
Rereference EEG signals using the common average method.

### Training
To train the model on your dataset, run:
   ```bash
   python train.py

This will train the model using the provided EEG signals and corresponding phoneme sequences. During training, the model learns to predict phoneme sequences from EEG signals, enabling it to generate unseen words composed of familiar phonemes.

