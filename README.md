# Music Generation with Deep Learning
## Overview
"Music Generation with Deep Learning" harnesses AI, specifically GANs, to create multi-track music. Led by the MSSP Team, this project explores AI's capabilities in generating aesthetically pleasing compositions.

## Description
1. Data: Includes the Lakh Pianoroll Dataset under lpd_5/lpd_5_cleansed for training the model, and cleansed_ids.txt to reference the specific subset of data used.

2. Models: Contains the trained state dictionaries for the Generative Adversarial Network (GAN) models. There are separate model files for the discriminator and generator for various music genres such as classic, dance, indie, love, and rock.

3. Musics: Stores the MIDI files generated by the AI model. Example files like generated_music_indie.mid demonstrate the model's capability to generate indie music.

4. UI.py: The Python script that launches the Streamlit interface, enabling user interaction with the model for music generation.

5. Model_module / Model_training_dance: Comprises the training scripts and modules specifically for the dance genre, including defining the neural network blocks and the training loop with gradient penalty calculation.

## Dependencies Installation
To make sure all these necessary libraries are in place to run this project: numpy, pypianoroll, matplotlib, streamlit, pretty_midi, mido, pygame, torch, please run this command in terminal first:

$ pip install numpy pypianoroll matplotlib streamlit pretty_midi mido pygame torch

## Usage
Open terminal and enter: streamlit run UI.py to run the model, generate music, and interacte with the Streamlit interface.

## Authors
MSSP Team: Tao Guo, Jiaqi Sun, Jing Wu, Hao He, Xu Luo
