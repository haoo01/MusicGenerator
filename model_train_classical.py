from IPython.display import clear_output
from pypianoroll import StandardTrack, Multitrack, write

import os
import os.path
import random
from pathlib import Path
import matplotlib.pyplot as plt
import pypianoroll
from pypianoroll import Multitrack, Track
from tqdm import tqdm
from livelossplot import PlotLosses
from livelossplot.outputs import MatplotlibPlot
import pandas as pd
import pretty_midi
import torch
import numpy as np

# os.chdir('/Users/willowwu/PycharmProjects/MET664final')

# set parameters
# Data
n_tracks = 5  # number of tracks
n_pitches = 72  # number of pitches
lowest_pitch = 24  # MIDI note number of the lowest pitch
n_samples_per_song = 8  # number of samples to extract from each song in the datset
n_measures = 4  # number of measures per sample
beat_resolution = 4  # temporal resolution of a beat (in timestep)
programs = [0, 0, 25, 33, 48]  # program number for each track
is_drums = [True, False, False, False, False]  # drum indicator for each track
track_names = ['Drums', 'Piano', 'Guitar', 'Bass', 'Strings']  # name of each track
tempo = 100

# Training
batch_size = 16
latent_dim = 128
n_steps = 30000

# Sampling
sample_interval = 100  # interval to run the sampler (in step)
n_samples = 4

# read the data
cleansed = pd.read_csv('cleansed_ids.txt', sep=" ", header=None)
id_list_love = pd.read_csv('id_list_rnb.txt', sep=" ", header=None)
id_list_love = id_list_love[0].tolist()
lpd_id = cleansed[4].tolist()

# read the data
X = pd.read_csv('msd_genre_dataset.txt', header=0, delimiter='\t')
X = X.iloc[8:].reset_index(drop=True)
split_values = X['# MILLION SONG GENRE DATASET'].str.split(',', expand=True)
mds_genre = pd.DataFrame()
mds_genre["genre"] = split_values[0]
mds_genre["track_id"] = split_values[1]
mds_genre = mds_genre.iloc[1:].reset_index(drop=True)

# check overlapping
overlapping1 = list(set(id_list_love) & set(lpd_id))
len(overlapping1)

measure_resolution = 4 * beat_resolution
tempo_array = np.full((4 * 4 * measure_resolution, 1), tempo)
assert 24 % beat_resolution == 0, (
    "beat_resolution must be a factor of 24 (the beat resolution used in "
    "the source dataset)."
)
assert len(programs) == len(is_drums) and len(programs) == len(track_names), (
    "Lengths of programs, is_drums and track_names must be the same."
)

id_list = overlapping1

# set raw data direction
dataset_root = Path("lpd_5/lpd_5_cleansed/")


def msd_id_to_dirs(msd_id):
    """Given an MSD ID, generate the path prefix.
    E.g. TRABCD12345678 -> A/B/C/TRABCD12345678"""
    return os.path.join(msd_id[2], msd_id[3], msd_id[4], msd_id)


# example
song_dir = dataset_root / msd_id_to_dirs('TREVDFX128E07859E0')
multitrack = pypianoroll.load(song_dir / os.listdir(song_dir)[0])
song_dir = dataset_root / msd_id_to_dirs('TREVDFX128E07859E0')
multitrack = pypianoroll.load(song_dir / os.listdir(song_dir)[0])
multitrack.trim(beat_resolution * 4, beat_resolution * 8)

data = []
# Iterate over all the songs in the ID list
for msd_id in tqdm(id_list):
    # Load the multitrack as a pypianoroll.Multitrack instance
    song_dir = dataset_root / msd_id_to_dirs(msd_id)
    multitrack = pypianoroll.load(song_dir / os.listdir(song_dir)[0])

    # Binarize the pianorolls
    multitrack.binarize()

    # Downsample the pianorolls (shape: n_timesteps x n_pitches)
    multitrack.set_resolution(beat_resolution)

    # Stack the pianoroll (shape: n_tracks x n_timesteps x n_pitches)
    pianoroll = (multitrack.stack() > 0)

    # Get the target pitch range only
    pianoroll = pianoroll[:, :, lowest_pitch:lowest_pitch + n_pitches]

    # Calculate the total measures
    n_total_measures = multitrack.get_max_length() // measure_resolution
    candidate = n_total_measures - n_measures
    target_n_samples = min(n_total_measures // n_measures, n_samples_per_song)

    # Randomly select a number of phrases from the multitrack pianoroll
    for idx in np.random.choice(candidate, target_n_samples, False):
        start = idx * measure_resolution
        end = (idx + n_measures) * measure_resolution

        # Skip the samples where some track(s) has too few notes
        if (pianoroll.sum(axis=(1, 2)) < 10).any():
            continue
        data.append(pianoroll[:, start:end])

# Stack all the collected pianoroll segments into one big array
random.shuffle(data)
data = np.stack(data)
print(f"Successfully collect {len(data)} samples from {len(id_list)} songs")
print(f"Data shape : {data.shape}")

tracks = []
for idx, (program, is_drum, track_name) in enumerate(zip(programs, is_drums, track_names)):
    pianoroll = np.pad(
        np.concatenate(data[:4], 1)[idx], ((0, 0), (lowest_pitch, 128 - lowest_pitch - n_pitches)))
    tracks.append(Track(name=track_name, program=program, is_drum=is_drum, pianoroll=pianoroll))
multitrack = Multitrack(tracks=tracks, tempo=tempo_array, resolution=beat_resolution)
axs = multitrack.plot()
plt.gcf().set_size_inches((16, 8))
for ax in axs:
    for x in range(measure_resolution, 4 * 4 * measure_resolution, measure_resolution):
        if x % (measure_resolution * 4) == 0:
            ax.axvline(x - 0.5, color='k')
        else:
            ax.axvline(x - 0.5, color='k', linestyle='-', linewidth=1)
plt.show()

data = torch.as_tensor(data, dtype=torch.float32)
dataset = torch.utils.data.TensorDataset(data)
data_loader = torch.utils.data.DataLoader(
    dataset, batch_size=batch_size, drop_last=True, shuffle=True)


class GeneraterBlock(torch.nn.Module):
    def __init__(self, in_dim, out_dim, kernel, stride):
        super().__init__()
        self.transconv = torch.nn.ConvTranspose3d(in_dim, out_dim, kernel, stride)
        self.batchnorm = torch.nn.BatchNorm3d(out_dim)

    def forward(self, x):
        x = self.transconv(x)
        x = self.batchnorm(x)
        return torch.nn.functional.relu(x)


class Generator(torch.nn.Module):
    """A convolutional neural network (CNN) based generator. The generator takes
    as input a latent vector and outputs a fake sample."""
    def __init__(self):
        super().__init__()
        self.transconv0 = GeneraterBlock(latent_dim, 256, (4, 1, 1), (4, 1, 1))
        self.transconv1 = GeneraterBlock(256, 128, (1, 4, 1), (1, 4, 1))
        self.transconv2 = GeneraterBlock(128, 64, (1, 1, 4), (1, 1, 4))
        self.transconv3 = GeneraterBlock(64, 32, (1, 1, 3), (1, 1, 1))
        self.transconv4 = torch.nn.ModuleList([
            GeneraterBlock(32, 16, (1, 4, 1), (1, 4, 1))
            for _ in range(n_tracks)
        ])
        self.transconv5 = torch.nn.ModuleList([
            GeneraterBlock(16, 1, (1, 1, 12), (1, 1, 12))
            for _ in range(n_tracks)
        ])

    def forward(self, x):
        x = x.view(-1, latent_dim, 1, 1, 1)
        x = self.transconv0(x)
        x = self.transconv1(x)
        x = self.transconv2(x)
        x = self.transconv3(x)
        x = [transconv(x) for transconv in self.transconv4]
        x = torch.cat([transconv(x_) for x_, transconv in zip(x, self.transconv5)], 1)
        x = x.view(-1, n_tracks, n_measures * measure_resolution, n_pitches)
        return x


class LayerNorm(torch.nn.Module):
    """An implementation of Layer normalization that does not require size
    information. Copied from https://github.com/pytorch/pytorch/issues/1959."""

    def __init__(self, n_features, eps=1e-5, affine=True):
        super().__init__()
        self.n_features = n_features
        self.affine = affine
        self.eps = eps
        if self.affine:
            self.gamma = torch.nn.Parameter(torch.Tensor(n_features).uniform_())
            self.beta = torch.nn.Parameter(torch.zeros(n_features))

    def forward(self, x):
        shape = [-1] + [1] * (x.dim() - 1)
        mean = x.view(x.size(0), -1).mean(1).view(*shape)
        std = x.view(x.size(0), -1).std(1).view(*shape)
        y = (x - mean) / (std + self.eps)
        if self.affine:
            shape = [1, -1] + [1] * (x.dim() - 2)
            y = self.gamma.view(*shape) * y + self.beta.view(*shape)
        return y


class DiscriminatorBlock(torch.nn.Module):
    def __init__(self, in_dim, out_dim, kernel, stride):
        super().__init__()
        self.transconv = torch.nn.Conv3d(in_dim, out_dim, kernel, stride)
        self.layernorm = LayerNorm(out_dim)

    def forward(self, x):
        x = self.transconv(x)
        x = self.layernorm(x)
        return torch.nn.functional.leaky_relu(x)


class Discriminator(torch.nn.Module):
    """A convolutional neural network (CNN) based discriminator. The
    discriminator takes as input either a real sample (in the training data) or
    a fake sample (generated by the generator) and outputs a scalar indicating
    its authentity.
    """
    def __init__(self):
        super().__init__()
        self.conv0 = torch.nn.ModuleList([
            DiscriminatorBlock(1, 16, (1, 1, 12), (1, 1, 12)) for _ in range(n_tracks)
        ])
        self.conv1 = torch.nn.ModuleList([
            DiscriminatorBlock(16, 16, (1, 4, 1), (1, 4, 1)) for _ in range(n_tracks)
        ])
        self.conv2 = DiscriminatorBlock(16 * 5, 64, (1, 1, 3), (1, 1, 1))
        self.conv3 = DiscriminatorBlock(64, 64, (1, 1, 4), (1, 1, 4))
        self.conv4 = DiscriminatorBlock(64, 128, (1, 4, 1), (1, 4, 1))
        self.conv5 = DiscriminatorBlock(128, 128, (2, 1, 1), (1, 1, 1))
        self.conv6 = DiscriminatorBlock(128, 256, (3, 1, 1), (3, 1, 1))
        self.dense = torch.nn.Linear(256, 1)

    def forward(self, x):
        x = x.view(-1, n_tracks, n_measures, measure_resolution, n_pitches)
        x = [conv(x[:, [i]]) for i, conv in enumerate(self.conv0)]
        x = torch.cat([conv(x_) for x_, conv in zip(x, self.conv1)], 1)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = x.view(-1, 256)
        x = self.dense(x)
        return x

def compute_gradient_penalty(discriminator, real_samples, fake_samples):
    """Compute the gradient penalty for regularization. Intuitively, the
    gradient penalty help stablize the magnitude of the gradients that the
    discriminator provides to the generator, and thus help stablize the training
    of the generator."""
    # Get random interpolations between real and fake samples
    alpha = torch.rand(real_samples.size(0), 1, 1, 1)#.cuda()
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples))
    interpolates = interpolates.requires_grad_(True)
    # Get the discriminator output for the interpolations
    d_interpolates = discriminator(interpolates)
    # Get gradients w.r.t. the interpolations
    fake = torch.ones(real_samples.size(0), 1)#.cuda()
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]
    # Compute gradient penalty
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty


def train_one_step(d_optimizer, g_optimizer, real_samples):
    """Train the networks for one step."""
    # Sample from the lantent distribution
    latent = torch.randn(batch_size, latent_dim)

    # Transfer data to GPU
    # if torch.cuda.is_available():
    # real_samples = real_samples.cuda()
    # latent = latent.cuda()

    # === Train the discriminator ===
    # Reset cached gradients to zero
    d_optimizer.zero_grad()
    # Get discriminator outputs for the real samples
    prediction_real = discriminator(real_samples)
    # Compute the loss function
    # d_loss_real = torch.mean(torch.nn.functional.relu(1. - prediction_real))
    d_loss_real = -torch.mean(prediction_real)
    # Backpropagate the gradients
    d_loss_real.backward()

    # Generate fake samples with the generator
    fake_samples = generator(latent)
    # Get discriminator outputs for the fake samples
    prediction_fake_d = discriminator(fake_samples.detach())
    # Compute the loss function
    # d_loss_fake = torch.mean(torch.nn.functional.relu(1. + prediction_fake_d))
    d_loss_fake = torch.mean(prediction_fake_d)
    # Backpropagate the gradients
    d_loss_fake.backward()

    # Compute gradient penalty
    gradient_penalty = 10.0 * compute_gradient_penalty(
        discriminator, real_samples.data, fake_samples.data)
    # Backpropagate the gradients
    gradient_penalty.backward()

    # Update the weights
    d_optimizer.step()

    # === Train the generator ===
    # Reset cached gradients to zero
    g_optimizer.zero_grad()
    # Get discriminator outputs for the fake samples
    prediction_fake_g = discriminator(fake_samples)
    # Compute the loss function
    g_loss = -torch.mean(prediction_fake_g)
    # Backpropagate the gradients
    g_loss.backward()
    # Update the weights
    g_optimizer.step()

    return d_loss_real + d_loss_fake, g_loss

# Create data loader
# data_loader = get_data_loader()

# Create neural networks
discriminator = Discriminator()
generator = Generator()
print("Number of parameters in G: {}".format(
    sum(p.numel() for p in generator.parameters() if p.requires_grad)))
print("Number of parameters in D: {}".format(
    sum(p.numel() for p in discriminator.parameters() if p.requires_grad)))

# Create optimizers
d_optimizer = torch.optim.Adam(
    discriminator.parameters(), lr=0.001,  betas=(0.5, 0.9))
g_optimizer = torch.optim.Adam(
    generator.parameters(), lr=0.001, betas=(0.5, 0.9))

# Prepare the inputs for the sampler, which wil run during the training
sample_latent = torch.randn(n_samples, latent_dim)

# Transfer the neural nets and samples to GPU
#if torch.cuda.is_available():
    #discriminator = discriminator.cuda()
    #generator = generator.cuda()
    #sample_latent = sample_latent.cuda()

# Create an empty dictionary to sotre history samples
history_samples = {}

# Create a LiveLoss logger instance for monitoring
liveloss = PlotLosses(outputs=[MatplotlibPlot(cell_size=(6,2))])

# Initialize step
step = 0

# Create a progress bar instance for monitoring
progress_bar = tqdm(total=n_steps, initial=step, ncols=80, mininterval=1)

# Start iterations
while step < n_steps + 1:
    # Iterate over the dataset
    for real_samples in data_loader:
        # Train the neural networks
        generator.train()
        d_loss, g_loss = train_one_step(d_optimizer, g_optimizer, real_samples[0])

        # Record smoothened loss values to LiveLoss logger
        if step > 0:
            running_d_loss = 0.05 * d_loss.item() + 0.95 * running_d_loss
            running_g_loss = 0.05 * g_loss.item() + 0.95 * running_g_loss
        else:
            running_d_loss, running_g_loss = 0.0, 0.0
        liveloss.update({'negative_critic_loss': -running_d_loss})
        # liveloss.update({'d_loss': running_d_loss, 'g_loss': running_g_loss})

        # Update losses to progress bar
        progress_bar.set_description_str(
            "(d_loss={: 8.6f}, g_loss={: 8.6f})".format(d_loss, g_loss))

        if step % sample_interval == 0:
            # Get generated samples
            generator.eval()
            samples = generator(sample_latent).cpu().detach().numpy()

            # Display loss curves
            clear_output(True)
            # if step > 0:
            # liveloss.send()

            # Display generated samples
            samples = samples.transpose(1, 0, 2, 3).reshape(n_tracks, -1, n_pitches)
            tracks = []
            for idx, (program, is_drum, track_name) in enumerate(
                    zip(programs, is_drums, track_names)
            ):
                pianoroll = np.pad(
                    samples[idx] > 0.5,
                    ((0, 0), (lowest_pitch, 128 - lowest_pitch - n_pitches))
                )
                tracks.append(
                    Track(
                        name=track_name,
                        program=program,
                        is_drum=is_drum,
                        pianoroll=pianoroll
                    )
                )
            m = Multitrack(
                tracks=tracks,
                tempo=tempo_array,
                resolution=beat_resolution
            )
            axs = m.plot()
            plt.gcf().set_size_inches((16, 8))
            for ax in axs:
                for x in range(
                        measure_resolution,
                        4 * measure_resolution * n_measures,
                        measure_resolution
                ):
                    if x % (measure_resolution * 4) == 0:
                        ax.axvline(x - 0.5, color='k')
                    else:
                        ax.axvline(x - 0.5, color='k', linestyle='-', linewidth=1)
            # plt.show()

        step += 1
        progress_bar.update(1)
        if step >= n_steps:
            break


# save the model trained
# Save state dictionaries
torch.save(generator.state_dict(), 'generator_state_dict3.pth')
torch.save(discriminator.state_dict(), 'discriminator_state_dict3.pth')

# Load the state dictionaries
generator.load_state_dict(torch.load('generator_state_dict3.pth'))
discriminator.load_state_dict(torch.load('discriminator_state_dict3.pth'))

# Initialize the models (make sure the architecture is the same as used during training)
generator = Generator() # Replace with your actual generator initialization
discriminator = Discriminator() # Replace with your actual discriminator initialization


# Assuming 'generator' is your trained generator model and 'latent_dim' is the dimension of your noise vector
noise = torch.randn(1, latent_dim)  # Generate a random noise vector
generated_music = generator(noise)  # Generate music

# Assuming 'generated_music' is your tensor with shape (num_tracks, time_steps, pitches)
# and your tensor is on a GPU, you need to move it to CPU and convert to a numpy array
generated_music_np = generated_music.squeeze().detach().cpu().numpy()

tracks = []
for i in range(generated_music_np.shape[0]):  # Assuming the first dimension is the track dimension
    # Binarize the track (you can adjust the threshold and the value for 'on' notes)
    binarized_track = (generated_music_np[i] > 0.5).astype(np.uint8) * 127
    # Create a Track object
    track = Track(pianoroll=binarized_track, program=0, is_drum=False, name=f'Track {i}')
    tracks.append(track)

# Assuming you have a list of Track objects created earlier
standard_tracks = []
for track in tracks:
    # Convert each Track to StandardTrack
    standard_track = StandardTrack(
        pianoroll=track.pianoroll,
        program=track.program,
        is_drum=track.is_drum,
        name=track.name
    )
    standard_tracks.append(standard_track)

# Create a Multitrack object with StandardTrack objects
multitrack = Multitrack(tracks=standard_tracks)

# Write to a MIDI file
midi_file = 'generated_music.mid'
write(midi_file, multitrack)


# Load your binarized tensor here, it should be a boolean array
# with shape (num_tracks, time_steps, pitches)
# For example:
# binarized_tensor = np.load('path_to_your_binarized_tensor.npy')

# Define your MIDI parameters
bpm = 120  # Beats per minute for the song
beats_per_time_step = 1/16  # How many beats does one time step represent? (e.g., 1/4 for a quarter note)
ppqn = 480  # Pulses per quarter note (ticks per beat), standard MIDI resolution


import numpy as np

# Assuming 'generated_music_np' is your tensor after converting to numpy
threshold = 0.7  # Adjust threshold as needed
minimum_note_duration = 2  # Minimum duration for notes to be on

# Binarize the tensor with a higher threshold
binarized_tensor = (generated_music_np > threshold).astype(np.uint8)

# Process tensor to include minimum note duration
for track in range(binarized_tensor.shape[0]):  # Iterating over tracks
    for pitch in range(binarized_tensor.shape[2]):  # Iterating over pitches
        note_on = False
        duration = 0
        for time_step in range(binarized_tensor.shape[1]):  # Iterating over time steps
            # Checking the value of a single element
            if binarized_tensor[track, time_step, pitch] == 1:
                if not note_on:
                    note_on = True
                    duration = 1
                elif duration < minimum_note_duration:
                    duration += 1
                else:
                    # The note has been on for at least the minimum duration, so keep it on
                    continue
            else:
                if note_on and duration < minimum_note_duration:
                    # The note was on but did not reach the minimum duration, turn it off
                    binarized_tensor[track, time_step-duration:time_step, pitch] = 0
                # Reset the note_on and duration
                note_on = False
                duration = 0

# Calculate the number of ticks per time step
ticks_per_time_step = ppqn * beats_per_time_step

# Create a PrettyMIDI object with the appropriate tempo
midi_file = pretty_midi.PrettyMIDI(initial_tempo=bpm)

# Add each track from the binarized tensor to the MIDI file
for track_index in range(binarized_tensor.shape[0]):
    # Create an Instrument instance
    instrument = pretty_midi.Instrument(program=0)  # Set to appropriate MIDI program (instrument)

    # Iterate through each pitch
    for pitch in range(binarized_tensor.shape[2]):
        current_note_start = None
        # Iterate through each time step
        for time_step in range(binarized_tensor.shape[1]):
            is_note_on = binarized_tensor[track_index, time_step, pitch]
            # Note start
            if is_note_on and current_note_start is None:
                current_note_start = time_step
            # Note end
            elif not is_note_on and current_note_start is not None:
                # Create a new Note object for this note
                note_start_time = current_note_start * ticks_per_time_step / ppqn
                note_end_time = time_step * ticks_per_time_step / ppqn
                note = pretty_midi.Note(
                    velocity=100,  # Note velocity
                    pitch=pitch + 21,  # MIDI note number
                    start=note_start_time,  # Note start time in seconds
                    end=note_end_time  # Note end time in seconds
                )
                # Add it to the instrument
                instrument.notes.append(note)
                current_note_start = None
        # Check if the last note extends to the end
        if current_note_start is not None:
            note_start_time = current_note_start * ticks_per_time_step / ppqn
            note_end_time = binarized_tensor.shape[1] * ticks_per_time_step / ppqn
            note = pretty_midi.Note(
                velocity=100,
                pitch=pitch + 21,
                start=note_start_time,
                end=note_end_time
            )
            instrument.notes.append(note)

    # Add the instrument to the PrettyMIDI object
    midi_file.instruments.append(instrument)

# Save the MIDI file
output_path = 'generated_music_.mid'
midi_file.write(output_path)


