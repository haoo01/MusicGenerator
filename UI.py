import streamlit as st
import torch
import numpy as np
from pypianoroll import Multitrack, Track
from model_module import Generator, Discriminator  
from pypianoroll import StandardTrack, Multitrack, write
import pretty_midi
import mido
import pygame
import tempfile
import os
import os.path
from threading import Thread
from pathlib import Path

# set up data
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
n_steps = 20000

# Sampling
sample_interval = 100  # interval to run the sampler (in step)
n_samples = 4
measure_resolution = 4 * beat_resolution
tempo_array = np.full((4 * 4 * measure_resolution, 1), tempo)
assert 24 % beat_resolution == 0, (
    "beat_resolution must be a factor of 24 (the beat resolution used in "
    "the source dataset)."
)
assert len(programs) == len(is_drums) and len(programs) == len(track_names), (
    "Lengths of programs, is_drums and track_names must be the same."
)
# Assuming 'latent_dim' is the size of the noise vector for the generator
 # Replace with the actual dimension of your latent space

# Initialize the generator and discriminator
generator = Generator()
discriminator = Discriminator()

def change_instrument(midi_file, track_instruments, drum_track=None):
    # Load the MIDI file
    mid = mido.MidiFile(midi_file)

    new_mid = mido.MidiFile()
    for i, track in enumerate(mid.tracks):
        new_track = mido.MidiTrack()
        new_mid.tracks.append(new_track)

        if drum_track is not None and i == drum_track:
            # Set the channel to 9 for drums (which is channel 10 in MIDI terms)
            for msg in track:
                if msg.type == 'note_on' or msg.type == 'note_off':
                    msg = msg.copy(channel=9)
                new_track.append(msg)
        else:
            # Change instrument for other tracks
            if i in track_instruments:
                instrument = track_instruments[i]
                program_change = mido.Message('program_change', program=instrument)
                new_track.append(program_change)

            for msg in track:
                new_track.append(msg)

    return new_mid


# Get the current directory
current_dir = os.getcwd()

# Set up data path relative to the current directory
dataset_root = os.path.join(current_dir, "lpd_5/lpd_5_cleansed/")

# Set up model paths relative to the current directory
model_paths = {
    'Love': 'models/generator_state_dict_love.pth',
    'Rock': 'models/generator_state_dict_rock.pth',
    'Classic': 'models/generator_state_dict_classic.pth',
    'Dance': 'models/generator_state_dict_dance.pth',
    'Indie': 'models/generator_state_dict_indie.pth'
}

# Function to generate a tensor representing music and save as MIDI
def generate_music(genre):
    # Load the corresponding model based on the selected genre
    if genre in model_paths:
        model_path = os.path.join(current_dir, model_paths[genre])
        generator.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        discriminator.load_state_dict(torch.load(model_path.replace('generator', 'discriminator'), map_location=torch.device('cpu')))
    else:
        raise ValueError('Invalid genre')

    # Generate latent vectors
    noise = torch.randn(1, latent_dim, 1, 1, 1)  # Batch size is 1
    
    # Generate music tensor

    generated_music = generator(noise).cpu()
    
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

# Create a Multitrack object with StandardTrack objects
    multitrack = Multitrack(tracks=standard_tracks)


# Load your binarized tensor here, it should be a boolean array
# with shape (num_tracks, time_steps, pitches)
# For example:
# binarized_tensor = np.load('path_to_your_binarized_tensor.npy')

    # Define your MIDI parameters
    bpm = 120  # Beats per minute for the song
    beats_per_time_step = 1/4  # How many beats does one time step represent? (e.g., 1/4 for a quarter note)
    ppqn = 480  # Pulses per quarter note (ticks per beat), standard MIDI resolution

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
    output_path = './musics/generated_music_temp.mid'
    midi_file.write(output_path)

    # Convert the tensor to a MIDI file (This function needs to be implemented)
      # Replace with your function to convert tensor to MIDI
    return output_path
    

# 5 tracks, 72 pitches, 24 is the lowest pitch
# Define the instruments for each track (excluding the drum track)
track_instruments = {
    1: 1,  # Bright Acoustic Piano
    2: 24, # Acoustic Guitar (nylon)
    3: 32, # Acoustic Bass
    4: 48  # Violin
}

# Define genre for selection

genre = ['Love', 'Rock', 'Dance', 'Classic', 'Indie']

midi_path = None

# Streamlit interface
# Genre selection in a neat layout
col1, col2 = st.columns(2)
with col1:
    selected_genre = st.selectbox('Select Genre', genre)

# Progress bar
progress_bar = st.empty()

# Function to update progress bar (example function, adjust as needed)
def update_progress():
    progress = 0
    progress_bar.progress(progress)
    while progress < 100:
        # Update progress as the music generation progresses
        progress += 10  # Update this based on actual progress
        progress_bar.progress(progress)

# Generate music button
if st.button('Generate Music'):
    # Update the user and show progress
    st.info('Generating music, please wait...')
    update_progress()  # Call this function to update the progress
    midi_path = generate_music(selected_genre)
    progress_bar.empty()  # Clear the progress bar
    st.success('Music generation complete!')

# Function to play MIDI
def play_midi(file_path):
    freq = 44100    # audio CD quality
    bitsize = -16   # unsigned 16 bit
    channels = 2    # 1 is mono, 2 is stereo
    buffer = 1024    # number of samples
    pygame.mixer.init(freq, bitsize, channels, buffer)
    pygame.mixer.music.set_volume(0.8)  # volume

    try:
        pygame.mixer.music.load(file_path)
        pygame.mixer.music.play()
    except pygame.error as e:
        print("Error playing MIDI file: ", e)
        return

# Play and Stop buttons in a row
col3, col4 = st.columns(2)
with col3:
    if st.button('Play'):
        st.info('Playing music...')
        midi_path = generate_music(selected_genre)
        new_midi = change_instrument(midi_path, track_instruments)
        new_midi.save(midi_path)
        Thread(target=play_midi, args=(midi_path,)).start()

with col4:
    if st.button('Stop'):
        pygame.mixer.music.stop()
        st.info('Music playback stopped.')

# Clean up and remove the generated MIDI file
if midi_path and os.path.exists(midi_path):
    os.unlink(midi_path)

st.markdown("---")
st.markdown("© 2023 Music Generator - All Rights Reserved")


