import os
import numpy as np
from tensorflow.keras.models import load_model
from music21 import stream, note, chord, instrument
from src.network import MusicLSTM

MODEL_PATH = "models/music_model.keras"
MAPPING_PATH = "models/mappings.pkl"
OUTPUT_DIR = "output"
NUM_NOTES_TO_GENERATE = 100

def generate():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # 1. Load Resources
    print("Loading model and mappings...")
    lstm = MusicLSTM()
    lstm.load_mappings(MAPPING_PATH)
    model = load_model(MODEL_PATH)
    
    # 2. Prepare Seed Sequence
    print("Preparing seed sequence...")
    network_input = []
    for i in range(0, len(lstm.notes) - lstm.sequence_length, 1):
        sequence_in = lstm.notes[i:i + lstm.sequence_length]
        network_input.append([lstm.pitch_to_int[char] for char in sequence_in])
    
    start_index = np.random.randint(0, len(network_input)-1)
    pattern = network_input[start_index]
    
    # 3. Generate Music
    print(f"Generating {NUM_NOTES_TO_GENERATE} notes...")
    prediction_output = []

    for i in range(NUM_NOTES_TO_GENERATE):
        prediction_input = np.reshape(pattern, (1, len(pattern), 1))
        prediction_input = prediction_input / float(lstm.n_vocab) # Normalize
        
        prediction = model.predict(prediction_input, verbose=0)        
        # Select the best note
        index = np.argmax(prediction)
        result = lstm.int_to_pitch[index]
        prediction_output.append(result)
        
        # Move the window forward
        pattern.append(index)
        pattern = pattern[1:]

    # 4. Convert to MIDI
    print("Converting to MIDI...")
    offset = 0
    output_notes = []

    for pattern in prediction_output:
        # If it's a chord
        if ('.' in pattern) or pattern.isdigit():
            notes_in_chord = pattern.split('.')
            notes = []
            for current_note in notes_in_chord:
                new_note = note.Note(int(current_note))
                new_note.storedInstrument = instrument.Piano()
                notes.append(new_note)
            new_chord = chord.Chord(notes)
            new_chord.offset = offset
            output_notes.append(new_chord)
        # If it's a note
        else:
            new_note = note.Note(pattern)
            new_note.offset = offset
            new_note.storedInstrument = instrument.Piano()
            output_notes.append(new_note)
        
        offset += 0.5

    # 5. Save File
    midi_stream = stream.Stream(output_notes)
    output_path = os.path.join(OUTPUT_DIR, "generated_music.mid")
    midi_stream.write('midi', fp=output_path)
    print(f"Success! Music saved to: {output_path}")

if __name__ == "__main__":
    generate()