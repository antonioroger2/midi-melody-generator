import os
import numpy as np
from tensorflow.keras.models import load_model
from music21 import stream, note, chord, instrument
from src.network import MusicLSTM


MODEL_PATH = "models/best_model.keras" 
MAPPING_PATH = "models/mappings.pkl"
OUTPUT_DIR = "output"
NUM_NOTES_TO_GENERATE = 200




TEMPERATURE = 0.8 



TOP_K = 5 

def sample_with_temperature(preds, temperature=1.0, top_k=None):
    """
    Advanced sampling strategy using Temperature and Top-K filtering.
    """
    preds = np.asarray(preds).astype('float64')
    
    
    
    preds = np.log(preds + 1e-7) / temperature 
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    
    
    if top_k and top_k < len(preds):
        
        top_indices = preds.argsort()[-top_k:][::-1]
        mask = np.zeros_like(preds)
        mask[top_indices] = 1
        preds = preds * mask
        
        preds = preds / np.sum(preds)
        
    
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

def generate():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    
    print("Loading model and mappings...")
    lstm = MusicLSTM() 
    
    lstm.sequence_length = 100 
    
    try:
        lstm.load_mappings(MAPPING_PATH)
    except FileNotFoundError:
        print(f"Error: {MAPPING_PATH} not found. Did you run train.py?")
        return

    try:
        model = load_model(MODEL_PATH)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    
    print("Preparing seed sequence...")
    
    if len(lstm.notes) < lstm.sequence_length:
        print("Error: Not enough notes in mapping file to create a seed.")
        return

    network_input = []
    for i in range(0, len(lstm.notes) - lstm.sequence_length, 1):
        sequence_in = lstm.notes[i:i + lstm.sequence_length]
        network_input.append([lstm.pitch_to_int[char] for char in sequence_in])
    
    if not network_input:
        print("Error: No sequences generated. Dataset might be too small.")
        return

    
    start_index = np.random.randint(0, len(network_input)-1)
    pattern = network_input[start_index]
    
    print(f"Seed selected. Generating {NUM_NOTES_TO_GENERATE} notes with Temperature {TEMPERATURE}...")

    
    prediction_output = []
    
    for i in range(NUM_NOTES_TO_GENERATE):
        prediction_input = np.reshape(pattern, (1, len(pattern), 1))
        prediction_input = prediction_input / float(lstm.n_vocab) 
        
        prediction = model.predict(prediction_input, verbose=0)[0]
        
        
        index = sample_with_temperature(prediction, temperature=TEMPERATURE, top_k=TOP_K)
        
        result = lstm.int_to_pitch[index]
        prediction_output.append(result)
        
        
        pattern.append(index)
        pattern = pattern[1:]
        
        if i % 20 == 0:
            print(f"Generated {i} notes...", end='\r')

    
    print("\nConverting to MIDI with humanization...")
    output_notes = []
    
    
    output_notes.append(instrument.Piano())
    
    offset = 0
    
    
    
    
    possible_durations = [0.25, 0.5, 0.5, 0.5, 1.0] 
    
    for pattern in prediction_output:
        
        current_velocity = np.random.randint(50, 90)
        
        current_duration = np.random.choice(possible_durations)
        
        
        if ('.' in pattern) or pattern.isdigit():
            notes_in_chord = pattern.split('.')
            notes = []
            for current_note in notes_in_chord:
                new_note = note.Note(int(current_note))
                new_note.storedInstrument = instrument.Piano()
                new_note.volume.velocity = current_velocity
                new_note.quarterLength = current_duration
                notes.append(new_note)
            new_chord = chord.Chord(notes)
            new_chord.offset = offset
            output_notes.append(new_chord)
            
        
        else:
            new_note = note.Note(pattern)
            new_note.offset = offset
            new_note.storedInstrument = instrument.Piano()
            new_note.volume.velocity = current_velocity
            new_note.quarterLength = current_duration
            output_notes.append(new_note)
        
        
        offset += current_duration

    
    midi_stream = stream.Stream(output_notes)
    output_path = os.path.join(OUTPUT_DIR, "advanced_generated_music.mid")
    midi_stream.write('midi', fp=output_path)
    print(f"Success! Music saved to: {output_path}")

if __name__ == "__main__":
    generate()