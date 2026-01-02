import os
import pickle
import numpy as np
from music21 import converter, instrument, note, chord, pitch
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, BatchNormalization, Bidirectional
from tensorflow.keras.optimizers import Adamax
from tensorflow.keras.utils import to_categorical
from collections import Counter

class MusicLSTM:
    def __init__(self, sequence_length=100):
        # Increased sequence_length to 100 for better long-term melody structure
        self.sequence_length = sequence_length
        self.notes = []
        self.network_input = []
        self.n_vocab = 0
        self.pitch_to_int = {}
        self.int_to_pitch = {}
        self.model = None

    def load_data(self, data_dir, parsing=True, augment=True):
        if parsing:
            print(f"Loading MIDI files recursively from {data_dir}...")
            all_notes = []
            
            # 1. Recursive file search for nested folders (Kaggle dataset structure)
            midi_files = []
            for root, dirs, files in os.walk(data_dir):
                for file in files:
                    if file.endswith(".mid") or file.endswith(".midi"):
                        midi_files.append(os.path.join(root, file))

            print(f"Found {len(midi_files)} MIDI files.")

            for file_path in midi_files:
                try:
                    midi = converter.parse(file_path)
                    
                    # 2. Data Augmentation: Transpose to all 12 keys
                    # This multiplies dataset size by 12x
                    if augment:
                        # Get key (or estimate it)
                        k = midi.analyze('key')
                        # Calculate interval to C major / A minor
                        if k.mode == 'major':
                            i = note.Interval(k.tonic, pitch.Pitch('C'))
                        else:
                            i = note.Interval(k.tonic, pitch.Pitch('A'))
                        
                        # Transpose to 'neutral' key first
                        neutral_midi = midi.transpose(i)
                        
                        # Generate variations (e.g., -6 to +5 semitones)
                        transpositions = range(-5, 7) # 12 keys
                    else:
                        transpositions = [0]
                        neutral_midi = midi

                    for semi in transpositions:
                        if augment:
                            score = neutral_midi.transpose(semi)
                        else:
                            score = midi

                        print(f"Parsing {os.path.basename(file_path)} (Transpose {semi})...")
                        
                        try:
                            s2 = instrument.partitionByInstrument(score)
                            notes_to_parse = s2.parts[0].recurse() 
                        except:
                            notes_to_parse = score.flat.notes
                        
                        for element in notes_to_parse:
                            if isinstance(element, note.Note):
                                all_notes.append(str(element.pitch))
                            elif isinstance(element, chord.Chord):
                                # Sort chord notes for consistency
                                sorted_notes = sorted(element.normalOrder)
                                all_notes.append('.'.join(str(n) for n in sorted_notes))
                                
                except Exception as e:
                    print(f"Skipping {file_path}: {e}")
            
            self.notes = all_notes
            self._filter_notes()
        
    def _filter_notes(self):
        # 3. Dynamic Filtering
        # Instead of deleting notes < 50, we keep the top X% or count > 1
        count = Counter(self.notes)
        # Keep notes that appear at least twice to remove total outliers
        self.notes = [n for n in self.notes if count[n] > 1]
        print(f"Total notes after filtering: {len(self.notes)}")

    def prepare_sequences(self):
        pitchnames = sorted(set(self.notes))
        self.n_vocab = len(pitchnames)
        self.pitch_to_int = dict((c, i) for i, c in enumerate(pitchnames))
        self.int_to_pitch = dict((i, c) for i, c in enumerate(pitchnames))

        print(f"Vocab size: {self.n_vocab}")

        self.network_input = []
        network_output = []

        # Create sequences
        for i in range(0, len(self.notes) - self.sequence_length, 1):
            sequence_in = self.notes[i:i + self.sequence_length]
            sequence_out = self.notes[i + self.sequence_length]
            self.network_input.append([self.pitch_to_int[char] for char in sequence_in])
            network_output.append(self.pitch_to_int[sequence_out])

        n_patterns = len(self.network_input)
        if n_patterns == 0:
            raise ValueError("No sequences created! Dataset too small or sequence_length too high.")

        self.normalized_input = np.reshape(self.network_input, (n_patterns, self.sequence_length, 1))
        self.normalized_input = self.normalized_input / float(self.n_vocab)
        self.network_output = to_categorical(network_output)

    def create_model(self):
        model = Sequential()
        model.add(Input(shape=(self.sequence_length, 1)))
        
        # 4. Improved Architecture
        # Bidirectional LSTM learns context from both directions
        model.add(Bidirectional(LSTM(256, return_sequences=True)))
        model.add(BatchNormalization()) # Helps stabilize training
        model.add(Dropout(0.3))
        
        model.add(LSTM(256))
        model.add(BatchNormalization())
        model.add(Dropout(0.3))
        
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.3))
        
        model.add(Dense(self.n_vocab, activation='softmax'))
        
        opt = Adamax(learning_rate=0.005) # Slightly lower LR for stability
        model.compile(loss='categorical_crossentropy', optimizer=opt)
        self.model = model
        return model

    def save_mappings(self, path):
        with open(path, "wb") as f:
            pickle.dump({
                'pitch_to_int': self.pitch_to_int,
                'int_to_pitch': self.int_to_pitch,
                'n_vocab': self.n_vocab,
                'notes': self.notes
            }, f)

    def load_mappings(self, path):
        with open(path, "rb") as f:
            data = pickle.load(f)
            self.pitch_to_int = data['pitch_to_int']
            self.int_to_pitch = data['int_to_pitch']
            self.n_vocab = data['n_vocab']
            self.notes = data['notes']