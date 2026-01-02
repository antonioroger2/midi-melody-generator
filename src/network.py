import os
import pickle
import numpy as np
from music21 import converter, instrument, note, chord
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.optimizers import Adamax
from tensorflow.keras.utils import to_categorical
from collections import Counter

class MusicLSTM:
    def __init__(self, sequence_length=40):
        self.sequence_length = sequence_length
        self.notes = []
        self.network_input = []
        self.n_vocab = 0
        self.pitch_to_int = {}
        self.int_to_pitch = {}
        self.model = None

    def load_data(self, data_dir, parsing=True):
        """
        parsing=True: Parses MIDI files (slow, for training).
        parsing=False: Just sets up structure (fast, for generation if we have mappings).
        """
        if parsing:
            print(f"Loading MIDI files from {data_dir}...")
            all_notes = []
            for file in os.listdir(data_dir):
                if file.endswith(".mid"):
                    try:
                        midi = converter.parse(os.path.join(data_dir, file))
                        print(f"Parsing {file}...")
                        try:
                            s2 = instrument.partitionByInstrument(midi)
                            notes_to_parse = s2.parts[0].recurse() 
                        except:
                            notes_to_parse = midi.flat.notes
                        
                        for element in notes_to_parse:
                            if isinstance(element, note.Note):
                                all_notes.append(str(element.pitch))
                            elif isinstance(element, chord.Chord):
                                all_notes.append('.'.join(str(n) for n in element.normalOrder))
                    except Exception as e:
                        print(f"Error parsing {file}: {e}")
            self.notes = all_notes
            self._remove_rare_notes()
        
        # If we didn't parse (generation mode), we assume notes/mappings are loaded manually

    def prepare_sequences(self):
        """Converts notes to one-hot encoded sequences for training."""
        pitchnames = sorted(set(self.notes))
        self.n_vocab = len(pitchnames)
        self.pitch_to_int = dict((c, i) for i, c in enumerate(pitchnames))
        self.int_to_pitch = dict((i, c) for i, c in enumerate(pitchnames))

        print(f"Vocab size: {self.n_vocab}")

        self.network_input = []
        network_output = []

        for i in range(0, len(self.notes) - self.sequence_length, 1):
            sequence_in = self.notes[i:i + self.sequence_length]
            sequence_out = self.notes[i + self.sequence_length]
            self.network_input.append([self.pitch_to_int[char] for char in sequence_in])
            network_output.append(self.pitch_to_int[sequence_out])

        n_patterns = len(self.network_input)
        
        # Reshape and normalize
        self.normalized_input = np.reshape(self.network_input, (n_patterns, self.sequence_length, 1))
        self.normalized_input = self.normalized_input / float(self.n_vocab)
        self.network_output = to_categorical(network_output)

    def _remove_rare_notes(self):
        count = Counter(self.notes)
        self.notes = [n for n in self.notes if count[n] >= 50]

    def create_model(self):
        model = Sequential()
        model.add(Input(shape=(self.sequence_length, 1)))
        model.add(LSTM(512, return_sequences=True)) # Recurrent layer
        model.add(Dropout(0.3))
        model.add(LSTM(256))
        model.add(Dense(256))
        model.add(Dropout(0.3))
        model.add(Dense(self.n_vocab, activation='softmax'))
        
        opt = Adamax(learning_rate=0.01)
        model.compile(loss='categorical_crossentropy', optimizer=opt)
        self.model = model
        return model

    def save_mappings(self, path):
        with open(path, "wb") as f:
            pickle.dump({
                'pitch_to_int': self.pitch_to_int,
                'int_to_pitch': self.int_to_pitch,
                'n_vocab': self.n_vocab,
                'notes': self.notes # Save notes to use as seeds later
            }, f)

    def load_mappings(self, path):
        with open(path, "rb") as f:
            data = pickle.load(f)
            self.pitch_to_int = data['pitch_to_int']
            self.int_to_pitch = data['int_to_pitch']
            self.n_vocab = data['n_vocab']
            self.notes = data['notes']