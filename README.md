# MIDI Melody Generator

An LSTM-based Recurrent Neural Network (RNN) built with TensorFlow/Keras to generate classical music melodies in MIDI format. This project uses a Bidirectional LSTM architecture to learn from existing MIDI files and compose original melodies with "humanized" dynamics.

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/midi-melody-generator.git](https://github.com/your-username/midi-melody-generator.git)
    cd midi-melody-generator
    ```
2.  **Install dependencies:**
    You will need Python 3.x and the following libraries:
    ```bash
    pip install tensorflow numpy music21
    ```

## Project Structure

```text
midi-melody-generator/
├── data/               # Training .mid/.midi files goes here
├── models/             # Stores trained models (.keras) and mappings (.pkl)
├── output/             # Stores generated music files
├── src/
│   ├── __init__.py
│   └── network.py      # Contains the MusicLSTM class and model architecture
├── generate.py         # Script to generate new music
├── train.py            # Script to train the model
└── README.md

```

## Usage

### 3. Data Preparation

Create a folder named `data` in the root directory and add your MIDI files. The script supports nested folders.

### 4. Training the Model

Run the training script to process the data and train the neural network.

```bash
python train.py

```


### 5. Generating Music

Once a model is trained, use the generation script to compose music.

```bash
python generate.py

```

## Model Architecture

The model is defined in `src/network.py`:

1. **Input Layer**: Sequence length of 100 notes.
2. **Bidirectional LSTM (256 units)**: Learns past and future context.
3. **LSTM (256 units)**: Refines features.
4. **Dense Layers**: Fully connected layers with Dropout to prevent overfitting.
5. **Softmax Output**: Predicts the probability of the next note.
