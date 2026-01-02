import os
from src.network import MusicLSTM
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau


DATA_DIR = "data" 
MODEL_DIR = "models"
EPOCHS = 200 
BATCH_SIZE = 64
SEQ_LENGTH = 100 

def main():
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)

    
    lstm = MusicLSTM(sequence_length=SEQ_LENGTH)
    
    
    print("--- Starting Data Processing ---")
    
    lstm.load_data(DATA_DIR, parsing=True, augment=True) 
    lstm.prepare_sequences()
    
    
    mapping_path = os.path.join(MODEL_DIR, "mappings.pkl")
    lstm.save_mappings(mapping_path)
    print(f"Mappings saved to {mapping_path}")

    
    print("--- Starting Training ---")
    model = lstm.create_model()
    
    
    checkpoint_path = os.path.join(MODEL_DIR, "best_model.keras")
    
    
    callbacks = [
        
        ModelCheckpoint(
            checkpoint_path,
            monitor='loss',
            verbose=1,
            save_best_only=True,
            mode='min'
        ),
        
        EarlyStopping(
            monitor='loss',
            patience=15,
            verbose=1,
            restore_best_weights=True
        ),
        
        ReduceLROnPlateau(
            monitor='loss',
            factor=0.5,
            patience=5,
            verbose=1,
            min_lr=0.00001
        )
    ]
    
    try:
        model.fit(
            lstm.normalized_input, 
            lstm.network_output, 
            epochs=EPOCHS, 
            batch_size=BATCH_SIZE,
            callbacks=callbacks
        )
        print(f"Training Complete. Best model available at {checkpoint_path}")
    except KeyboardInterrupt:
        print("\nTraining interrupted manually. Best model already saved via Checkpoint.")

if __name__ == "__main__":
    main()