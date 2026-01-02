import os
from src import MusicLSTM

# Configuration
DATA_DIR = "data"
MODEL_DIR = "models"
EPOCHS = 100
BATCH_SIZE = 64

def main():
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)

    # 1. Initialize
    lstm = MusicLSTM()
    
    # 2. Load and Process Data
    print("--- Starting Data Processing ---")
    lstm.load_data(DATA_DIR, parsing=True)
    lstm.prepare_sequences()
    
    # 3. Save Mappings 
    mapping_path = os.path.join(MODEL_DIR, "mappings.pkl")
    lstm.save_mappings(mapping_path)
    print(f"Mappings saved to {mapping_path}")

    # 4. Build and Train Model
    print("--- Starting Training ---")
    model = lstm.create_model()
    
    model_path = os.path.join(MODEL_DIR, "music_model.keras")
    
    try:
        model.fit(
            lstm.normalized_input, 
            lstm.network_output, 
            epochs=EPOCHS, 
            batch_size=BATCH_SIZE
        )
        model.save(model_path)
        print(f"Training Complete. Model saved to {model_path}")
    except KeyboardInterrupt:
        print("\nTraining interrupted manually. Saving current state...")
        model.save(model_path)
        print("Model saved.")

if __name__ == "__main__":
    main()