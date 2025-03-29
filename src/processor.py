from transformers import AutoModelForCausalLM, AutoTokenizer
import h5py
import numpy as np

class LLMTIMEPreprocessor:
    def __init__(self, scale_factor=10, decimal_places=2, model_name="Qwen/Qwen2.5-0.5B-Instruct"):
        """
        Initialize LLMTIME Preprocessor.

        :param scale_factor: Factor to scale numeric values before rounding.
        :param decimal_places: Number of decimal places to round to.
        :param model_name: HuggingFace model name for tokenization.
        """
        self.scale_factor = scale_factor
        self.decimal_places = decimal_places
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def preprocess_sequence(self, sequence):
        """
        Preprocesses a single time series sequence for LLMTIME tokenization.

        :param sequence: np.array of shape (time_steps, 2) where:
                         - [:,0] = Prey population
                         - [:,1] = Predator population
        :return: Preprocessed string representation.
        """
        scaled = sequence / self.scale_factor
        rounded = np.round(scaled, self.decimal_places)

        # Convert to string format: "x1,y1; x2,y2; ..."
        formatted_sequence = ";".join([",".join(map(str, row)) for row in rounded])
        return formatted_sequence

    def tokenize_sequence(self, sequence_str):
        """
        Tokenizes a preprocessed time series string using Qwen tokenizer.

        :param sequence_str: Preprocessed LLMTIME-formatted string.
        :return: List of token IDs.
        """
        tokens = self.tokenizer(sequence_str, return_tensors="pt")["input_ids"].tolist()[0]
        return tokens



def load_and_preprocess(filename="coursework/lotka_volterra_data.h5", num_systems=100, time_steps=20, fraction=0.5):
    """
    Loads multiple time series from the dataset and preprocesses them.

    :param filename: Path to HDF5 dataset.
    :param num_systems: Number of systems to load.
    :param time_steps: Number of timesteps per system.
    :return: Tuple (train_texts, val_texts)
    """
    with h5py.File(filename, "r") as f:
        trajectories = f["trajectories"][:]

    # Initialize Preprocessor
    preprocessor = LLMTIMEPreprocessor()

    # Process all systems
    all_sequences = []
    for i in range(min(num_systems, len(trajectories))):
        sequence = trajectories[i, :time_steps, :]
        preprocessed_seq = preprocessor.preprocess_sequence(sequence)
        all_sequences.append(preprocessed_seq)

    # Split into train and validation sets (80% train, 20% val)
    split_idx = int(fraction * len(all_sequences))
    train_texts = all_sequences[:split_idx]
    val_texts = all_sequences[split_idx:]

    return train_texts, val_texts



def load_sample_data(filename="coursework/lotka_volterra_data.h5", system_id=0, time_steps=10):
    """
    Loads a sample time series from the dataset.

    :param filename: Path to HDF5 dataset.
    :param system_id: ID of the system to sample.
    :param time_steps: Number of timesteps to extract.
    :return: np.array of shape (time_steps, 2).
    """
    with h5py.File(filename, "r") as f:
        trajectories = f["trajectories"][:]
    return trajectories[system_id, :time_steps, :]

if __name__ == "__main__":
    # Load sample data
    sample_data = load_sample_data(time_steps=100)

    # Initialize preprocessor
    preprocessor = LLMTIMEPreprocessor()

    # Process and tokenize the sequence
    preprocessed_seq = preprocessor.preprocess_sequence(sample_data)
    tokenized_seq = preprocessor.tokenize_sequence(preprocessed_seq)

    # Print results
    print("\n🔹 Preprocessed Sequence:")
    print(preprocessed_seq)
    print("\n🔹 Tokenized Sequence:")
    print(tokenized_seq)


def load_sample_data2(filename="coursework/lotka_volterra_data.h5", system_id=0, time_steps=10):
    with h5py.File(filename, "r") as f:
        trajectories = f["trajectories"][:]
        time_points = f["time"][:]
        # Access a single trajectory
        system_id = 0 # First system
        # First 50 time points:
        prey = trajectories[0, :50, 0]
        predator = trajectories[0, :50, 1]
        times = time_points[:50]

    return prey, predator, times


if __name__ == "__main__":
    prey, predator, times = load_sample_data2()

    print(f"Prey Population: {prey}")
    print(f"Predator Population: {predator}")
    print(f"Time Points: {times}")