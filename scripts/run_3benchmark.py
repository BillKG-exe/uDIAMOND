import subprocess
import os
from pathlib import Path

# Assuming the current working directory is the root of the uDIAMOND project
# Get the original list of games from src/utils.py
# In a real scenario, you would import ATARI_100K_GAMES from src.utils
# For this example, we'll hardcode a subset or the full list for demonstration.
# Full list is available in src/utils.py
ATARI_GAMES_NAMES = [
    "Alien"]

#"Amidar", "Freeway","Jamesbond", "KungFuMaster", "MsPacman", "Pong","RoadRunner"

# Define the number of runs per game
NUM_RUNS_PER_GAME = 1

# Define the device to use (e.g., 0 for GPU 0, or 'cpu')
DEVICE = 0 # Or 'cpu', or a list like [0, 1] for multiple GPUs if supported by your setup

# Base command to run the main training script
MAIN_SCRIPT = Path("src/main.py")

if not MAIN_SCRIPT.exists():
    print(f"Error: Could not find {MAIN_SCRIPT}. Make sure you run this script from the root of the uDIAMOND project.")
    exit(1)

print(f"Starting benchmark with {NUM_RUNS_PER_GAME} runs per game.")

for game_name in ATARI_GAMES_NAMES:
    game_id = f"{game_name}NoFrameskip-v4"
    print(f"\n--- Running benchmark for game: {game_name} ({game_id}) ---")

    for i in range(NUM_RUNS_PER_GAME):
        seed = i + 1 # Use sequential seeds for each run
        print(f"\n--- Run {i+1}/{NUM_RUNS_PER_GAME} for {game_name} (Seed: {seed}) ---")

        # Construct the command to run src/main.py
        # We use common.seed to explicitly set the random seed for reproducibility
        # and common.devices to specify the GPU.
        # We also set hydra.output_subdir=null and hydra.run.dir=. to prevent Hydra
        # from creating a new output directory for each run, so all runs for a game
        # can be within the same top-level output folder if you want to manage them that way.
        # However, to ensure distinct run folders for each seed, Hydra's default behavior
        # (creating timestamped directories) is usually preferred.
        # So, we don't set hydra.run.dir=. here to keep separate run folders.
        command = [
            "python",
            str(MAIN_SCRIPT),
            f"env.train.id={game_id}",
            f"common.seed={seed}",
            f"common.devices={DEVICE}"
        ]

        try:
            # Execute the command
            process = subprocess.run(command, check=True, text=True, capture_output=True)
            print("STDOUT:")
            print(process.stdout)
            if process.stderr:
                print("STDERR:")
                print(process.stderr)
            print(f"Successfully completed run {i+1} for {game_name}.")
        except subprocess.CalledProcessError as e:
            print(f"Error running training for {game_name} (Seed: {seed}):")
            print("STDOUT:")
            print(e.stdout)
            print("STDERR:")
            print(e.stderr)
            print("Continuing to next run/game...")
        except FileNotFoundError:
            print(f"Error: 'python' command not found. Make sure Python is installed and in your PATH.")
            exit(1)

print("\nBenchmark automation complete.")