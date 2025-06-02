import argparse
from pathlib import Path
import numpy as np
import torch
import shutil # For cleaning up temporary directories

# Import necessary components from the uDIAMOND project
from omegaconf import OmegaConf, DictConfig
from src.agent import Agent
from src.envs import make_atari_env
from src.coroutines.collector import make_collector, NumToCollect
from src.utils import get_path_agent_ckpt, ATARI_100K_GAMES # get_path_agent_ckpt for latest checkpoint

# Register resolver for OmegaConf, if needed in your config files
OmegaConf.register_new_resolver("eval", eval)

def main():
    parser = argparse.ArgumentParser(description="Evaluate a uDIAMOND trained model.")
    parser.add_argument("--run_path", type=str, required=True,
                        help="Path to the trained model's run folder (e.g., 'outputs/YYYY-MM-DD/hh-mm-ss').")
    parser.add_argument("--episodes", type=int, default=100,
                        help="Number of episodes to run for evaluation.")
    parser.add_argument("--device", type=str, default="cuda:0",
                        help="Device to use for evaluation (e.g., 'cuda:0', 'cpu').")
    
    args = parser.parse_args()

    run_folder = Path(args.run_path)
    if not run_folder.is_dir():
        print(f"Error: Run folder not found at '{run_folder}'. Please provide a valid path.")
        return

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load the configuration used during the training run
    # The configuration file is copied into the run folder during training.
    cfg_path_in_run_folder = run_folder / "config" / "trainer.yaml"
    if not cfg_path_in_run_folder.is_file():
        print(f"Error: Config file '{cfg_path_in_run_folder}' not found within the run folder.")
        print("Make sure this is a valid uDIAMOND training run directory.")
        return

    print(f"Loading configuration from: {cfg_path_in_run_folder}")
    cfg: DictConfig = OmegaConf.load(cfg_path_in_run_folder)
    OmegaConf.resolve(cfg) # Resolve any interpolated values (${eval:..}, ${..train.id}, etc.)

    # Extract game ID from the loaded configuration
    game_id_full = cfg.env.train.id #
    game_name = game_id_full.replace("NoFrameskip-v4", "")
    print(f"Detected game from run config: {game_name} ({game_id_full})")

    # 2. Get the path to the latest agent checkpoint
    checkpoints_dir = run_folder / "checkpoints"
    path_to_ckpt = get_path_agent_ckpt(path_ckpt_dir=checkpoints_dir, epoch=-1) # -1 gets the last checkpoint
    
    if not path_to_ckpt.is_file():
        print(f"Error: No agent checkpoint found at '{path_to_ckpt}'.")
        print("Please ensure the model was trained successfully and checkpoints exist.")
        return

    print(f"Loading model checkpoint from: {path_to_ckpt}")

    # 3. Initialize environment
    test_env = make_atari_env(
        num_envs=1, # Typically evaluate with a single environment for consistent results
        device=device,
        **cfg.env.test # Uses the 'test' part of the env config from the loaded trainer.yaml
    )
    num_actions = int(test_env.num_actions)
    
    # Ensure num_actions is correctly set in the agent config (might be missing in base config)
    if 'num_actions' not in cfg.agent:
        cfg.agent.num_actions = num_actions
    else: # If it exists, ensure it matches
        if cfg.agent.num_actions != num_actions:
            print(f"Warning: Configured agent num_actions ({cfg.agent.num_actions}) does not match env num_actions ({num_actions}). Adjusting config...")
            cfg.agent.num_actions = num_actions


    # 4. Instantiate Agent and load trained weights
    agent = Agent(cfg.agent).to(device) # Agent initialization based on run's config
    agent.load(path_to_ckpt) # Load trained weights
    agent.eval() # Set agent to evaluation mode (e.g., disable dropout, batch norm updates)

    # 5. Set up a dummy dataset for the collector (not used for data storage here, but required by make_collector)
    # The collector requires a Dataset instance. We use an in-memory dataset that won't save to disk.
    temp_dataset_dir = Path("./temp_eval_dataset_local")
    temp_dataset_dir.mkdir(exist_ok=True)
    from src.data.dataset import Dataset
    eval_dataset = Dataset(temp_dataset_dir, name="eval_temp", cache_in_ram=True, save_on_disk=False)

    # 6. Create a collector for evaluation
    # Set epsilon to 0.0 for deterministic policy evaluation
    # reset_every_collect=True ensures a clean start for each episode in the collector
    collector = make_collector(
        test_env,
        agent.actor_critic, # Collector uses actor_critic for policy
        eval_dataset,
        epsilon=0.0,
        reset_every_collect=True,
        verbose=True # Show progress bar
    )

    # 7. Run evaluation
    print(f"\n--- Running evaluation for {game_name} for {args.episodes} episodes ---")
    
    # Send the number of episodes to collect
    logs = collector.send(NumToCollect(episodes=args.episodes)) # Collect episodes

    # 8. Process and print results
    returns = [d["return"] for d in logs if "return" in d] # Extract returns from logs
    if returns:
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        print(f"\nEvaluation Results for {game_name} ({args.episodes} episodes):")
        print(f"  Mean Return: {mean_return:.2f}")
        print(f"  Std Return: {std_return:.2f}")
    else:
        print("No episode returns recorded. Something might have gone wrong during collection.")

    # Clean up temporary dataset directory
    if temp_dataset_dir.exists():
        shutil.rmtree(temp_dataset_dir)
        print(f"Cleaned up temporary dataset directory: {temp_dataset_dir}")

if __name__ == "__main__":
    main()