from ray import train, tune
from ray.rllib.algorithms.ppo import PPOConfig

from ray.rllib.utils.metrics import (
    ENV_RUNNER_RESULTS,
    EPISODE_RETURN_MEAN,
)
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.test_utils import (
    add_rllib_example_script_args,
    run_rllib_example_script_experiment,
)
from ray.rllib.utils.numpy import convert_to_numpy, softmax
from ray.rllib.connectors.learner.frame_stacking import FrameStackingLearner

from ray.rllib.core.columns import Columns
from pathlib import Path
import gymnasium as gym
import numpy as np
from ray.rllib.core.rl_module import RLModule
from ray.tune import CheckpointConfig

torch, _ = try_import_torch()


def train_ray_sample(config):
    # Create a Tuner instance to manage the trials.
    tuner = tune.Tuner(
        config.algo_class,
        param_space=config,
        # Specify a stopping criterion. Note that the criterion has to match one of the
        # pretty printed result metrics from the results returned previously by
        # ``.train()``. Also note that -1100 is not a good episode return for
        # Pendulum-v1, we are using it here to shorten the experiment time.
        run_config=train.RunConfig(
            # stop={"env_runners/episode_return_mean": -2000.0},
            stop={"training_iteration": 300},
            checkpoint_config=CheckpointConfig(
                num_to_keep=5,
                checkpoint_frequency=10,
                checkpoint_score_attribute="evaluation/env_runners/episode_return_mean",
                checkpoint_at_end=True
            ),
            verbose=1,  # ログ出力を詳細に
        ),
    )
    # Run the Tuner and capture the results.
    results = tuner.fit()

    # Get the last checkpoint from the above training run.
    best_result = results.get_best_result(
        metric=f"{ENV_RUNNER_RESULTS}/{EPISODE_RETURN_MEAN}", mode="max"
    )

    # Get the checkpoint corresponding to the best result
    best_checkpoint = best_result.checkpoint

    return best_checkpoint


def inference(best_checkpoint, args):
    # Create only the neural network (RLModule) from our algorithm checkpoint.
    # See here (https://docs.ray.io/en/master/rllib/checkpoints.html)
    # to learn more about checkpointing and the specific "path" used.
    # best_checkpoint_path = "/root/ray_results/PPO_2025-07-22_13-56-08/PPO_BipedalWalker-v3_9f234_00000_0_2025-07-22_13-56-10/checkpoint_000029"
    if best_checkpoint:
        best_checkpoint_path = best_checkpoint.path
    else:
        best_checkpoint_path = args.checkpoint_file
    print()
    print("#" * 100)
    print("best_checkpoint.path:", best_checkpoint_path)
    print()
    print("If you want to use this checkpoint for inference, please execute the script with the ")
    print()
    print("            python3 scripts/ray_examples/ppo_continuous.py -f <path_to_checkpoint>")
    print("#" * 100)
    print()
    rl_module = RLModule.from_checkpoint(
        Path(best_checkpoint_path)
        / "learner_group"
        / "learner"
        / "rl_module"
        / "default_policy"
    )

    # Create the RL environment to test against (same as was used for training earlier).
    env = gym.make(args.env, render_mode="human")

    episode_return = 0.0
    done = False

    # Reset the env to get the initial observation.
    obs, info = env.reset()

    while not done:
        # Uncomment this line to render the env.
        # env.render()

        # Compute the next action from a batch (B=1) of observations.
        obs_batch = torch.from_numpy(obs).unsqueeze(0)  # add batch B=1 dimension
        model_outputs = rl_module.forward_inference({"obs": obs_batch})

        # Extract the action distribution parameters from the output and dissolve batch dim.
        action_dist_params = model_outputs["action_dist_inputs"][0].numpy()

        # We have continuous actions -> take the mean (max likelihood).
        greedy_action = np.clip(
            action_dist_params[0:4],  # 0=mean, 1=log(stddev), [0:1]=use mean, but keep shape=(1,)
            a_min=env.action_space.low[0],
            a_max=env.action_space.high[0],
        )
        # For discrete actions, you should take the argmax over the logits:
        # greedy_action = np.argmax(action_dist_params)

        # Send the action to the environment for the next step.
        obs, reward, terminated, truncated, info = env.step(greedy_action)

        # Perform env-loop bookkeeping.
        episode_return += reward
        done = terminated or truncated

    print(f"Reached episode return of {episode_return}.")

def main(args, config):
    # Train the RL algorithm and get the best checkpoint.
    if args.checkpoint_file:
        best_checkpoint = None
    else:
        best_checkpoint = train_ray_sample(config)

    # Perform inference using the best checkpoint.
    inference(best_checkpoint, args)

    print("Done with training and inference.")

if __name__ == "__main__":
    parser = add_rllib_example_script_args(default_reward=200.0)
    parser.add_argument(
        "--explore-during-inference",
        action="store_true",
        help="Whether the trained policy should use exploration during action "
        "inference.",
    )
    parser.add_argument(
        "--num-episodes-during-inference",
        type=int,
        default=10,
        help="Number of episodes to do inference over (after restoring from a checkpoint).",
    )
    parser.set_defaults(
        # Make sure that - by default - we produce checkpoints during training.
        checkpoint_freq=1,
        checkpoint_at_end=True,
        # Use "BipedalWalker-v3" by default.
        env="BipedalWalker-v3",
    )
    parser.add_argument(
        "-f",
        "--checkpoint-file",
        type=str,
        default="",
        help="Path to a checkpoint file to use for inference. ",
    )
    args = parser.parse_args()

    config = (
        PPOConfig()
        .environment(args.env)
        # Specify a simple tune hyperparameter sweep.
        .env_runners(
            # Following the paper.
            num_env_runners=8
            # rollout_fragment_length=64,
        )
        .learners(
            # Let's start with a small number of learner workers and
            # add later a tune grid search for these resources.
            num_learners=1,
            num_gpus_per_learner=1,
        )
        # TODO (simon): Adjust to new model_config_dict.
        .training(
            # Following the paper.
            train_batch_size_per_learner=4000,
            minibatch_size=128,
            lambda_=0.95,
            kl_coeff=0.5,
            clip_param=0.1,
            vf_clip_param=10.0,
            entropy_coeff=0.01,
            num_epochs=10,
            lr=0.00015 * 1,
            grad_clip=100.0,
            grad_clip_by="global_norm",
            model={
                "fcnet_hiddens": [64, 64],
                "fcnet_activation": "tanh",
                "vf_share_layers": True,
            },
        )
        .reporting(
            metrics_num_episodes_for_smoothing=5,
            min_sample_timesteps_per_iteration=1000,
        )
        .evaluation(
            evaluation_duration="auto",
            evaluation_interval=1,
            evaluation_num_env_runners=1,
            evaluation_parallel_to_training=True,
            evaluation_config={
                "explore": True,
            },
        )
    )

    main(args, config)
