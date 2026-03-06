"""Evaluate pi0.5 on LIBERO with LIBERO-X perturbations.

This is a modified version of examples/libero/main.py that adds support for
LIBERO-X perturbation configs (lighting, texture, view, robot_state, layout, etc.).

Setup (without Docker):
    Terminal 1 - Model server (uses the main openpi uv env):
        uv run scripts/serve_policy.py --env LIBERO

    Terminal 2 - Evaluation client (uses a separate Python 3.11 env for LIBERO-X):
        uv venv --python 3.11 examples/libero/.venv_x
        source examples/libero/.venv_x/bin/activate
        uv pip install -e third_party/LIBERO-X
        uv pip install -e packages/openpi-client
        uv pip install imageio imageio-ffmpeg tyro tqdm

        # Run evaluation with perturbations:
        python examples/libero/main_x.py \\
            --libero-x third_party/LIBERO-X/perturbations/full_perturbation.yaml

        # Run without perturbations (equivalent to original main.py):
        python examples/libero/main_x.py
"""

import collections
import dataclasses
import functools
import logging
import math
import os
import pathlib

import imageio
from libero.libero import benchmark

# PyTorch 2.6+ defaults torch.load to weights_only=True, but LIBERO's
# init state files are numpy-pickled and require weights_only=False.
import torch
torch.load = functools.partial(torch.load, weights_only=False)
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv
import numpy as np
from openpi_client import image_tools
from openpi_client import websocket_client_policy as _websocket_client_policy
import tqdm
import tyro

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256  # resolution used to render training data


@dataclasses.dataclass
class Args:
    #################################################################################################################
    # Model server parameters
    #################################################################################################################
    host: str = "0.0.0.0"
    port: int = 8000
    resize_size: int = 224
    replan_steps: int = 5

    #################################################################################################################
    # LIBERO environment-specific parameters
    #################################################################################################################
    task_suite_name: str = (
        "libero_spatial"  # Task suite. Options: libero_spatial, libero_object, libero_goal, libero_10, libero_90
    )
    num_steps_wait: int = 10  # Number of steps to wait for objects to stabilize i n sim
    num_trials_per_task: int = 50  # Number of rollouts per task

    #################################################################################################################
    # LIBERO-X perturbation
    #################################################################################################################
    libero_x: str | None = None  # Path to LIBERO-X perturbation config YAML file

    #################################################################################################################
    # Utils
    #################################################################################################################
    video_out_path: str = "data/libero/videos"  # Path to save videos

    seed: int = 7  # Random Seed (for reproducibility)


def _setup_libero_x(config_path: str):
    """Load LIBERO-X config and apply LIBERO-level monkey-patches.

    Only applies patches that target LIBERO/robosuite internals (scene, view,
    object_scale).  LeRobot-level patches (env_create, reset) are handled
    manually in the evaluation loop below.
    """
    from libero_x.config import load_perturbation_config
    import libero_x.patches as lx_patches

    cfg = load_perturbation_config(config_path)

    # Set the active config so patches can read it at runtime
    lx_patches._active_config = cfg

    if cfg.is_empty:
        logging.info("LIBERO-X: no perturbations configured")
        return cfg

    logging.info("LIBERO-X: applying perturbations: %s", cfg.summary())

    # Apply LIBERO-level patches (these do NOT depend on LeRobot)
    if cfg.has_scene_override:
        lx_patches._patch_scene()
    if cfg.view:
        lx_patches._patch_view()
    if cfg.object_scale:
        lx_patches._patch_object_scale()

    # Register distractor objects if needed
    if cfg.layout == "distractor" or cfg.object_swap:
        import libero_x.distractors  # noqa: F401

    return cfg


def _get_libero_env(task, resolution, seed, cfg=None):
    """Initializes and returns the LIBERO environment, along with the task description.

    When cfg is provided, applies LIBERO-X perturbations:
    - layout: substitutes BDDL file for perturbed/distractor variants
    - object_swap: rewrites BDDL with substituted object type names
    - language_swap: applies text replacements to task description
    - view: passes camera view override to the environment
    """
    task_description = task.language

    if cfg is not None:
        import libero_x.patches as lx_patches

        # BDDL file: layout override → fallback to original
        bddl_file = None
        if cfg.layout:
            bddl_file = lx_patches._layout_file(
                task.name, cfg.layout, lx_patches._BDDL_ROOT, "bddl"
            )

        if bddl_file is None:
            bddl_file = str(
                pathlib.Path(get_libero_path("bddl_files"))
                / task.problem_folder
                / task.bddl_file
            )

        # Object swap: rewrite BDDL with substituted type names
        if cfg.object_swap and os.path.isfile(bddl_file):
            bddl_file = lx_patches._swap_bddl_objects(bddl_file, cfg.object_swap)

        # Language swap
        task_description = cfg.apply_language_swap(task_description)

        env_args = {
            "bddl_file_name": bddl_file,
            "camera_heights": resolution,
            "camera_widths": resolution,
        }

        # View override (consumed by the _patch_view monkey-patch)
        if cfg.view:
            env_args["view"] = cfg.view

    else:
        task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
        env_args = {"bddl_file_name": task_bddl_file, "camera_heights": resolution, "camera_widths": resolution}

    env = OffScreenRenderEnv(**env_args)
    env.seed(seed)  # IMPORTANT: seed seems to affect object positions even when using fixed initial state
    return env, task_description


def _load_init_states(task_suite, task_id, task_name, cfg):
    """Load initial states, handling layout overrides and object_swap invalidation.

    Returns:
        init_states: array of initial states, or None if fresh resets should be used.
    """
    init_states = task_suite.get_task_init_states(task_id)

    if cfg is None:
        return init_states

    # Object swap changes MuJoCo model structure → saved init states are invalid
    if cfg.object_swap:
        logging.info("LIBERO-X: object_swap active, using fresh resets (no init states)")
        return None

    # Layout override → load variant init states
    if cfg.layout:
        import libero_x.patches as lx_patches

        init_path = lx_patches._layout_file(
            task_name, cfg.layout, lx_patches._INIT_ROOT, "pruned_init"
        )
        if init_path:
            import torch

            init_states = torch.load(init_path, weights_only=False)
            logging.info("LIBERO-X: using layout init states from %s", init_path)

    return init_states


def _apply_post_reset_perturbations(env, cfg):
    """Apply robot_state and recolor_robot perturbations after env reset.

    Args:
        env: OffScreenRenderEnv instance.
        cfg: PerturbationConfig or None.

    Returns:
        True if observations need to be re-rendered.
    """
    if cfg is None:
        return False

    rerender = False

    if cfg.robot_state:
        from libero_x.robot_utils import apply_robot_perturbation

        rerender = apply_robot_perturbation(env, cfg.robot_state)

    if cfg.recolor_robot:
        import libero_x.patches as lx_patches

        lx_patches._recolor_robot_geoms(env)
        rerender = True

    return rerender


def _refresh_obs(env):
    """Re-render observations from current sim state."""
    env._post_process()
    env._update_observables(force=True)
    return env.env._get_observations()


def eval_libero(args: Args) -> None:
    # Set random seed
    np.random.seed(args.seed)

    # Load LIBERO-X perturbation config (if provided)
    cfg = None
    if args.libero_x:
        cfg = _setup_libero_x(args.libero_x)

    # Initialize LIBERO task suite
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks
    logging.info(f"Task suite: {args.task_suite_name}")

    pathlib.Path(args.video_out_path).mkdir(parents=True, exist_ok=True)

    if args.task_suite_name == "libero_spatial":
        max_steps = 220  # longest training demo has 193 steps
    elif args.task_suite_name == "libero_object":
        max_steps = 280  # longest training demo has 254 steps
    elif args.task_suite_name == "libero_goal":
        max_steps = 300  # longest training demo has 270 steps
    elif args.task_suite_name == "libero_10":
        max_steps = 520  # longest training demo has 505 steps
    elif args.task_suite_name == "libero_90":
        max_steps = 400  # longest training demo has 373 steps
    else:
        raise ValueError(f"Unknown task suite: {args.task_suite_name}")

    client = _websocket_client_policy.WebsocketClientPolicy(args.host, args.port)

    # Start evaluation
    total_episodes, total_successes = 0, 0
    for task_id in tqdm.tqdm(range(num_tasks_in_suite)):
        # Get task
        task = task_suite.get_task(task_id)

        # Get initial states (may be overridden by LIBERO-X layout config)
        initial_states = _load_init_states(task_suite, task_id, task.name, cfg)
        use_fresh_resets = initial_states is None

        # Initialize LIBERO environment and task description
        env, task_description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed, cfg)

        # Start episodes
        task_episodes, task_successes = 0, 0
        for episode_idx in tqdm.tqdm(range(args.num_trials_per_task)):
            logging.info(f"\nTask: {task_description}")

            # Reset environment
            if use_fresh_resets:
                # Object swap mode: use stability-retry reset
                import libero_x.patches as lx_patches

                obs = lx_patches._reset_until_stable(env, args.seed + episode_idx)
            else:
                env.reset()
                obs = env.set_init_state(initial_states[episode_idx])

            action_plan = collections.deque()

            # Apply post-reset perturbations (robot_state, recolor_robot)
            if _apply_post_reset_perturbations(env, cfg):
                obs = _refresh_obs(env)

            # Setup
            t = 0
            replay_images = []

            logging.info(f"Starting episode {task_episodes+1}...")
            while t < max_steps + args.num_steps_wait:
                try:
                    # IMPORTANT: Do nothing for the first few timesteps because the simulator drops objects
                    # and we need to wait for them to fall
                    if t < args.num_steps_wait:
                        obs, reward, done, info = env.step(LIBERO_DUMMY_ACTION)
                        t += 1
                        continue

                    # Get preprocessed image
                    # IMPORTANT: rotate 180 degrees to match train preprocessing
                    img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                    wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
                    img = image_tools.convert_to_uint8(
                        image_tools.resize_with_pad(img, args.resize_size, args.resize_size)
                    )
                    wrist_img = image_tools.convert_to_uint8(
                        image_tools.resize_with_pad(wrist_img, args.resize_size, args.resize_size)
                    )

                    # Save preprocessed image for replay video
                    replay_images.append(img)

                    if not action_plan:
                        # Finished executing previous action chunk -- compute new chunk
                        # Prepare observations dict
                        element = {
                            "observation/image": img,
                            "observation/wrist_image": wrist_img,
                            "observation/state": np.concatenate(
                                (
                                    obs["robot0_eef_pos"],
                                    _quat2axisangle(obs["robot0_eef_quat"]),
                                    obs["robot0_gripper_qpos"],
                                )
                            ),
                            "prompt": str(task_description),
                        }

                        # Query model to get action
                        action_chunk = client.infer(element)["actions"]
                        assert (
                            len(action_chunk) >= args.replan_steps
                        ), f"We want to replan every {args.replan_steps} steps, but policy only predicts {len(action_chunk)} steps."
                        action_plan.extend(action_chunk[: args.replan_steps])

                    action = action_plan.popleft()

                    # Execute action in environment
                    obs, reward, done, info = env.step(action.tolist())
                    if done:
                        task_successes += 1
                        total_successes += 1
                        break
                    t += 1

                except Exception as e:
                    logging.error(f"Caught exception: {e}")
                    break

            task_episodes += 1
            total_episodes += 1

            # Save a replay video of the episode
            suffix = "success" if done else "failure"
            task_segment = task_description.replace(" ", "_")
            imageio.mimwrite(
                pathlib.Path(args.video_out_path) / f"rollout_{task_segment}_{suffix}.mp4",
                [np.asarray(x) for x in replay_images],
                fps=10,
            )

            # Log current results
            logging.info(f"Success: {done}")
            logging.info(f"# episodes completed so far: {total_episodes}")
            logging.info(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)")

        # Log final results
        logging.info(f"Current task success rate: {float(task_successes) / float(task_episodes)}")
        logging.info(f"Current total success rate: {float(total_successes) / float(total_episodes)}")

    logging.info(f"Total success rate: {float(total_successes) / float(total_episodes)}")
    logging.info(f"Total episodes: {total_episodes}")


def _quat2axisangle(quat):
    """
    Copied from robosuite: https://github.com/ARISE-Initiative/robosuite/blob/eafb81f54ffc104f905ee48a16bb15f059176ad3/robosuite/utils/transform_utils.py#L490C1-L512C55
    """
    # clip quaternion
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        # This is (close to) a zero degree rotation, immediately return
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    eval_libero(tyro.cli(Args))
