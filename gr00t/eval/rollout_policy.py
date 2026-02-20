import argparse
import json
from collections import defaultdict
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
import time
from typing import Any
import uuid

import numpy as np

from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.eval.sim.env_utils import get_embodiment_tag_from_env_name
from gr00t.eval.sim.wrapper.multistep_wrapper import MultiStepWrapper
from gr00t.policy import BasePolicy
import gymnasium as gym
from tqdm import tqdm


def _json_serializable(obj):
    """Recursively convert ndarrays and numpy scalars in obj to Python types for JSON."""
    if isinstance(obj, np.ndarray):
        return _json_serializable(obj.tolist())
    if isinstance(obj, (np.floating, np.integer, np.bool_)):
        return float(obj) if isinstance(obj, np.floating) else int(obj) if isinstance(obj, np.integer) else bool(obj)
    if isinstance(obj, dict):
        return {k: _json_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_serializable(x) for x in obj]
    return obj


def _to_list_4f(arr) -> list:
    """Convert array or list to list of 4-decimal strings. Handles nested/object arrays (e.g. from vector env)."""
    try:
        a = np.asarray(arr, dtype=float).ravel()
    except (ValueError, TypeError):
        flat = np.asarray(arr, dtype=object).ravel()
        parts = [np.asarray(x, dtype=float).ravel() for x in flat]
        a = np.concatenate(parts) if parts else np.array([])
    return [f"{x:.4f}" for x in a.tolist()]


def _to_list_4f_float(arr) -> list:
    """Convert array or list to list of floats with 4 decimal places (for JSON numbers in dump)."""
    try:
        a = np.asarray(arr, dtype=float).ravel()
    except (ValueError, TypeError):
        flat = np.asarray(arr, dtype=object).ravel()
        parts = [np.asarray(x, dtype=float).ravel() for x in flat]
        a = np.concatenate(parts) if parts else np.array([])
    return [round(float(x), 4) for x in a.tolist()]


def _observation_sim_for_dump(obs: dict, env_idx: int = 0, use_floats: bool = True) -> dict:
    """Build observation_sim dict for one env for JSONL dump (includes floating_base_pose, floating_base_vel).
    If use_floats, output numbers with 4 decimal places; else 4-decimal strings."""
    to_list = _to_list_4f_float if use_floats else _to_list_4f
    out = {}
    for key in ("q", "dq", "floating_base_pose", "floating_base_vel"):
        if key not in obs:
            continue
        v = obs[key]
        if hasattr(v, "shape") and len(v.shape) > 1:
            v = v[env_idx] if v.shape[0] > env_idx else v
        out[key] = to_list(v)
    if "annotation.human.task_description" in obs:
        v = obs["annotation.human.task_description"]
        if hasattr(v, "shape") and len(v.shape) > 0:
            v = v[env_idx] if len(v) > env_idx else v
        out["annotation.human.task_description"] = str(v)
    for key in ("ego_view_image", "tpp_view_image", "video.ego_view", "video.tpp_view"):
        if key not in obs:
            continue
        v = obs[key]
        if hasattr(v, "shape"):
            out[key] = {"shape": list(v.shape), "dtype": str(v.dtype)}
        else:
            out[key] = v
    return out


def _gr00t_action_for_dump(actions: dict, env_idx: int = 0, first_timestep_only: bool = True, use_floats: bool = True) -> dict:
    """Build gr00t action dict for dump (list of numbers per key, 4 decimal places).
    If first_timestep_only, take only the next-frame prediction (first timestep), not the full T-step horizon.
    If use_floats, output numbers; else 4-decimal strings."""
    to_list = _to_list_4f_float if use_floats else _to_list_4f
    out = {}
    for k, v in actions.items():
        if not k.startswith("action."):
            continue
        a = np.asarray(v)
        if hasattr(a, "shape") and len(a.shape) > 1 and a.shape[0] > env_idx:
            a = a[env_idx]
        if first_timestep_only and hasattr(a, "shape") and a.ndim >= 2 and a.shape[0] > 1:
            a = a[0]
        elif first_timestep_only and hasattr(a, "shape") and a.ndim == 1:
            if k == "action.base_height_command" and a.size > 1:
                a = a[:1]
            elif k == "action.navigate_command" and a.size > 3:
                a = a[:3]
            elif "left_arm" in k and a.size > 7:
                a = a[:7]
            elif "right_arm" in k and a.size > 7:
                a = a[:7]
            elif "left_hand" in k and a.size > 7:
                a = a[:7]
            elif "right_hand" in k and a.size > 7:
                a = a[:7]
            elif "waist" in k and a.size > 3:
                a = a[:3]
        out[k] = to_list(a)
    return out


@dataclass
class VideoConfig:
    """Configuration for video recording settings.

    Attributes:
        video_dir: Directory to save videos (if None, no videos are saved)
        steps_per_render: Number of steps between each call to env.render() while recording
            during rollout
        fps: Frames per second for the output video
        codec: Video codec to use for compression
        input_pix_fmt: Input pixel format
        crf: Constant Rate Factor for video compression (lower = better quality)
        thread_type: Threading strategy for video encoding
        thread_count: Number of threads to use for encoding
    """

    video_dir: str | None = None
    steps_per_render: int = 2
    max_episode_steps: int = 720
    fps: int = 20
    codec: str = "h264"
    input_pix_fmt: str = "rgb24"
    crf: int = 22
    thread_type: str = "FRAME"
    thread_count: int = 1
    overlay_text: bool = True
    n_action_steps: int = 8


@dataclass
class MultiStepConfig:
    """Configuration for multi-step environment settings.

    Attributes:
        video_delta_indices: Indices of video observations to stack
        state_delta_indices: Indices of state observations to stack
        n_action_steps: Number of action steps to execute
        max_episode_steps: Maximum number of steps per episode
    """

    video_delta_indices: np.ndarray = field(default_factory=lambda: np.array([0]))
    state_delta_indices: np.ndarray = field(default_factory=lambda: np.array([0]))
    n_action_steps: int = 16
    max_episode_steps: int = 720
    terminate_on_success: bool = False


@dataclass
class WrapperConfigs:
    """Container for various environment wrapper configurations.

    Attributes:
        video: Configuration for video recording
        multistep: Configuration for multi-step processing
    """

    video: VideoConfig = field(default_factory=VideoConfig)
    multistep: MultiStepConfig = field(default_factory=MultiStepConfig)


def get_robocasa_env_fn(
    env_name: str,
):
    def env_fn():
        import os

        import robocasa  # noqa: F401
        from robocasa.utils.gym_utils import GrootRoboCasaEnv  # noqa: F401
        import robosuite  # noqa: F401

        os.environ["MUJOCO_GL"] = "egl"
        return gym.make(env_name, enable_render=True)

    return env_fn


def get_groot_locomanip_env_fn(
    env_name: str,
):
    def env_fn():
        from gr00t_wbc.control.envs.robocasa.sync_env import SyncEnv  # noqa: F401
        from gr00t_wbc.control.main.teleop.configs.configs import BaseConfig
        from gr00t_wbc.control.utils.n1_utils import WholeBodyControlWrapper
        import robocasa  # noqa: F401

        gym_env = gym.make(
            env_name,
            onscreen=False,
            offscreen=True,
            enable_waist=True,
            randomize_cameras=False,
            camera_names=[
                "robot0_oak_egoview",
                "robot0_rs_tppview",
            ],
        )
        wbc_config = BaseConfig(wbc_version="gear_wbc", enable_waist=True).to_dict()
        gym_env = WholeBodyControlWrapper(gym_env, wbc_config)
        return gym_env

    return env_fn


def get_simpler_env_fn(
    env_name: str,
):
    def env_fn():
        from gr00t.eval.sim.SimplerEnv.simpler_env import register_simpler_envs

        register_simpler_envs()
        return gym.make(env_name)

    return env_fn


def get_libero_env_fn(
    env_name: str,
):
    def env_fn():
        from gr00t.eval.sim.LIBERO.libero_env import register_libero_envs

        register_libero_envs()
        return gym.make(env_name)

    return env_fn


def get_behavior_env_fn(
    env_name: str,
    env_idx: int,
    total_n_envs: int,
):
    def env_fn():
        from gr00t.eval.sim.BEHAVIOR.behavior_env import register_behavior_envs

        register_behavior_envs()
        return gym.make(env_name, env_idx=env_idx, total_n_envs=total_n_envs)

    return env_fn


def get_gym_env(env_name: str, env_idx: int, total_n_envs: int):
    """Create Ray environment factory function without wrappers."""

    env_embodiment = get_embodiment_tag_from_env_name(env_name)

    if env_embodiment in (
        EmbodimentTag.GR1,
        EmbodimentTag.ROBOCASA_PANDA_OMRON,
    ):
        env_fn = get_robocasa_env_fn(env_name)

    elif env_embodiment in (EmbodimentTag.UNITREE_G1,):
        env_fn = get_groot_locomanip_env_fn(env_name)

    elif env_embodiment in (EmbodimentTag.OXE_GOOGLE, EmbodimentTag.OXE_WIDOWX):
        env_fn = get_simpler_env_fn(env_name)

    elif env_embodiment in (EmbodimentTag.LIBERO_PANDA,):
        env_fn = get_libero_env_fn(env_name)

    elif env_embodiment in (EmbodimentTag.BEHAVIOR_R1_PRO,):
        env_fn = get_behavior_env_fn(env_name, env_idx, total_n_envs)
    else:
        raise ValueError(f"Invalid environment name: {env_name}")

    return env_fn()


def create_eval_env(
    env_name: str, env_idx: int, total_n_envs: int, wrapper_configs: WrapperConfigs
) -> gym.Env:
    """Create a single evaluation environment with wrappers.

    Args:
        env_name: Name of the gymnasium environment to use
        idx: Environment index (used to determine video recording)
        wrapper_configs: Configuration for environment wrappers
    Returns:
        Wrapped gymnasium environment
    """

    env = get_gym_env(env_name, env_idx, total_n_envs)
    if wrapper_configs.video.video_dir is not None:
        from gr00t.eval.sim.wrapper.video_recording_wrapper import (
            VideoRecorder,
            VideoRecordingWrapper,
        )

        video_recorder = VideoRecorder.create_h264(
            fps=wrapper_configs.video.fps,
            codec=wrapper_configs.video.codec,
            input_pix_fmt=wrapper_configs.video.input_pix_fmt,
            crf=wrapper_configs.video.crf,
            thread_type=wrapper_configs.video.thread_type,
            thread_count=wrapper_configs.video.thread_count,
        )
        env = VideoRecordingWrapper(
            env,
            video_recorder,
            video_dir=Path(wrapper_configs.video.video_dir),
            steps_per_render=wrapper_configs.video.steps_per_render,
            max_episode_steps=wrapper_configs.video.max_episode_steps,
            overlay_text=wrapper_configs.video.overlay_text,
        )

    env = MultiStepWrapper(
        env,
        video_delta_indices=wrapper_configs.multistep.video_delta_indices,
        state_delta_indices=wrapper_configs.multistep.state_delta_indices,
        n_action_steps=wrapper_configs.multistep.n_action_steps,
        max_episode_steps=wrapper_configs.multistep.max_episode_steps,
        terminate_on_success=wrapper_configs.multistep.terminate_on_success,
    )
    return env


def run_rollout_gymnasium_policy(
    env_name: str,
    policy: BasePolicy,
    wrapper_configs: WrapperConfigs,
    n_episodes: int = 10,
    n_envs: int = 1,
    dump_io_path: str | Path | None = None,
) -> Any:
    """Run policy rollouts in parallel environments.

    Args:
        env_name: Name of the gymnasium environment to use
        policy_fn: Function that creates a policy instance
        n_episodes: Number of episodes to run
        n_envs: Number of parallel environments
        wrapper_configs: Configuration for environment wrappers
        dump_io_path: If set (and env is gr00tlocomanip), write one JSONL line per step with observation_sim (q, dq, floating_base_pose, floating_base_vel), gr00t, wbc_goal, wbc, wbc_action.
    Returns:
        Collection results from running the episodes
    """
    start_time = time.time()
    n_episodes = max(n_episodes, n_envs)
    print(f"Running collecting {n_episodes} episodes for {env_name} with {n_envs} vec envs")

    dump_io_path = Path(dump_io_path) if dump_io_path else None
    dump_file = None
    dump_step_count = 0
    dump_episode_id = 0
    if dump_io_path and env_name.startswith("gr00tlocomanip"):
        dump_file = open(dump_io_path, "w", buffering=1)
        print(f"Dump IO: writing to {dump_io_path}")

    env_fns = [
        partial(
            create_eval_env,
            env_idx=idx,
            env_name=env_name,
            total_n_envs=n_envs,
            wrapper_configs=wrapper_configs,
        )
        for idx in range(n_envs)
    ]

    if n_envs == 1:
        env = gym.vector.SyncVectorEnv(env_fns)
    else:
        env = gym.vector.AsyncVectorEnv(
            env_fns,
            shared_memory=False,
            context="spawn",
        )

    # Storage for results
    episode_lengths = []
    current_rewards = [0] * n_envs
    current_lengths = [0] * n_envs
    completed_episodes = 0
    current_successes = [False] * n_envs
    episode_successes = []
    episode_infos = defaultdict(list)

    # Initial reset
    observations, _ = env.reset()
    policy.reset()
    i = 0

    pbar = tqdm(total=n_episodes, desc="Episodes")
    while completed_episodes < n_episodes:
        actions, _ = policy.get_action(observations)
        next_obs, rewards, terminations, truncations, env_infos = env.step(actions)

        if dump_file is not None and n_envs >= 1:
            env_idx = 0

            def _take_env(v, idx):
                if isinstance(v, np.ndarray) and v.ndim >= 1 and v.shape[0] > idx:
                    return v[idx]
                if isinstance(v, (list, tuple)) and len(v) > idx:
                    return v[idx]
                return v

            def _dict_to_4f(d):
                if d is None:
                    return None
                out = {}
                for k, val in d.items():
                    out[k] = _to_list_4f_float(val)
                return out

            def _first_step_value(v):
                """Extract single next-frame value from nested multistep data.

                Iteratively peels list/numpy wrappers until it finds a dict
                (for wbc_goal/wbc) or a simple numeric array.
                """
                if v is None:
                    return None
                for _ in range(10):
                    if isinstance(v, dict):
                        return v
                    if isinstance(v, (list, tuple)):
                        if len(v) == 0:
                            return v
                        if isinstance(v[0], dict):
                            return v[0]
                        v = v[0]
                        continue
                    if isinstance(v, np.ndarray):
                        if v.dtype == object:
                            if v.ndim == 0:
                                v = v.item()
                                continue
                            if v.size > 0:
                                v = v.flat[0]
                                continue
                            return v
                        if v.ndim >= 2 and v.shape[0] > 1:
                            return v[0]
                        return v
                    return v
                return v

            obs_sim = _observation_sim_for_dump(observations, env_idx)
            gr00t = _gr00t_action_for_dump(actions, env_idx, first_timestep_only=True)
            rec = {
                "step": f"{dump_step_count:6d}",
                "episode": dump_episode_id,
                "observation_sim": obs_sim,
                "gr00t": gr00t,
            }
            if "wbc_goal" in env_infos:
                v = _take_env(env_infos["wbc_goal"], env_idx)
                v = _first_step_value(v) if not isinstance(v, dict) else v
                rec["wbc_goal"] = _dict_to_4f(v) if isinstance(v, dict) else v
            if "wbc" in env_infos:
                v = _take_env(env_infos["wbc"], env_idx)
                v = _first_step_value(v) if not isinstance(v, dict) else v
                if isinstance(v, dict):
                    rec["wbc"] = {k: _to_list_4f_float(np.asarray(val).ravel()) for k, val in v.items()}
                else:
                    rec["wbc"] = v
            if "wbc_action" in env_infos:
                v = env_infos["wbc_action"]
                wbc_q = _take_env(v, env_idx)
                wbc_q = np.asarray(wbc_q)
                if wbc_q.dtype == object:
                    wbc_q = np.concatenate(
                        [np.asarray(x, dtype=float).ravel() for x in wbc_q.ravel()]
                    )
                if wbc_q.ndim >= 2 and wbc_q.shape[0] > 1:
                    wbc_q = wbc_q[0]
                wbc_q = wbc_q.ravel()[:43]
                wbc_action_4f = _to_list_4f_float(wbc_q)
                rec["wbc_action"] = wbc_action_4f
                rec["sent_to_simulator"] = {"robot": wbc_action_4f}
            def _round_floats_4(obj):
                """Recursively round all floats to 4 decimal places so dump has no long decimals."""
                if isinstance(obj, dict):
                    return {k: _round_floats_4(val) for k, val in obj.items()}
                if isinstance(obj, (list, tuple)):
                    return [_round_floats_4(x) for x in obj]
                if isinstance(obj, (float, np.floating)):
                    return round(float(obj), 4)
                if isinstance(obj, (int, np.integer, np.bool_)):
                    return int(obj) if isinstance(obj, (np.integer, np.bool_)) else obj
                return obj

            def _json_default(o):
                if isinstance(o, np.ndarray):
                    return o.tolist()
                if isinstance(o, (np.floating, np.integer, np.bool_)):
                    return float(o) if isinstance(o, np.floating) else (int(o) if isinstance(o, np.integer) else bool(o))
                raise TypeError(f"Object of type {type(o).__name__} is not JSON serializable")

            rec = _round_floats_4(rec)
            dump_file.write(
                json.dumps(_json_serializable(rec), separators=(",", ":"), default=_json_default) + "\n"
            )
            dump_step_count += 1
        # NOTE (FY): Currently we don't properly handle policy reset. For now, our policy are stateless,
        # but in the future if we need policy to be stateful, we need to detect env reset and call policy.reset()
        i += 1
        # Update episode tracking
        for env_idx in range(n_envs):
            if "success" in env_infos:
                env_success = env_infos["success"][env_idx]
                if isinstance(env_success, list):
                    env_success = np.any(env_success)
                elif isinstance(env_success, np.ndarray):
                    env_success = np.any(env_success)
                elif isinstance(env_success, bool):
                    env_success = env_success
                elif isinstance(env_success, int):
                    env_success = bool(env_success)
                else:
                    raise ValueError(f"Unknown success dtype: {type(env_success)}")
                current_successes[env_idx] |= bool(env_success)
            else:
                current_successes[env_idx] = False

            if "final_info" in env_infos and env_infos["final_info"][env_idx] is not None:
                env_success = env_infos["final_info"][env_idx]["success"]
                if isinstance(env_success, list):
                    env_success = any(env_success)
                elif isinstance(env_success, np.ndarray):
                    env_success = np.any(env_success)
                elif isinstance(env_success, bool):
                    env_success = env_success
                elif isinstance(env_success, int):
                    env_success = bool(env_success)
                else:
                    raise ValueError(f"Unknown success dtype: {type(env_success)}")
                current_successes[env_idx] |= bool(env_success)
            current_rewards[env_idx] += rewards[env_idx]
            current_lengths[env_idx] += 1

            # If episode ended, store results
            if terminations[env_idx] or truncations[env_idx]:
                if env_idx == 0:
                    dump_episode_id += 1
                    dump_step_count = 0
                if "final_info" in env_infos:
                    current_successes[env_idx] |= any(env_infos["final_info"][env_idx]["success"])
                if "task_progress" in env_infos:
                    episode_infos["task_progress"].append(env_infos["task_progress"][env_idx][-1])
                if "q_score" in env_infos:
                    episode_infos["q_score"].append(np.max(env_infos["q_score"][env_idx]))
                if "valid" in env_infos:
                    episode_infos["valid"].append(all(env_infos["valid"][env_idx]))
                # Accumulate results
                episode_lengths.append(current_lengths[env_idx])
                episode_successes.append(current_successes[env_idx])
                # Reset trackers for this environment.
                current_successes[env_idx] = False
                # only update completed_episodes if valid
                if "valid" in episode_infos:
                    if episode_infos["valid"][-1]:
                        completed_episodes += 1
                        pbar.update(1)
                else:
                    # envs don't return valid
                    completed_episodes += 1
                    pbar.update(1)
                current_rewards[env_idx] = 0
                current_lengths[env_idx] = 0
        observations = next_obs
    pbar.close()

    if dump_file is not None:
        dump_file.close()
        print(f"Dump IO: closed {dump_io_path}")

    env.reset()
    env.close()
    print(f"Collecting {n_episodes} episodes took {time.time() - start_time} seconds")

    assert len(episode_successes) >= n_episodes, (
        f"Expected at least {n_episodes} episodes, got {len(episode_successes)}"
    )

    episode_infos = dict(episode_infos)  # Convert defaultdict to dict
    for key, value in episode_infos.items():
        assert len(value) == len(episode_successes), (
            f"Length of {key} is not equal to the number of episodes"
        )

    # process valid results
    if "valid" in episode_infos:
        valids = episode_infos["valid"]
        valid_idxs = np.where(valids)[0]
        episode_successes = [episode_successes[i] for i in valid_idxs]
        episode_infos = {k: [v[i] for i in valid_idxs] for k, v in episode_infos.items()}

    return env_name, episode_successes, episode_infos


def create_gr00t_sim_policy(
    model_path: str,
    embodiment_tag: EmbodimentTag,
    policy_client_host: str = "",
    policy_client_port: int | None = None,
) -> BasePolicy:
    from gr00t.policy.gr00t_policy import Gr00tPolicy, Gr00tSimPolicyWrapper

    if policy_client_host and policy_client_port:
        from gr00t.policy.server_client import PolicyClient

        policy = PolicyClient(host=policy_client_host, port=policy_client_port)
    else:
        policy = Gr00tSimPolicyWrapper(
            Gr00tPolicy(
                embodiment_tag=embodiment_tag,
                model_path=model_path,
                device=0,
            )
        )
    return policy


def run_gr00t_sim_policy(
    env_name: str,
    n_episodes: int,
    max_episode_steps: int,
    model_path: str = "",
    policy_client_host: str = "",
    policy_client_port: int | None = None,
    n_envs: int = 8,
    n_action_steps: int = 8,
    dump_io_path: str | Path | None = None,
):
    embodiment_tag = get_embodiment_tag_from_env_name(env_name)

    if dump_io_path is not None:
        video_dir = str(Path(dump_io_path).resolve().parent)
    elif model_path:
        video_dir = (
            f"/tmp/sim_eval_videos_{model_path.split('/')[-3]}_ac{n_action_steps}_{uuid.uuid4()}"
        )
    else:
        video_dir = f"/tmp/sim_eval_videos_{env_name}_ac{n_action_steps}_{uuid.uuid4()}"
    if env_name.startswith("sim_behavior_r1_pro"):
        # BEHAVIOR sim will crash if decord is imported in video_utils.py
        video_dir = None
    wrapper_configs = WrapperConfigs(
        video=VideoConfig(
            video_dir=video_dir,
            max_episode_steps=max_episode_steps,
        ),
        multistep=MultiStepConfig(
            n_action_steps=n_action_steps,
            max_episode_steps=max_episode_steps,
            terminate_on_success=True,
        ),
    )

    policy = create_gr00t_sim_policy(
        model_path, embodiment_tag, policy_client_host, policy_client_port
    )

    results = run_rollout_gymnasium_policy(
        env_name=env_name,
        policy=policy,
        wrapper_configs=wrapper_configs,
        n_episodes=n_episodes,
        n_envs=n_envs,
        dump_io_path=dump_io_path,
    )
    print("Video saved to: ", wrapper_configs.video.video_dir)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_episode_steps", type=int, default=504)
    parser.add_argument("--n_episodes", type=int, default=50)
    parser.add_argument(
        "--model_path",
        type=str,
        default="",
    )
    parser.add_argument("--policy_client_host", type=str, default="")
    parser.add_argument("--policy_client_port", type=int, default=None)
    parser.add_argument(
        "--env_name",
        type=str,
        default="gr1_unified/PosttrainPnPNovelFromPlateToBowlSplitA_GR1ArmsAndWaistFourierHands_Env",
    )
    parser.add_argument("--n_envs", type=int, default=8)
    parser.add_argument("--n_action_steps", type=int, default=8)
    parser.add_argument(
        "--dump_io_path",
        type=str,
        default=None,
        help="If set, write one JSONL line per step (gr00tlocomanip only) with observation_sim (q, dq, floating_base_pose, floating_base_vel), gr00t, wbc_goal, wbc, wbc_action.",
    )

    args = parser.parse_args()

    # validate policy configuration
    assert (args.model_path and not (args.policy_client_host or args.policy_client_port)) or (
        not args.model_path and args.policy_client_host and args.policy_client_port is not None
    ), (
        "Invalid policy configuration: You must provide EITHER model_path OR (policy_client_host & policy_client_port), not both.\n"
        "If all 3 arguments are provided, explicitly choose one:\n"
        '  - To use policy client: set --policy_client_host and --policy_client_port, and set --model_path ""\n'
        '  - To use model path: set --model_path, and set --policy_client_host "" (and leave --policy_client_port unset)'
    )

    results = run_gr00t_sim_policy(
        env_name=args.env_name,
        n_episodes=args.n_episodes,
        max_episode_steps=args.max_episode_steps,
        model_path=args.model_path,
        policy_client_host=args.policy_client_host,
        policy_client_port=args.policy_client_port,
        n_envs=args.n_envs,
        n_action_steps=args.n_action_steps,
        dump_io_path=args.dump_io_path,
    )
    print("results: ", results)
    print("success rate: ", np.mean(results[1]))
