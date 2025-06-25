import argparse
import pathlib
import os
import imageio
import numpy as np
import warnings
import sys
import importlib
import gymnasium as gym
import cv2 

if 'MUJOCO_GL' not in os.environ:
    os.environ['MUJOCO_GL'] = 'egl'

script_dir = pathlib.Path(__file__).parent.resolve()
sys.path.append(str(script_dir / 'dreamerv3'))

from ruamel.yaml import YAML
import embodied
from embodied.agents import dreamerv3
from embodied import wrappers
from embodied.envs import from_gymnasium
import humanoid_bench


def wrap_env(env, config):
    """Applies the stack of embodied wrappers to the environment."""
    args = config.wrapper
    for name, space in env.act_space.items():
        if name == "reset": continue
        elif not space.discrete:
            env = wrappers.NormalizeAction(env, name)
            if args.discretize:
                env = wrappers.DiscretizeAction(env, name, args.discretize)
    env = wrappers.ExpandScalars(env)
    if args.length:
        env = wrappers.TimeLimit(env, args.length, args.reset)
    if args.checks:
        env = wrappers.CheckSpaces(env)
    for name, space in env.act_space.items():
        if not space.discrete:
            env = wrappers.ClipAction(env, name)
    return env

def generate_video(logdir: str, resolution=(640, 480), steps: int = 1000):
    logdir = pathlib.Path(logdir).expanduser()
    print(f"Loading agent from: {logdir}")

    # Config Creation
    config_dict = dreamerv3.configs['defaults'].copy()
    config = embodied.Config(config_dict)
    config = config.update(dreamerv3.Agent.configs['humanoid_benchmark'])
    config_path = logdir / 'config.yaml'
    yaml_loader = YAML(typ='safe', pure=True)
    saved_config_dict = yaml_loader.load(config_path.read_text())
    config = config.update(saved_config_dict)
    
    # Environment Creation
    print("Creating environment...")
    gym_task_name = config.task.replace('humanoid_', '')
    
    env = gym.make(
        gym_task_name, 
        render_mode='rgb_array', 
        width=resolution[0], 
        height=resolution[1]
    )
    
    env = from_gymnasium.FromGymnasium(env, obs_key='vector')
    env = wrap_env(env, config)
    agent = dreamerv3.Agent(env.obs_space, env.act_space, config)
    
    print("Loading checkpoint...")
    latest_ckpt_path = max(list(logdir.glob('*.ckpt')), key=os.path.getmtime)
    full_checkpoint_state = np.load(str(latest_ckpt_path), allow_pickle=True)
    agent_state_dict = full_checkpoint_state['agent']
    agent.load(agent_state_dict)

    # Manual Rollout Loop
    print("Starting manual rollout...")
    frames = []
    
    action = {'action': env.act_space['action'].sample(), 'reset': True}
    obs = env.step(action)
    
    agent_state = agent.init_policy(batch_size=1) 
    
    for _ in range(steps):
        obs_batch = {k: np.expand_dims(v, 0) for k, v in obs.items()}
        action, agent_state = agent.policy(obs_batch, agent_state, mode='eval')
        action_unpacked = {k: np.array(v[0]) for k, v in action.items()}
        
        action_unpacked['reset'] = False
        
        obs = env.step(action_unpacked)
        
        frame = env.render()
        # Resize as a final guarantee, though it should already be the correct size.
        frame = cv2.resize(frame, resolution, interpolation=cv2.INTER_AREA)
        frames.append(frame)

        if obs['is_last']:
            print("Episode finished.")
            agent_state = agent.init_policy(batch_size=1)

    env.close()

    # Save the video
    outfile = logdir / f"evaluation_{gym_task_name}_{resolution[0]}x{resolution[1]}.mp4"
    imageio.mimsave(outfile, frames, fps=30)
    print(f"\nSUCCESS! High-resolution video saved to {outfile.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--logdir", type=str, required=True)
    parser.add_argument("--width", type=int, default=720)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--steps", type=int, default=1000)
    args = parser.parse_args()
    generate_video(logdir=args.logdir, resolution=(args.width, args.height), steps=args.steps)