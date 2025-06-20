import argparse
import pathlib
import os
import imageio

import cv2
import gymnasium as gym

import humanoid_bench
from .env import ROBOTS, TASKS

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="HumanoidBench environment test")
    parser.add_argument("--env", help="e.g. h1-walk-v0")
    parser.add_argument("--steps", type=int, default=-1, help="number of environment steps")
    parser.add_argument("--save_video", default=None, help="path to save the generated video")
    parser.add_argument("--keyframe", default=None)
    parser.add_argument("--policy_path", default=None)
    parser.add_argument("--mean_path", default=None)
    parser.add_argument("--var_path", default=None)
    parser.add_argument("--policy_type", default=None)
    parser.add_argument("--blocked_hands", default="False")
    parser.add_argument("--small_obs", default="False")
    parser.add_argument("--obs_wrapper", default="False")
    parser.add_argument("--sensors", default="")
    parser.add_argument("--render_mode", default="rgb_array")  # "human" or "rgb_array".
    parser.add_argument("--width",  type=int, default=640,  help="render width")
    parser.add_argument("--height", type=int, default=480,  help="render height")

    # NOTE: to get (nicer) 'human' rendering to work, you need to fix the compatibility issue between mujoco>3.0 and gymnasium: https://github.com/Farama-Foundation/Gymnasium/issues/749
    args = parser.parse_args()

    kwargs = vars(args).copy()
    kwargs.pop("env")
    kwargs.pop("render_mode")
    if kwargs["keyframe"] is None:
        kwargs.pop("keyframe")
    print(f"arguments: {kwargs}")

    # Test offscreen rendering
    print(f"Test offscreen mode...")
    env = gym.make(args.env, render_mode="rgb_array", **kwargs)
    ob, _ = env.reset()
    if isinstance(ob, dict):
        print(f"ob_space = {env.observation_space}")
        print(f"ob = ")
        for k, v in ob.items():
            print(f"  {k}: {v.shape}")
    else:
        print(f"ob_space = {env.observation_space}, ob = {ob.shape}")
    print(f"ac_space = {env.action_space.shape}")

    img = env.render()
    # Resize the image afterwards since can't seem to be able to do better
    img = cv2.resize(img, (args.width, args.height), interpolation=cv2.INTER_CUBIC)
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    cv2.imwrite("test_env_img.png", rgb_img)

    # Test online rendering with interactive viewer
    if args.steps == -1:
        print(f"Test onscreen mode...")
        env = gym.make(args.env, render_mode=args.render_mode, **kwargs)
        ob, _ = env.reset()
        if isinstance(ob, dict):
            print(f"ob_space = {env.observation_space}")
            print(f"ob = ")
            for k, v in ob.items():
                print(f"  {k}: {v.shape}")
                assert (
                    v.shape == env.observation_space.spaces[k].shape
                ), f"{v.shape} != {env.observation_space.spaces[k].shape}"
            assert ob.keys() == env.observation_space.spaces.keys()
        else:
            print(f"ob_space = {env.observation_space}, ob = {ob.shape}")
            assert env.observation_space.shape == ob.shape
        print(f"ac_space = {env.action_space.shape}")
        # print("observation:", ob)
        env.render()
        ret = 0
        while True:
            action = env.action_space.sample()
            ob, rew, terminated, truncated, info = env.step(action)
            img = env.render()
            img = cv2.resize(img, (args.width, args.height), interpolation=cv2.INTER_CUBIC)
            ret += rew

            if args.render_mode == "rgb_array":
                cv2.imshow("test_env", img[:, :, ::-1])
                cv2.waitKey(1)

            if terminated or truncated:
                ret = 0
                env.reset()
        env.close()

    else:
        print("Starting rollout...")
        env = gym.make(args.env, render_mode="rgb_array", **kwargs)
        ob, _ = env.reset()
        frames = []                    # collected only when we save a video
        ret = 0.0

        for t in range(args.steps):
            action = env.action_space.sample()
            ob, rew, terminated, truncated, info = env.step(action)
            img = env.render()
            img = cv2.resize(img, (args.width, args.height), interpolation=cv2.INTER_CUBIC)
            ret += rew

            if args.save_video:
                frames.append(img)
            elif args.render_mode == "rgb_array":
                cv2.imshow("test_env", img[:, :, ::-1])
                cv2.waitKey(1)

            if terminated or truncated:
                ret = 0.0
                env.reset()

        env.close()

        if args.save_video is not None:
            outdir = pathlib.Path(args.save_video).expanduser().resolve()
            outdir.mkdir(parents=True, exist_ok=True)
            outfile = outdir / f"{args.env}_random.mp4"
            imageio.mimsave(outfile, frames, fps=30)
            print(f"Saved video to {outfile.resolve()}")
