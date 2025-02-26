import time
import numpy as np
import collections
import dm_env
import importlib
from discoverse.envs.mmk2_base import MMK2Cfg
from discoverse.task_base import MMK2TaskBase

class MMK2DiscoverseEnv(object):
    """
    Mujoco environment for mmk2
    path: path to the script containing the SimNode class and config (mainly for data collection)
    """

    def __init__(self, path: str, joint_num: int):
        module = importlib.import_module(path.replace("/", ".").replace(".py", ""))
        node_cls = getattr(module, "SimNode")
        cfg: MMK2Cfg = getattr(module, "cfg")
        cfg.headless = False
        self.exec_node: MMK2TaskBase = node_cls(cfg)
        self.reset_position = None
        self.joint_num = joint_num
        print("MujocoEnv initialized")

    def set_reset_position(self, reset_position):
        self.reset_position = reset_position
        print("Resetting to the given position: ", self.reset_position)

    def get_reward(self):
        return 0

    def reset(self, fake=False, sleep_time=0):
        self.exec_node.domain_randomization()
        raw_obs = self.exec_node.reset()

        time.sleep(sleep_time)
        
        obs = collections.OrderedDict()
        raw_qpos = list(raw_obs["jq"])
        if self.joint_num == 19:
            obs["qpos"]=raw_qpos
        elif self.joint_num == 17:
            obs["qpos"] = raw_qpos[2:]
        else:
            raise ValueError(f"Wrong joint_num:{self.joint_num}")

        obs["images"] = {}
        for id in self.exec_node.config.obs_rgb_cam_id:
            obs["images"][f"{id}"] = raw_obs["img"][id][:, :, ::-1]
        return dm_env.TimeStep(
            step_type=dm_env.StepType.FIRST,
            reward=self.get_reward(),
            discount=None,
            observation=obs,
        )

    def step(
        self,
        action,
        get_obs=True,
        sleep_time=0,
    ):
        if self.joint_num==19:
            raw_obs, pri_obs, rew, ter, info = self.exec_node.step(action)
        elif self.joint_num==17:
            sim_action=np.zeros(19)
            sim_action[2:]=action
            raw_obs, pri_obs, rew, ter, info = self.exec_node.step(sim_action)
        else:
            raise ValueError(f"Wrong joint_num:{len(self.joint_num)}")
        time.sleep(sleep_time)

        if get_obs:
            obs = collections.OrderedDict()
            raw_qpos = list(raw_obs["jq"])
            if self.joint_num == 19:
                obs["qpos"]=raw_qpos
            elif self.joint_num == 17:
                obs["qpos"] = raw_qpos[2:]
            else:
                raise ValueError(f"Wrong joint_num:{len(self.joint_num)}")
            
            obs["images"] = {}
            for id in self.exec_node.config.obs_rgb_cam_id:
                obs["images"][f"{id}"] = raw_obs["img"][id][:, :, ::-1]
        else:
            obs = None
        return dm_env.TimeStep(
            step_type=dm_env.StepType.MID,
            reward=self.get_reward(),
            discount=None,
            observation=obs,
        )

def make_env(path, joint_num):
    env = MMK2DiscoverseEnv(path, joint_num)
    return env
