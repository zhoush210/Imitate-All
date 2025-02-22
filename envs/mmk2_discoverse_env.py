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

    def __init__(self, path: str):
        module = importlib.import_module(path.replace("/", ".").replace(".py", ""))
        node_cls = getattr(module, "SimNode")
        cfg: MMK2Cfg = getattr(module, "cfg")
        cfg.headless = False
        self.exec_node: MMK2TaskBase = node_cls(cfg)
        self.reset_position = None
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
        obs["qpos"] = list(raw_obs["jq"])[2:]

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
        # convert mmk2 to sim
        sim_action = np.zeros(19)
        sim_action[2]=action[16] # slide
        sim_action[3]=action[14] # head_yaw
        sim_action[4]=-action[15] # head_pitch
        sim_action[5:12]=action[0:7] # left arm
        sim_action[12:19]=action[7:14] #right arm

        #fix some
        sim_action[3]=0
        sim_action[4]=1

        raw_obs, pri_obs, rew, ter, info = self.exec_node.step(sim_action)

        time.sleep(sleep_time)

        if get_obs:
            obs = collections.OrderedDict()
            sim_qpos=list(raw_obs["jq"])
            real_qpos=np.zeros(17)
            real_qpos[16]=sim_qpos[2]
            real_qpos[14]=sim_qpos[3]
            real_qpos[15]=-sim_qpos[4]
            real_qpos[0:7]=sim_qpos[5:12]
            real_qpos[7:14]=sim_qpos[12:19]
            obs["qpos"] = real_qpos#list(raw_obs["jq"])[2:]
            obs["images"] = {}
            # for id in self.exec_node.config.obs_rgb_cam_id:
            #     obs["images"][f"{id}"] = raw_obs["img"][id][:, :, ::-1]
            obs["images"][f"{0}"] = raw_obs["img"][1][::-1, ::-1, ::-1]
            obs["images"][f"{1}"] = raw_obs["img"][2][::-1, :, ::-1]
            obs["images"][f"{2}"] = raw_obs["img"][0][:, :, ::-1]
        else:
            obs = None
        return dm_env.TimeStep(
            step_type=dm_env.StepType.MID,
            reward=self.get_reward(),
            discount=None,
            observation=obs,
        )

def make_env(path):
    env = MMK2DiscoverseEnv(path)
    return env
