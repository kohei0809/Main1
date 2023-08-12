import os
import random

import numpy as np
from gym import spaces

from typing import cast

from matplotlib import pyplot as plt

from PIL import Image

from habitat_sim.utils.common import d3_40_colors_rgb

from habitat_baselines.config.default import get_config  
from habitat.sims.habitat_simulator.habitat_simulator import HabitatSim
from habitat_baselines.common.env_utils import construct_envs
from habitat_baselines.common.environments import get_env_class
from habitat.core.env_point import Env
#from habitat.core.env import Env
from habitat.utils.visualizations import maps
from utils.log_manager import LogManager
from utils.log_writer import LogWriter

def display_sample(
    rgb_obs, semantic_obs=np.array([]), depth_obs=np.array([]), depth_writer=None, env=None, config=None, opt=-1
):
    rgb_img = Image.fromarray(rgb_obs, mode="RGB")
    
    arr = [rgb_obs]
    titles = ["rgb"]
    if semantic_obs.size != 0:
        semantic_img = Image.new(
            "P", (semantic_obs.shape[1], semantic_obs.shape[0])
        )

        semantic_img.putpalette(d3_40_colors_rgb.flatten())
        semantic_img.putdata((semantic_obs.flatten() % 40).astype(np.uint8))
        semantic_img = semantic_img.convert("RGBA")
        arr.append(semantic_img)
        titles.append("semantic")
        
    if depth_obs.size != 0:
        for i in range(depth_obs.shape[0]):
            for j in range(depth_obs.shape[1]):
                depth_writer.write(str(depth_obs[i][j]))
            depth_writer.writeLine()
            
        print(depth_obs[0][0])
        print(depth_obs[128][128])
        print(np.amax(depth_obs))
        print(np.amin(depth_obs))
        
        depth_obs = (depth_obs - config.SIMULATOR.DEPTH_SENSOR.MIN_DEPTH) / (
                    config.SIMULATOR.DEPTH_SENSOR.MAX_DEPTH - config.SIMULATOR.DEPTH_SENSOR.MIN_DEPTH
                )
        print("SIZE")
        print(depth_obs.shape)
        depth_img = Image.fromarray(
            (depth_obs / 10 * 255).astype(np.uint8), mode="L"
        )
        arr.append(depth_img)
        titles.append("depth")
        
    plt.figure(figsize=(12, 8))
    for i, data in enumerate(arr):
        ax = plt.subplot(1, 3, i + 1)
        ax.axis("off")
        ax.set_title(titles[i])
        plt.imshow(data)
    
    path = "./figures/check_depth/fig.png"
    if opt !=-1:
        path = "./figures/check_depth/fig" + str(opt) + ".png"
    plt.savefig(path)
    plt.show(block=False)
                
def check_semantic(env, semantic_obs, sem_writer):
    # obtain mapping from instance id to semantic label id
    scene = env.sim._sim.semantic_scene
    instance_id_to_label_id = {int(obj.id.split("_")[-1]): obj.category.index() for obj in scene.objects}
    mapping = np.array([ instance_id_to_label_id[i] for i in range(len(instance_id_to_label_id)) ])
    
    obs = np.take(mapping, semantic_obs)
    print(obs[255][0])
    print(obs[255][1])
    
    for i in range(obs.shape[0]):
        for j in range(obs.shape[1]):
            sem_writer.write(str(obs[i][j]))
        sem_writer.writeLine()
    
    
if __name__ == '__main__':
    exp_config = "./habitat/config/test.yaml"
    opts = None
    config = get_config(exp_config, opts)
    
    dataset_path = "figures/test3.json.gz"
        
    config.defrost()
    config.DATASET.DATA_PATH = dataset_path
    config.TASK_CONFIG.SIMULATOR.DEPTH_SENSOR.NORMALIZE_DEPTH = False
    config.TASK_CONFIG.SIMULATOR.DEPTH_SENSOR.MIN_DEPTH = 0.0
    config.TASK_CONFIG.SIMULATOR.DEPTH_SENSOR.MAX_DEPTH = 5.0
    config.TASK_CONFIG.SIMULATOR.AGENT_0.HEIGHT = 1.5
    config.TASK_CONFIG.SIMULATOR.SEMANTIC_SENSOR.HEIGHT = 256
    config.TASK_CONFIG.SIMULATOR.SEMANTIC_SENSOR.WIDTH = 256
    config.TASK_CONFIG.SIMULATOR.AGENT_0.SENSORS = ["RGB_SENSOR", "DEPTH_SENSOR", "SEMANTIC_SENSOR"]
    config.TASK_CONFIG.TASK.MEASUREMENTS = ['DISTANCE_TO_CURR_GOAL', 'DISTANCE_TO_MULTI_GOAL', 'SUB_SUCCESS', 'SUCCESS', 'EPISODE_LENGTH', 'MSPL', 'PERCENTAGE_SUCCESS', 'RATIO', 'PSPL', 'RAW_METRICS', 'TOP_DOWN_MAP']
    config.TASK_CONFIG.TRAINER_NAME = "oracle"
    config.TASK_CONFIG.DATASET.DATA_PATH = dataset_path
    config.freeze()
    
    #print(config)
    #print("########################")
    
    log_manager = LogManager()
    log_manager.setLogDirectory("check")
    depth_writer = log_manager.createLogWriter("depth")
    
    with Env(config=config.TASK_CONFIG) as env:
        observation = env.reset()
        info = env.get_metrics()
        display_sample(observation['rgb'], observation['semantic'], np.squeeze(observation['depth']), depth_writer, env, config.TASK_CONFIG, opt=-1)
        #scene = env.sim._sim.semantic_scene
        #check_semantic(env, observation['semantic'], sem2_writer)
    