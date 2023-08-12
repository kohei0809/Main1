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

def display_sample(
    rgb_obs, semantic_obs=np.array([]), depth_obs=np.array([]), config=None, opt=-1
):
    rgb_img = Image.fromarray(rgb_obs, mode="RGB")
    print("rgb:" + str(rgb_obs.shape))
    
    arr = [rgb_obs]
    titles = ["rgb"]
    if semantic_obs.size != 0:
        semantic_img = Image.new(
            "P", (semantic_obs.shape[1], semantic_obs.shape[0])
        )

        semantic_img.putpalette(d3_40_colors_rgb.flatten())
        semantic_img.putdata((semantic_obs.flatten() % 40).astype(np.uint8))
        semantic_img = semantic_img.convert("RGBA")
        #semantic_img = semantic_img.resize((256, 256))
        print("semantic:" + str(semantic_img.size))
        arr.append(semantic_img)
        titles.append("semantic")
        
    if depth_obs.size != 0:
        depth_obs = (depth_obs - config.SIMULATOR.DEPTH_SENSOR.MIN_DEPTH) / (
                    config.SIMULATOR.DEPTH_SENSOR.MAX_DEPTH - config.SIMULATOR.DEPTH_SENSOR.MIN_DEPTH
                )
        depth_img = Image.fromarray(
            (depth_obs / 10 * 255).astype(np.uint8), mode="L"
        )
        print("depth:" + str(depth_img.size))
        arr.append(depth_img)
        titles.append("depth")
        
    plt.figure(figsize=(12, 8))
    for i, data in enumerate(arr):
        ax = plt.subplot(1, 3, i + 1)
        ax.axis("off")
        ax.set_title(titles[i])
        plt.imshow(data)
    
    path = "./figures/fig.png"
    if opt !=-1:
        path = "./figures/fig" + str(opt) + ".png"
    plt.savefig(path)
    plt.show(block=False)
    
def example_get_topdown_map(config, env, opt=-1):
    # Generate topdown map
    top_down_map = maps.get_topdown_map(
        cast("HabitatSim", env.sim)
    )
    recolor_map = np.array(
        [[255, 255, 255], [128, 128, 128], [0, 0, 0]], dtype=np.uint8
    )
    
    top_down_map = recolor_map[top_down_map]
    print(type(top_down_map))
    print(top_down_map.shape)
    top_down_map = Image.fromarray(top_down_map)
    #print(top_down_map.size)
    #top_down_map = top_down_map.resize((top_down_map.size[0]*10, top_down_map.size[1]*10))
    #print(top_down_map.size)

    fig = plt.figure()
    ax = plt.subplot(1, 1, 1)
    ax.axis("off")
    ax.set_title("top_down_map.png")
    plt.imshow(top_down_map)
    plt.xlim(400, 650)
    plt.ylim(750, 600)
    #plt.figure(figsize=(4, 4))
    path = "./figures/top_down_map.png"
    if opt !=-1:
        path = "./figures/top_down_map" + str(opt) + ".png"
    fig.savefig(path)
    plt.show(block=False)
    
    
if __name__ == '__main__':
    exp_config = "./habitat/config/test.yaml"
    opts = None
    config = get_config(exp_config, opts)
    
    dataset_path = "figures/test2.json.gz"
        
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
    
    print(config)
    print("########################")
    
    with Env(config=config.TASK_CONFIG) as env:
        observation = env.reset()
        info = env.get_metrics()
        print("OBSERVATION:")
        print(observation)
        print("INFO:")
        print(info)
        display_sample(observation['rgb'], observation['semantic'], np.squeeze(observation['depth']), config=config.TASK_CONFIG, opt=-1)
        print("State:")
        print(env.sim.get_agent_state())
        example_get_topdown_map(config, env, opt=-1)
        print("ddd")
    