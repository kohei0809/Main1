import os
import random
import math

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

def to_category_id(obs, env):
    print(type(env.sim._sim))
    scene = env.sim._sim.semantic_scene
    instance_id_to_label_id = {int(obj.id.split("_")[-1]): obj.category.index() for obj in scene.objects}
    mapping = np.array([ instance_id_to_label_id[i] for i in range(len(instance_id_to_label_id)) ])

    semantic_obs = np.take(mapping, obs)
    semantic_obs[semantic_obs>=40] = 0
    semantic_obs[semantic_obs<0] = 0
    return semantic_obs

def cal_CI(semantic_obs, depth_obs, score_writer, region_writer, semantic_writer, depth_writer):
    H = semantic_obs.shape[0]
    W = semantic_obs.shape[1]
    size = H * W
    print("SIZE: " + str(size))
    
    #objectのスコア別リスト
    #void, wall, floor, door, stairs, ceiling, column, railing
    score0 = [0, 1, 2, 4, 16, 17, 24, 30] 
    #chair, table, picture, cabinet, window, sofa, bed, curtain, chest_of_drawers, sink, toilet, stool, shower, bathtub, counter, lighting, beam, shelving, blinds, seating, objects
    score1 = [3, 5, 6, 7, 9, 10, 11, 12, 13, 15, 18, 19, 23, 25, 26, 28, 29, 31, 32, 34, 39]
    #cushion, plant, towel, mirror, tv_monitor, fireplace, gym_equipment, board_panel, furniture, appliances, clothes
    score2 = [8, 14, 20, 21, 22, 27, 33, 35, 36, 37, 38]
    
    #objectのcategoryリスト
    category = []
    
    ci = 0.0
    num_0 = 0
    num_1 = 0
    num_2 = 0
    region_1 = 0
    region_3 = 0
    region_5 = 0
    for i in range(H):
        for j in range(W):
            #領域スコア
            if i >= 96 and i <= 159 and j >= 96 and j <= 159:
                w = 5.0
                region_5 += 1
            elif i >= 64 and i <= 191 and j >= 64 and j <= 191:
                w = 3.0
                region_3 += 1
            else:
                w = 1.0
                region_1 += 1
            
            #オブジェクトまでの距離
            d = math.sqrt(depth_obs[i][j])
            d = max(d, 1.0)
                
            obs = semantic_obs[i][j]
            if obs in score0:
                #オブジェクトのスコア
                v = -0.01
                num_0 += 1
            else:
                if obs not in category:
                    category.append(obs)
                if obs in score1:
                    v = 1.0
                    num_1 += 1
                else:
                    v = 2.0
                    num_2 += 1
                
            score = w * v / d
            score_writer.write(str(score))
            region_writer.write(str(w))
            semantic_writer.write(str(v))
            depth_writer.write(str(d))
            
            ci += score
            
        score_writer.writeLine()
        region_writer.writeLine()
        semantic_writer.writeLine()
        depth_writer.writeLine()
        
    ci *= len(category)
    ci /= size
    print("CATEGORY: " + str(category))
    print("NUM_0: " + str(num_0))
    print("NUM_1: " + str(num_1))
    print("NUM_2: " + str(num_2))
    #print("REGION_1: " + str(region_1))
    #print("REGION_3: " + str(region_3))
    #print("REGION_5: " + str(region_5))
    print("CATEGORY_NUM: " + str(len(category)))
    print("SCORE: " + str(ci))
                
                

def display_sample(
    rgb_obs, semantic_obs=np.array([]), depth_obs=np.array([]), env=None, config=None, opt=-1
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
        depth_obs = (depth_obs - config.SIMULATOR.DEPTH_SENSOR.MIN_DEPTH) / (
                    config.SIMULATOR.DEPTH_SENSOR.MAX_DEPTH - config.SIMULATOR.DEPTH_SENSOR.MIN_DEPTH
                )
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
    
    path = "./figures/cal_CI/fig.png"
    if opt !=-1:
        path = "./figures/cal_CI/fig" + str(opt) + ".png"
    plt.savefig(path)
    plt.show(block=False)
    
    
def display_sample_each(
    rgb_obs, semantic_obs=np.array([]), depth_obs=np.array([]), env=None, config=None, opt=-1
):  
    plt.figure()
    ax = plt.subplot(1, 1, 1)
    ax.axis("off")
    plt.imshow(rgb_obs)
    plt.subplots_adjust(left=0.1, right=0.95, bottom=0.05, top=0.95)
    path = "./figures/cal_CI/fig_rgb.png"
    if opt !=-1:
        path = "./figures/cal_CI/fig_rgb" + str(opt) + ".png"
    plt.savefig(path)
    plt.show(block=False)
    
    if semantic_obs.size != 0:
        semantic_img = Image.new(
            "P", (semantic_obs.shape[1], semantic_obs.shape[0])
        )

        semantic_img.putpalette(d3_40_colors_rgb.flatten())
        semantic_img.putdata((semantic_obs.flatten() % 40).astype(np.uint8))
        semantic_img = semantic_img.convert("RGBA")
        plt.figure()
        ax = plt.subplot(1, 1, 1)
        ax.axis("off")
        plt.imshow(semantic_img)
        plt.subplots_adjust(left=0.1, right=0.95, bottom=0.05, top=0.95)
        path = "./figures/cal_CI/fig_semantic.png"
        if opt !=-1:
            path = "./figures/cal_CI/fig_semantic" + str(opt) + ".png"
        plt.savefig(path)
        plt.show(block=False)
        
    if depth_obs.size != 0:
        depth_obs = (depth_obs - config.SIMULATOR.DEPTH_SENSOR.MIN_DEPTH) / (
                    config.SIMULATOR.DEPTH_SENSOR.MAX_DEPTH - config.SIMULATOR.DEPTH_SENSOR.MIN_DEPTH
                )
        depth_img = Image.fromarray(
            (depth_obs / 10 * 255).astype(np.uint8), mode="L"
        )
        plt.figure()
        ax = plt.subplot(1, 1, 1)
        ax.axis("off")
        plt.imshow(depth_img)
        plt.subplots_adjust(left=0.1, right=0.95, bottom=0.05, top=0.95)
        path = "./figures/cal_CI/fig_depth.png"
        if opt !=-1:
            path = "./figures/cal_CI/fig_depth" + str(opt) + ".png"
        plt.savefig(path)
        plt.show(block=False)
    
    
if __name__ == '__main__':
    exp_config = "./habitat/config/test.yaml"
    opts = None
    config = get_config(exp_config, opts)
    
    dataset_path = "figures/test6.json.gz"
        
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
    log_manager.setLogDirectory("check/CI")
    score_writer = log_manager.createLogWriter("score4")
    region_writer = log_manager.createLogWriter("region4")
    semantic_writer = log_manager.createLogWriter("semantic4")
    depth_writer = log_manager.createLogWriter("depth4")
    
    with Env(config=config.TASK_CONFIG) as env:
        observation = env.reset()
        info = env.get_metrics()
        #print(env.sim.get_agent_state())
        semantic_obs = to_category_id(observation['semantic'], env)
        #cal_CI(semantic_obs, np.squeeze(observation['depth']), score_writer, region_writer, semantic_writer, depth_writer)
        #display_sample(observation['rgb'], semantic_obs, np.squeeze(observation['depth']), env, config.TASK_CONFIG, opt=4)
        #display_sample_each(observation['rgb'], semantic_obs, np.squeeze(observation['depth']), env, config.TASK_CONFIG, opt=4)
    