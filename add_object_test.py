import os
import random
import math

import numpy as np
from gym import spaces

from typing import cast

from matplotlib import pyplot as plt

from PIL import Image
import habitat_sim
from habitat_sim.utils.common import d3_40_colors_rgb

from habitat_baselines.config.default import get_config  
from habitat.sims.habitat_simulator.habitat_simulator import HabitatSim
from habitat_baselines.common.env_utils import construct_envs
from habitat_baselines.common.environments import get_env_class
from habitat.core.env_point import Env
#from habitat.core.env import Env
from habitat.utils.visualizations import maps
from habitat_baselines.common.utils import quat_from_angle_axis

def display_sample(
    rgb_obs, semantic_obs=np.array([]), depth_obs=np.array([]), config=None, opt=-1
):
    rgb_img = Image.fromarray(rgb_obs, mode="RGB")
    
    arr = [rgb_obs]
    titles = ["rgb"]
    print(rgb_img.size)
    print(rgb_img)
    
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
        #print(config)
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
    
    path = "./figures/fig.png"
    if opt !=-1:
        path = "./figures/fig" + str(opt) + ".png"
    plt.savefig(path)
    plt.show(block=False)

def add_objects(env, sim, config=None, num=1):
    object_position = np.array(
        #[8.816797, 3.5141459, 7.3605705]
        [8.816797, 3.8141459, 7.7605705]
    )
    
    # hard coded dimensions of maximum bounding box for all 3 default objects:
    max_union_bb_dim = np.array([0.125, 0.19, 0.26])
    
    object_lib_size = sim._sim.get_physics_object_library_size()
    print("object_lib_size:" + str(object_lib_size))
    object_init_grid_dim = (3, 1, 3)
    object_init_grid = {}
    
    # clear the objects if we are re-running this initializer
    for old_obj_id in sim._sim.get_existing_object_ids():
        print("remove:" + str(old_obj_id))
        sim._sim.remove_object(old_obj_id)
        
    index = [0, 1]
    position = [[8.816797, 3.8141459, 7.7605705], [8.316797, 3.8141459, 6.7605705]]
    for obj_id in range(num):
        #rand_obj_index = 0
        rand_obj_index = index[obj_id]
        object_position = position[obj_id]
        object_id = sim._sim.add_object(rand_obj_index)
        sim._sim.set_translation(object_position, object_id)
        print("added object: " + str(object_id) + " of type " + str(rand_obj_index) + " at: " + str(object_position))
        
    observation = env.step("MOVE_FORWARD")
    display_sample(observation['rgb'], observation['semantic'], np.squeeze(observation['depth']), config=config, opt=3)
    
    
def add_objects_mesh(env, sim, config=None, num=1):
    object_position = np.array(
        #[8.816797, 3.5141459, 7.3605705]
        [8.816797, 3.8141459, 7.7605705]
    )
    
    # hard coded dimensions of maximum bounding box for all 3 default objects:
    max_union_bb_dim = np.array([0.125, 0.19, 0.26])
    
    object_lib_size = sim._sim.get_physics_object_library_size()
    print("object_lib_size:" + str(object_lib_size))
    object_init_grid_dim = (3, 1, 3)
    object_init_grid = {}
    
    # clear the objects if we are re-running this initializer
    for old_obj_id in sim._sim.get_existing_object_ids():
        print("remove:" + str(old_obj_id))
        sim._sim.remove_object(old_obj_id)
        
    index = [3, 2]
    position = [[8.816797, 3.8141459, 7.7605705], [8.316797, 3.8141459, 6.7605705]]
    for obj_id in range(num):
        #rand_obj_index = 0
        rand_obj_index = index[obj_id]
        object_position = position[obj_id]
        object_id = sim._sim.add_object(rand_obj_index)
        sim._sim.set_translation(object_position, object_id)
        sim._sim.set_object_motion_type(habitat_sim.physics.MotionType.STATIC, object_id)
        navmesh_settings = habitat_sim.NavMeshSettings()
        navmesh_settings.set_defaults()
        sim._sim.recompute_navmesh(sim._sim.pathfinder, navmesh_settings, include_static_objects=True)
        print("added object: " + str(object_id) + " of type " + str(rand_obj_index) + " at: " + str(object_position))
        
    observation = env.step("MOVE_FORWARD")
    observation = env.step("MOVE_FORWARD")
    display_sample(observation['rgb'], observation['semantic'], np.squeeze(observation['depth']), config=config, opt=13)
    observation = env.step("MOVE_FORWARD")
    observation = env.step("MOVE_FORWARD")
    display_sample(observation['rgb'], observation['semantic'], np.squeeze(observation['depth']), config=config, opt=14)
    observation = env.step("MOVE_FORWARD")
    observation = env.step("MOVE_FORWARD")
    display_sample(observation['rgb'], observation['semantic'], np.squeeze(observation['depth']), config=config, opt=15)
    observation = env.step("MOVE_FORWARD")
    observation = env.step("MOVE_FORWARD")
    display_sample(observation['rgb'], observation['semantic'], np.squeeze(observation['depth']), config=config, opt=16)
    observation = env.step("MOVE_FORWARD")
    observation = env.step("MOVE_FORWARD")
    display_sample(observation['rgb'], observation['semantic'], np.squeeze(observation['depth']), config=config, opt=17)
    observation = env.step("MOVE_FORWARD")
    observation = env.step("MOVE_FORWARD")
    display_sample(observation['rgb'], observation['semantic'], np.squeeze(observation['depth']), config=config, opt=18)
    observation = env.step("MOVE_FORWARD")
    observation = env.step("MOVE_FORWARD")
    display_sample(observation['rgb'], observation['semantic'], np.squeeze(observation['depth']), config=config, opt=19)
    observation = env.step("MOVE_FORWARD")
    observation = env.step("MOVE_FORWARD")
    display_sample(observation['rgb'], observation['semantic'], np.squeeze(observation['depth']), config=config, opt=20)
        
    
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
    config.TASK_CONFIG.SIMULATOR.HABITAT_SIM_V0.PHYSICS_CONFIG_FILE = ("./data/default.phys_scene_config.json")
    config.freeze()
    
    print(config)
    print("########################")
    
    with Env(config=config.TASK_CONFIG) as env:
        sim = env.sim
        observation = env.reset()
        info = env.get_metrics()
        print("OBSERVATION:")
        print(observation)
        print("INFO:")
        print(info)
        display_sample(observation['rgb'], observation['semantic'], np.squeeze(observation['depth']), config=config.TASK_CONFIG, opt=2)
        print("State:")
        print(env.sim.get_agent_state())
        
        
        add_objects(env, sim, config=config.TASK_CONFIG, num=2)
        #add_objects_mesh(env, sim, num=2)