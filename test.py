from habitat_baselines.config.default import get_config  
from habitat import registry
from habitat.tasks.nav.nav import NavigationTask
from habitat.core.env import Env
#from habitat.config import read_write

import numpy as np
from matplotlib import pyplot as plt

from PIL import Image


@registry.register_task(name="TestNav-v0")
class NewNavigationTask(NavigationTask):
    def __init__(self, config, sim, dataset):
        print("Creating a new type of task")
        super().__init__(config=config, sim=sim, dataset=dataset)
        
    def _check_episode_is_active(self, *args, **kwards):
        print(
            "Current agent position: {}".format(self._sim.get_agent_state())
        )
        collision = self._sim.previous_step_collided
        stop_called = not getattr(self, "is_stop_called", False)
        return collision or stop_called


def display_sample(
    rgb_obs, semantic_obs=np.array([]), depth_obs=np.array([]), opt=-1
):
    rgb_img = Image.fromarray(rgb_obs, mode="RGB")
    
    arr = [rgb_obs]
    titles = ["rgb"]
    if semantic_obs.size != 0:
        semantic_img = Image.new(
            "P", (semantic_obs.shape[1], semantic_obs[0])
        )
        semantic_img.putpalette(d3_40_colors_rgb.flatten())
        semantic_img.putdata((semantic_obs.flatten() % 40).astype(np.uint8))
        semantic_img = semantic_img.convert("RGBA")
        arr.append(semantic_img)
        titles.append("semantic")
        
    if depth_obs.size != 0:
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
        path = "./figures/" + str(opt) + ".png"
    plt.savefig(path)
    plt.show(block=False)


if __name__ == "__main__":
    config = get_config(
        config_paths="./pointnav_habitat_test.yaml",
    )
    #with read_write(config):
    #    config.habitat.task.type = "TestNav-v0"
    
    print(config)
        
    env = Env(config=config)
    obs = env.reset()
    display_sample(obs["rgb"], obs["semantic"], obs["depth"])
        
    env.close()