#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
r"""
This file hosts task-specific or trainer-specific environments for trainers.
All environments here should be a (direct or indirect ) subclass of Env class
in habitat. Customized environments should be registered using
``@baseline_registry.register_env(name="myEnv")` for reusability
"""

from typing import Optional, Type
from utils.log_manager import LogManager
from utils.log_writer import LogWriter
import sys

from habitat.core.logging import logger
import habitat
from habitat import Config, Dataset
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat.utils.visualizations import maps


def get_env_class(env_name: str) -> Type[habitat.RLEnv]:
    r"""Return environment class based on name.

    Args:
        env_name: name of the environment.

    Returns:
        Type[habitat.RLEnv]: env class.
    """
    return baseline_registry.get_env(env_name)


@baseline_registry.register_env(name="NavRLEnv")
class NavRLEnv(habitat.RLEnv):
    def __init__(self, config: Config, dataset: Optional[Dataset] = None):
        self._rl_config = config.RL
        self._core_env_config = config.TASK_CONFIG
        self._reward_measure_name = self._rl_config.REWARD_MEASURE
        self._take_measure_name = self._rl_config.SUCCESS_MEASURE
        self._subsuccess_measure_name = self._rl_config.SUBSUCCESS_MEASURE


        self._previous_measure = None
        self._previous_action = None
        super().__init__(self._core_env_config, dataset)

    def reset(self):
        self._previous_action = None
        observations = super().reset()
        self._previous_measure = self._env.get_metrics()[
            self._reward_measure_name
        ]
        return observations

    def step(self, *args, **kwargs):
        self._previous_action = kwargs["action"]
        return super().step(*args, **kwargs)

    def get_reward_range(self):
        return (
            self._rl_config.SLACK_REWARD - 1.0,
            self._rl_config.SUCCESS_REWARD + 1.0,
        )

    def get_reward(self, observations, **kwargs):
        reward = self._rl_config.SLACK_REWARD

        current_measure = self._env.get_metrics()[self._reward_measure_name]

        if self._episode_subsuccess():
            current_measure = self._env.task.foundDistance

        reward += self._previous_measure - current_measure
        self._previous_measure = current_measure

        if self._episode_subsuccess():
            self._previous_measure = self._env.get_metrics()[self._reward_measure_name]

        if self._episode_success():
            reward += self._rl_config.SUCCESS_REWARD
        elif self._episode_subsuccess():
            reward += self._rl_config.SUBSUCCESS_REWARD
        elif self._env.task.is_found_called and self._rl_config.FALSE_FOUND_PENALTY:
            reward -= self._rl_config.FALSE_FOUND_PENALTY_VALUE

        return reward

    def _episode_success(self):
        return self._env.get_metrics()[self._success_measure_name]


    def _episode_subsuccess(self):
        return self._env.get_metrics()[self._subsuccess_measure_name]

    def get_done(self, observations):
        done = False
        if self._env.episode_over or self._episode_success():
            done = True
        return done

    def get_info(self, observations):
        return self.habitat_env.get_metrics()
    
    
@baseline_registry.register_env(name="InfoRLEnv")
class InfoRLEnv(habitat.RLEnv):
    def __init__(self, config: Config, dataset: Optional[Dataset] = None):
        self._rl_config = config.RL
        self._core_env_config = config.TASK_CONFIG
        self._take_picture_name = self._rl_config.TAKE_PICTURE_MEASURE
        self._reward_measure_name = self._rl_config.REWARD_MEASURE
        self._picture_measure_name = self._rl_config.PICTURE_MEASURE


        self._previous_measure = None
        self._previous_action = None
        super().__init__(self._core_env_config, dataset)

    def reset(self):
        self._previous_action = None
        observations = super().reset()
        self._previous_measure = self._env.get_metrics()[
            self._reward_measure_name
        ].sum() * 0.08 * 0.08 
        return observations

    def step(self, *args, **kwargs):
        self._previous_action = kwargs["action"]
        return super().step(*args, **kwargs)

    def get_reward_range(self):
        return (
            self._rl_config.SLACK_REWARD - 1.0,
            self._rl_config.SUCCESS_REWARD + 1.0,
        )

    def get_reward(self, observations, **kwargs):
        reward = self._rl_config.SLACK_REWARD
        ci = -sys.float_info.max

        #観測領域のreward
        exp_area = self._env.get_metrics()[self._reward_measure_name].sum() 
        current_measure = exp_area * 0.8 * 0.8 * 0.1
        """
        logger.info("AREA: " + str(exp_area) + "," + str(current_measure))
        logger.info(current_measure - self._previous_measure)
        top_down_map = self._env.task.sceneMap
        count = 0
        for i in range(top_down_map.shape[0]):
            for j in range(top_down_map.shape[1]):
                if top_down_map[i][j] != maps.MAP_INVALID_POINT:
                    count += 1
                
        logger.info("TOP_DOWN_MAP: " + str(count))
        logger.info(str(current_measure/count*100) + "%")
        """

        area_reward = current_measure - self._previous_measure
        reward += area_reward
        self._previous_measure = current_measure
        matrics = None

        if self._take_picture():
            measure = self._env.get_metrics()[self._picture_measure_name]
            ci, matrics = measure[0], measure[1]
            #reward += metrics
        elif self._env.task.is_found_called and self._rl_config.FALSE_FOUND_PENALTY:
            reward -= self._rl_config.FALSE_FOUND_PENALTY_VALUE

        return [reward, ci, current_measure], matrics

    def _take_picture(self):
        return self._env.get_metrics()[self._take_picture_name]

    def get_done(self, observations):
        done = False
        if self._env.episode_over:
            done = True
        return done

    def get_info(self, observations):
        return self.habitat_env.get_metrics()

