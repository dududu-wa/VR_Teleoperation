from legged_gym import LEGGED_GYM_ROOT_DIR, LEGGED_GYM_ENVS_DIR

from .r2.r2 import R2Robot
from legged_gym.envs.r2.r2interrupt import R2InterruptRobot

from legged_gym.envs.r2.r2_config import R2Cfg, R2CfgPPO
from legged_gym.envs.r2.r2interrupt_config import R2InterruptCfg, R2InterruptCfgPPO
from legged_gym.envs.r2.r2_amp_config import R2AmpCfg, R2AmpCfgPPO

from legged_gym.utils.task_registry import task_registry

task_registry.register("r2int", R2InterruptRobot, R2InterruptCfg(), R2InterruptCfgPPO())
task_registry.register("r2amp", R2InterruptRobot, R2AmpCfg(), R2AmpCfgPPO())

