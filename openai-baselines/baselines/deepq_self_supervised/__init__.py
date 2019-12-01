from baselines.deepq_self_supervised import models  # noqa
from baselines.deepq_self_supervised.build_graph import build_act, build_train  # noqa
from baselines.deepq_self_supervised.deepq_self_supervised import learn, load_act  # noqa
from baselines.deepq_self_supervised.replay_buffer import ReplayBuffer, PrioritizedReplayBuffer  # noqa

def wrap_atari_dqn(env):
    from baselines.common.atari_wrappers import wrap_deepmind
    return wrap_deepmind(env, frame_stack=True, scale=False)
