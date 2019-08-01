import os
from multiprocessing import Process, Queue

import cloudpickle
import numpy as np


__all__ = ['SubprocVecEnv']


def _worker(exclusive_queue, shared_queue, env_fn_wrapper):
    """Env worker
    Args:
        exclusive_queue (Queue): queue that receive signal
        shared_queue (Queue): queue that put result
        env_fn_wrapper (PickleWrapper): pickled env
    """
    env = env_fn_wrapper.x()
    pid = os.getpid()
    while True:
        cmd, data = exclusive_queue.get()
        if cmd == 'step':
            ob, reward, done, info = env.step(data)
            if done:
                ob = env.reset()
            shared_queue.put(((ob, reward, done, info), pid))
        elif cmd == 'reset':
            ob = env.reset()
            shared_queue.put(((ob, 0, False, {}), pid))
        elif cmd == 'close':
            exclusive_queue.close()
            break
        elif cmd == 'get_spaces':
            shared_queue.put((env.observation_space, env.action_space))
        else:
            raise NotImplementedError


class PickleWrapper(object):
    """Uses cloudpickle to serialize contents"""
    def __init__(self, x):
        self.x = x

    def __getstate__(self):
        return cloudpickle.dumps(self.x)

    def __setstate__(self, ob):
        import pickle
        self.x = pickle.loads(ob)


class SubprocVecEnv(object):
    def __init__(self, env_fns, menvs=None):
        """
        Args:
            env_fns (list): list of callable objects to build envs
            menvs (int): number of env send in as batch
        """
        self.nenvs = len(env_fns)  # all env in sample buffer
        self.menvs = menvs or self.nenvs
        assert self.nenvs >= self.menvs
        self.closed = False  # all envs are closed

        env_queues = [Queue() for _ in range(self.nenvs)]
        self.shared_queue = Queue()
        self.ps = [Process(
            target=_worker,
            args=(env_queues[i], self.shared_queue, PickleWrapper(env_fns[i])))
            for i in range(self.nenvs)
        ]
        for p in self.ps:
            # if the main process crashes, we should not cause things to hang
            p.daemon = True
            p.start()

        self.env_queues[0].put(('get_spaces', None))
        observation_space, action_space = self.shared_queue.get()
        self.observation_space = observation_space
        self.action_space = action_space

        self.env_queues = dict()
        for p, queue in zip(self.ps, env_queues):
            self.env_queues[p.pid] = queue
        self.current_pids = None

    def _step_async(self, actions):
        """Tell the environments in the buffer to start taking a step
        with the given actions.
        """
        for pid, action in zip(self.current_pids, actions):
            self.env_queues[pid].put(('step', action))

    def _step_wait(self):
        """Wait for the step taken with _step_async().
        Returns:
            obs (np.array): an array of observations with len menvs
            rews (np.array): an array of rewards with len menvs
            dones (np.array): an array of episode done with len menvs
            infos (list[dict]): a sequence of info objects with len menvs
        """
        results = []
        self.current_pids = []
        while len(results) < self.menvs:
            data, pid = self.shared_queue.get()
            results.append(data)
            self.current_pids.append(pid)
        obs, rews, dones, infos = zip(*results)
        return np.stack(obs), np.stack(rews), np.stack(dones), infos

    def reset(self):
        """Reset all the environments and return an array of observations."""
        for queue in self.env_queues.values():  # initialize all
            queue.put(('reset', None))
        results = []
        self.current_pids = []
        while len(results) < self.menvs:
            data, pid = self.shared_queue.get()
            results.append(data[0])
            self.current_pids.append(pid)
        return np.stack(results)

    def close(self):
        if self.closed:
            return
        for queue in self.env_queues.values():
            queue.put(('close', None))
        self.shared_queue.close()
        for p in self.ps:
            p.join()
            self.closed = True

    def __len__(self):
        return self.nenvs

    def step(self, actions):
        self._step_async(actions)
        return self._step_wait()
