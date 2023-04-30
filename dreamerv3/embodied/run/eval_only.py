import re

import embodied
import numpy as np
import moviepy.editor


def eval_only(agent, env, logger, args, save_video=False):

  logdir = embodied.Path(args.logdir)
  logdir.mkdirs()
  if save_video:
    video_dir = logdir / "eval_videos"
    video_dir.mkdirs()

  print('Logdir', logdir)
  should_log = embodied.when.Clock(args.log_every)
  step = logger.step
  metrics = embodied.Metrics()
  print('Observation space:', env.obs_space)
  print('Action space:', env.act_space)

  timer = embodied.Timer()
  timer.wrap('agent', agent, ['policy'])
  timer.wrap('env', env, ['step'])
  timer.wrap('logger', logger, ['write'])

  nonzeros = set()
  n_videos = 0
  def per_episode(ep):
    nonlocal n_videos
    length = len(ep['reward']) - 1
    score = float(ep['reward'].astype(np.float64).sum())
    logger.add({'length': length, 'score': score}, prefix='episode')
    print(f'Episode has {length} steps and return {score:.1f}.')
    stats = {}
    for key in args.log_keys_video:
      if key in ep:
        stats[f'policy_{key}'] = ep[key]
        if save_video:
          video = moviepy.editor.ImageSequenceClip(list(ep[key]), fps=20)
          video.write_videofile(str(video_dir / f"video_{n_videos}.mp4"), logger=None)
          n_videos += 1
    for key, value in ep.items():
      if not args.log_zeros and key not in nonzeros and (value == 0).all():
        continue
      nonzeros.add(key)
      if re.match(args.log_keys_sum, key):
        stats[f'sum_{key}'] = ep[key].sum()
      if re.match(args.log_keys_mean, key):
        stats[f'mean_{key}'] = ep[key].mean()
      if re.match(args.log_keys_max, key):
        stats[f'max_{key}'] = ep[key].max(0).mean()
    metrics.add(stats, prefix='stats')

  driver = embodied.Driver(env)
  driver.on_episode(lambda ep, worker: per_episode(ep))
  driver.on_step(lambda tran, _: step.increment())

  if args.from_checkpoint:
    checkpoint = embodied.Checkpoint()
    checkpoint.agent = agent
    checkpoint.load(args.from_checkpoint, keys=['agent'])

  print('Start evaluation loop.')
  policy = lambda *args: agent.policy(*args, mode='eval')
  driver(policy, episodes=args.eval_eps)
  logger.add(metrics.result())
  logger.add(timer.stats(), prefix='timer')
  logger.write(fps=True)
  logger.write()
