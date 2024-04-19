# RLProject

## Первое занятие
Выбранная среда - Atari Tetris.
Характеристики среды:
- env.action_space = Discrete(5)
- env.metadata = {'render_modes': ['human', 'rgb_array'], 'obs_types': {'rgb', 'ram', 'grayscale'}}
- env.observation_space = Box(0, 255, (210, 160, 3), uint8)
- env.reward_range = (-inf, inf)
- env.spec = EnvSpec(id='ALE/Tetris-v5', entry_point='shimmy.atari_env:AtariEnv', reward_threshold=None, nondeterministic=False, max_episode_steps=None, order_enforce=True, autoreset=False, disable_env_checker=False, apply_api_compatibility=False, kwargs={'game': 'tetris', 'obs_type': 'rgb', 'repeat_action_probability': 0.25, 'full_action_space': False, 'frameskip': 4, 'max_num_frames_per_episode': 108000}, namespace='ALE', name='Tetris', version=5, additional_wrappers=(), vector_entry_point=None)
