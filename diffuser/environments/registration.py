import gym

ENVIRONMENT_SPECS = (
    {
        'id': 'SingleSpermBezierDeepmimic-v0',
        'entry_point': ('diffuser.environments.sperm:SingleSpermBezierDeepmimic'),
    },
    {
        'id': 'SingleSpermBezierIncrements-v0',
        'entry_point': ('diffuser.environments.sperm:SingleSpermBezierIncrements'),
    },
    {
        'id': 'SingleSpermBezierIncrementsDataAug-v0',
        'entry_point': ('diffuser.environments.sperm:SingleSpermBezierIncrementsDataAug'),
    },
    {
        'id': 'SingleSpermBezierIncrementsDataAugSimplified-v0',
        'entry_point': ('diffuser.environments.sperm:SingleSpermBezierIncrementsDataAugSimplified'),
    },
    {
        'id': 'SingleSpermBezierIncrementsDataAugSimplified2-v0',
        'entry_point': ('diffuser.environments.sperm:SingleSpermBezierIncrementsDataAugSimplified2'),
    },
    {
        'id': 'SingleSpermBezierIncrementsSynth-v0',
        'entry_point': ('diffuser.environments.sperm:SingleSpermBezierIncrementsSynth'),
    },
    {
        'id': 'SingleSpermBezierIncrementsDataAugMulticlass-v0',
        'entry_point': ('diffuser.environments.sperm:SingleSpermBezierIncrementsDataAugMulticlass'),
    },

)

def register_environments():
    try:
        for environment in ENVIRONMENT_SPECS:
            gym.register(**environment)

        gym_ids = tuple(
            environment_spec['id']
            for environment_spec in  ENVIRONMENT_SPECS)

        return gym_ids
    except:
        print('[ diffuser/environments/registration ] WARNING: not registering diffuser environments')
        return tuple()