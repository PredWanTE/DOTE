from networking_env.environments.ecmp.history_env import ECMPHistoryEnv

 
def get_env(props):
    env = ECMPHistoryEnv(props)
    return env