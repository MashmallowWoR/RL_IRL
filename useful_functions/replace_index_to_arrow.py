def replace_index_to_arrow(policy):
    #Original policy is in number. After replaced to arrow in the first time, policy become string
    policy= np.where(policy==0, '\u27a1', policy) #'\u27a1' is right arrow in Unicode
    policy= np.where(policy=='1.0', '\u2b05', policy) #'\u2b05' is left arrow in Unicode
    policy= np.where(policy=='2.0', '\u2b06', policy) #'\u2b06' is up arrow in Unicode
    policy= np.where(policy=='3.0', '\u2b07', policy) #'\u2b07' is down arrow in Unicode
    return policy