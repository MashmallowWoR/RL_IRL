def value_iteration(P, R, n_states=100, n_actions=4, discount_factor=0.8, threshold=0.01):
    Vs= np.zeros(n_states)
    delta=float('inf')
    while delta> threshold:
        delta=0
        temp=Vs
        for s in range(n_states): #s is in range of [0,100)
            v=Vs[s]
            Va=np.zeros(n_actions)
            for a in range(n_actions): #a is in range of [0,4)
                Vsp=np.zeros(n_states)
                for sp in range(n_states): #sp is in range of [0,100)
                    Vsp[sp] = P[a][s][sp]*(R[sp] + discount_factor*temp[sp])
                Va[a]=np.sum(Vsp)
            Vs[s]=max(Va)
            delta=max(delta, np.abs(v-Vs[s]))
    
    policy= np.zeros(n_states)
    for s in range(n_states): #s is in range of [0,100)
        Va=np.zeros(n_actions)
        for a in range(n_actions):
            Vsp=np.zeros(n_states)
            for sp in range(n_states): #sp is in range of [0,100)
                Vsp[sp] = P[a][s][sp]*(R[sp] + discount_factor*temp[sp])
            Va[a]=np.sum(Vsp)        
        policy[s] =np.argmax(Va) #return the index of maximum Va
    return Vs, policy 