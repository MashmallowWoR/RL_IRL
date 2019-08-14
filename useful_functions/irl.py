from cvxopt import matrix,solvers

def irl(P, policy, Rmax, l1, n_states=100, n_actions=4, discount_factor=0.8):
    tran_prob=np.array(P)
    c= -np.hstack([np.ones(n_states), -l1*np.ones(n_states), np.zeros(n_states)])
    
    b = np.zeros((n_states*(n_actions-1)*2 + 2*n_states, 1))
    b_bounds = np.vstack([Rmax*np.ones((n_states, 1))]*2)
    b = np.vstack((b, b_bounds))
    
    def T(a, s):
        return np.dot(tran_prob[int(policy[s]), s] - tran_prob[a, s], 
                  np.linalg.inv(np.eye(n_states) - discount_factor*tran_prob[int(policy[s])]))
    A = set(range(n_actions)) 
    zero_stack1 = np.zeros((n_states*(n_actions-1), n_states))
    T_stack = np.vstack([-T(a, s)
            for s in range(n_states)
            for a in A - {policy[s]}
        ])
    I_stack1 = np.vstack([np.eye(1, n_states, s)
            for s in range(n_states)
            for a in A - {policy[s]}
        ])
    I_stack2 = np.eye(n_states)
    zero_stack2 = np.zeros((n_states, n_states))
    
    D_left = np.vstack([I_stack1, zero_stack1, zero_stack2, zero_stack2])
    D_middle = np.vstack([zero_stack1, zero_stack1, -I_stack2, -I_stack2])
    D_right = np.vstack([T_stack, T_stack, I_stack2, -I_stack2])

    D = np.hstack([D_left, D_middle, D_right])
    
    D_bounds = np.hstack([
            np.vstack([
                np.zeros((n_states, n_states)),
                np.zeros((n_states, n_states))]),
            np.vstack([
                np.zeros((n_states, n_states)),
                np.zeros((n_states, n_states))]),
            np.vstack([
                np.eye(n_states),
                -np.eye(n_states)])
            ])   
    D = np.vstack((D, D_bounds))
    
    solvers.options['show_progress'] = False
    sol=solvers.lp(matrix(c),matrix(D),matrix(b))
    R_irl=np.asarray(sol['x'][(2*n_states):])
    R_irl=R_irl.reshape((n_states,))
    return R_irl