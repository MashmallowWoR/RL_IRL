
# coding: utf-8

# ## Question 1:

# In[61]:


from matplotlib import pyplot as plt
import numpy as np


# In[2]:


rf1=np.zeros((10,10))
rf1[9,9]=1
rf1


# In[3]:


rf2=np.zeros((10,10))
vals = [-100,-100,-100,-100,-100,-100,-100,-100,-100,-100, -100,-100,-100,-100,-100,-100,-100,-100,-100,10]
pos = [(1,4),(2,4),(3,4),(4,4),(5,4),(6,4),(1,5),(1,6),(2,6),(3,6),(7,6),(8,6),(3,7),(7,7),(3,8),(4,8),(5,8),(6,8),(7,8),(9,9)]
rows, cols = zip(*pos)
rf2[rows, cols] = vals
rf2


# In[4]:


def plot_heatmap(data, title):
    heatmap = plt.pcolor(data, cmap='plasma')
    plt.colorbar()
    plt.gca().invert_yaxis() #plt.gca(): get the current polar axes on the current figure
    plt.title(title)
    plt.show()


# In[5]:


plot_heatmap(rf1, 'Reward Function 1')


# In[6]:


plot_heatmap(rf2, 'Reward Function 2')


# ## Question 2:

# In[7]:


w=0.1
Pr=np.zeros((100,100))
for s in range(len(Pr[0])):
    if (s==0):
        #print('s==0', s)
        Pr[s,s]=w/4+w/4
        Pr[s,s+1]=w/4
        Pr[s,s+10]=1-w+w/4
    elif (s>0) and (s<9):
        #print('s>0 and s<9', s)
        Pr[s,s-1]=w/4
        Pr[s,s]=w/4
        Pr[s,s+1]=w/4
        Pr[s,s+10]=1-w+w/4   
    elif (s==9):
        #print('s==9', s)
        Pr[s,s-1]=w/4
        Pr[s,s]=w/4+w/4
        Pr[s,s+10]=1-w+w/4
    elif (s%10==0) and (s>0) and (s<90):
        #print('s%10==0 and s>0 and s<9',s)
        Pr[s,s-10]=w/4
        Pr[s,s]=w/4
        Pr[s,s+1]=w/4
        Pr[s,s+10]=1-w+w/4          
    elif (s==90):
        #print('s==90', s)
        Pr[s,s-10]=w/4
        Pr[s,s]=1-w+w/4+w/4
        Pr[s,s+1]=w/4
    elif (s>90) and (s<99):
        #print('s>90 and s<99', s)
        Pr[s,s-10]=w/4
        Pr[s,s-1]=w/4
        Pr[s,s]=1-w+w/4
        Pr[s,s+1]=w/4
    elif ((s-9)%10==0) and (s>9) and (s<99):
        #print('s%9==0 and s>9 and s<99', s)
        Pr[s,s-10]=w/4
        Pr[s,s-1]=w/4
        Pr[s,s]=w/4
        Pr[s,s+10]=1-w+w/4
    elif (s==99):
        #print('s==99', s)
        Pr[s,s-10]=w/4
        Pr[s,s-1]=w/4
        Pr[s,s]=1-w+w/4+w/4
    else:
        #print('non boundry', s)
        Pr[s,s-10]=w/4
        Pr[s,s-1]=w/4
        Pr[s,s+1]=w/4
        Pr[s,s+10]=1-w+w/4
print(Pr)


# In[8]:


Pl=np.zeros((100,100))
for s in range(len(Pl[0])):
    if (s==0):
        #print('s==0', s)
        Pl[s,s]=1-w+w/4+w/4
        Pl[s,s+1]=w/4
        Pl[s,s+10]=w/4
    elif (s>0) and (s<9):
        #print('s>0 and s<9', s)
        Pl[s,s-1]=w/4
        Pl[s,s]=1-w+w/4
        Pl[s,s+1]=w/4
        Pl[s,s+10]=w/4   
    elif (s==9):
        #print('s==9', s)
        Pl[s,s-1]=w/4
        Pl[s,s]=1-w+w/4+w/4
        Pl[s,s+10]=w/4
    elif (s%10==0) and (s>0) and (s<90):
        #print('s%10==0 and s>0 and s<9',s)
        Pl[s,s-10]=1-w+w/4
        Pl[s,s]=w/4
        Pl[s,s+1]=w/4
        Pl[s,s+10]=w/4          
    elif (s==90):
        #print('s==90', s)
        Pl[s,s-10]=1-w+w/4
        Pl[s,s]=w/4+w/4
        Pl[s,s+1]=w/4
    elif (s>90) and (s<99):
        #print('s>90 and s<99', s)
        Pl[s,s-10]=1-w+w/4
        Pl[s,s-1]=w/4
        Pl[s,s]=w/4
        Pl[s,s+1]=w/4
    elif ((s-9)%10==0) and (s>9) and (s<99):
        #print('s%9==0 and s>9 and s<99', s)
        Pl[s,s-10]=1-w+w/4
        Pl[s,s-1]=w/4
        Pl[s,s]=w/4
        Pl[s,s+10]=w/4
    elif (s==99):
        #print('s==99', s)
        Pl[s,s-10]=1-w+w/4
        Pl[s,s-1]=w/4
        Pl[s,s]=w/4+w/4
    else:
        #print('non boundry', s)
        Pl[s,s-10]=1-w+w/4
        Pl[s,s-1]=w/4
        Pl[s,s+1]=w/4
        Pl[s,s+10]=w/4
print(Pl)


# In[9]:


Pu=np.zeros((100,100))
for s in range(len(Pu[0])):
    if (s==0):
        print('s==0', s)
        Pu[s,s]=1-w+w/4+w/4
        Pu[s,s+1]=w/4
        Pu[s,s+10]=w/4
    elif (s>0) and (s<9):
        print('s>0 and s<9', s)
        Pu[s,s-1]=1-w+w/4
        Pu[s,s]=w/4
        Pu[s,s+1]=w/4
        Pu[s,s+10]=w/4   
    elif (s==9):
        print('s==9', s)
        Pu[s,s-1]=1-w+w/4
        Pu[s,s]=w/4+w/4
        Pu[s,s+10]=w/4
    elif (s%10==0) and (s>0) and (s<90):
        print('s%10==0 and s>0 and s<9',s)
        Pu[s,s-10]=w/4
        Pu[s,s]=1-w+w/4
        Pu[s,s+1]=w/4
        Pu[s,s+10]=w/4          
    elif (s==90):
        print('s==90', s)
        Pu[s,s-10]=w/4
        Pu[s,s]=1-w+w/4+w/4
        Pu[s,s+1]=w/4
    elif (s>90) and (s<99):
        print('s>90 and s<99', s)
        Pu[s,s-10]=w/4
        Pu[s,s-1]=1-w+w/4
        Pu[s,s]=w/4
        Pu[s,s+1]=w/4
    elif ((s-9)%10==0) and (s>9) and (s<99):
        print('s%9==0 and s>9 and s<99', s)
        Pu[s,s-10]=w/4
        Pu[s,s-1]=1-w+w/4
        Pu[s,s]=w/4
        Pu[s,s+10]=w/4
    elif (s==99):
        print('s==99', s)
        Pu[s,s-10]=w/4
        Pu[s,s-1]=1-w+w/4
        Pu[s,s]=w/4+w/4
    else:
        print('non boundry', s)
        Pu[s,s-10]=w/4
        Pu[s,s-1]=1-w+w/4
        Pu[s,s+1]=w/4
        Pu[s,s+10]=w/4
print(Pu)


# In[10]:


Pd=np.zeros((100,100))
for s in range(len(Pd[0])):
    if (s==0):
        print('s==0', s)
        Pd[s,s]=w/4+w/4
        Pd[s,s+1]=1-w+w/4
        Pd[s,s+10]=w/4
    elif (s>0) and (s<9):
        print('s>0 and s<9', s)
        Pd[s,s-1]=w/4
        Pd[s,s]=w/4
        Pd[s,s+1]=1-w+w/4
        Pd[s,s+10]=w/4 
    elif (s==9):
        print('s==9', s)
        Pd[s,s-1]=w/4
        Pd[s,s]=1-w+w/4+w/4
        Pd[s,s+10]=w/4
    elif (s%10==0) and (s>0) and (s<90):
        print('s%10==0 and s>0 and s<9',s)
        Pd[s,s-10]=w/4
        Pd[s,s]=w/4
        Pd[s,s+1]=1-w+w/4 
        Pd[s,s+10]=w/4     
    elif (s==90):
        print('s==90', s)
        Pd[s,s-10]=w/4
        Pd[s,s]=w/4+w/4
        Pd[s,s+1]=1-w+w/4
    elif (s>90) and (s<99):
        print('s>90 and s<99', s)
        Pd[s,s-10]=w/4
        Pd[s,s-1]=w/4
        Pd[s,s]=w/4
        Pd[s,s+1]=1-w+w/4
    elif ((s-9)%10==0) and (s>9) and (s<99):
        print('s%9==0 and s>9 and s<99', s)
        Pd[s,s-10]=w/4
        Pd[s,s-1]=w/4
        Pd[s,s]=1-w+w/4
        Pd[s,s+10]=w/4
    elif (s==99):
        print('s==99', s)
        Pd[s,s-10]=w/4
        Pd[s,s-1]=w/4
        Pd[s,s]=1-w+w/4+w/4
    else:
        print('non boundry', s)
        Pd[s,s-10]=w/4
        Pd[s,s-1]=w/4
        Pd[s,s+1]=1-w+w/4
        Pd[s,s+10]=w/4
print(Pd)


# In[11]:


P=[Pr, Pl, Pu, Pd]
P


# In[14]:


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


# In[15]:


Vs1,policy1= value_iteration(P, rf1.T.flatten())
print(Vs1)
print(policy1)


# In[16]:


Vs1_t=Vs1.reshape(10,10).T
Vs1_t


# In[17]:


def plot_table(data):
    fig = plt.figure()
    ax = fig.add_subplot(111)   
    col_labels = list(range(0,10))
    row_labels = [' 0 ', ' 1 ', ' 2 ', ' 3 ', ' 4 ', ' 5 ', ' 6 ', ' 7 ', ' 8 ', ' 9 ']

    # Draw table
    value_table = plt.table(cellText=data, colWidths=[0.05] * 10,
                          rowLabels=row_labels, colLabels=col_labels,
                          loc='center')
    value_table.auto_set_font_size(True)
    value_table.set_fontsize(24)
    value_table.scale(2.5, 2.5)

    # Removing ticks and spines enables you to get the figure only with table
    plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    plt.tick_params(axis='y', which='both', right=False, left=False, labelleft=False)
    for pos in ['right','top','bottom','left']:
        plt.gca().spines[pos].set_visible(False)


# In[18]:


plot_table(Vs1_t.round(3))


# ## Question 3:

# In[19]:


plot_heatmap(Vs1_t, 'Optimal state values')


# ## Question 4:

# Higher optimal state values is on the right bottom corner and lower down as states farther away from the right bottom corner. Because reward function 1 is high on right bottom corner.

# ## Question 5:

# In[146]:


policy1_t=policy1.reshape(10,10).T
policy1_t


# In[93]:


def replace_index_to_arrow(policy):
    #Original policy is in number. After replaced to arrow in the first time, policy become string
    policy= np.where(policy==0, '\u27a1', policy) #'\u27a1' is right arrow in Unicode
    policy= np.where(policy=='1.0', '\u2b05', policy) #'\u2b05' is left arrow in Unicode
    policy= np.where(policy=='2.0', '\u2b06', policy) #'\u2b06' is up arrow in Unicode
    policy= np.where(policy=='3.0', '\u2b07', policy) #'\u2b07' is down arrow in Unicode
    return policy


# In[147]:


policy1_ta= replace_index_to_arrow(policy1_t)
plot_table(policy1_ta)


# The optimal policy matches with intuition. The lower value states tend to move to higher value states. Since the right bottom corner has higher value states, the action agent takes goes to that direction too. 
# It is possible for the agent to compute the optimal action to take at each state by observing the optimal values of its neighboring states. The action will be going towards the neighbor that has higher state value.

# ## Question 6:

# In[73]:


Vs2,policy2= value_iteration(P, rf2.T.flatten())
print(Vs2)
print(policy2)


# In[75]:


Vs2_t=Vs2.reshape(10,10).T
plot_table(Vs2_t.round(3))


# ## Question 7:

# In[76]:


plot_heatmap(Vs2_t, 'Optimal state values')


# ## Question 8:

# Right bottom corner has higher optimal state values and top middle section has lower values in "n" shape, which is similar to heatmap in reward function 2 has. Because reward function 2 is high on right bottom corner and gives penalties (negative values) in some section in the middle.

# ## Question 9:

# In[79]:


policy2_t=policy2.reshape(10,10).T
policy2_ta= replace_index_to_arrow(policy2_t)
plot_table(policy2_ta)


# The optimal policy matches with intuition. The lower value states tend to move to higher value states. Since the right bottom corner has higher value states, the action agent takes goes to that direction too. And agent tends to avoid areas with lower values.
# It is possible for the agent to compute the optimal action to take at each state by observing the optimal values of its neighboring states. The action will be going towards the neighbor that has higher state value.

# ## Question 10:

# c=[I,
#    -lambda,
#    0]
# x=[ti ui R]
# D= [I  0  (Pa-Pa1)(I-rPa)^(-1),
#     0  0  (Pa-Pa1)(I-rPa)^(-1),
#     0 -I  I,
#     0 -I  -I,
#     0  0  I,
#     0  0  -I]
# b= [0
#     0
#     0
#     0
#     Rmax
#     Rmax]

# ## Question 11:

# In[20]:


from cvxopt import matrix,solvers


# In[138]:


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


# In[58]:


n_states=100
L1=np.arange(0, 5, 5/500)
accuracy1=np.zeros(500)
i=0
for l in L1:
    R_irl1= irl(P, policy1, max(rf1.T.flatten()), l)
    Vs1_irl, policy1_irl = value_iteration(P, R_irl1)
    m=0   
    for s in range(n_states):
        if (policy1_irl[s]==policy1[s]):       
            m+=1
    
    accuracy1[i]=m/n_states
    i+=1
print(accuracy1)


# In[64]:


def plot_accuracy(x, y):
    plt.plot(x, y)
    plt.xlabel('Lambda')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs Lambda')
    plt.show()


# In[82]:


plot_accuracy(L1, accuracy1)


# ## Question 12:

# In[98]:


max_index1=np.where(accuracy1==max(accuracy1))
l1_max1=L1[max_index1]
l1_max1


# ## Question 13:

# In[89]:


R_irl1_max= irl(P, policy1, max(rf1.T.flatten()), l1_max1[0])
print(R_irl1_max)
R_irl1_max_t= R_irl1_max.reshape(10,10).T
plot_heatmap(R_irl1_max_t, 'Reward function 1 from solving the linear program')


# In[90]:


plot_heatmap(rf1, 'Reward Function 1')


# ## Question 14:

# In[96]:


Vs1_irl_max, policy1_irl_max = value_iteration(P, R_irl1_max)
Vs1_irl_max_t=Vs1_irl_max.reshape(10,10).T
plot_heatmap(Vs1_irl_max_t, 'Optimal state values from using extracted reward function 1')


# In[97]:


print(Vs1_irl_max)


# ## Question 15:

# Similarities: Both has high value on right bottom corner and lower down as states farther away from the right bottom corner.
# 
# Differences: Heat map from Question 14 (extracted optimal values from inverse reinforcement learning) has larger high value area and the boundary line is a lot more blur.

# ## Question 16:

# In[94]:


policy1_irl_max_t=policy1_irl_max.reshape(10,10).T
policy1_irl_max_ta= replace_index_to_arrow(policy1_irl_max_t)
plot_table(policy1_irl_max_ta)


# ## Question 17:

# Similarities: In both optimal policy, agent move towards right bottom corner, which has higher state value than other areas. 
# 
# Differences: Optimal policy from Question 16 (extracted optimal policy from inverse reinforcement learning) takes more steps to move to the very right bottom corner. Because there is a larger high vaue area, agent tend to stay in that area longer and eventually move to the highest value grid. The boundary line is a lot more blur in Question 16.

# ## Question 18:

# In[80]:


n_states=100
L1=np.arange(0, 5, 5/500)
accuracy2=np.zeros(500)
i=0
for l in L1:
    R_irl2= irl(P, policy2, max(rf2.T.flatten()), l)
    Vs2_irl, policy2_irl = value_iteration(P, R_irl2)
    m=0   
    for s in range(n_states):
        if (policy2_irl[s]==policy2[s]):       
            m+=1
    
    accuracy2[i]=m/n_states
    i+=1
print(accuracy2)


# In[81]:


plot_accuracy(L1, accuracy2)


# ## Question 19:

# In[101]:


max_index2=np.where(accuracy2==max(accuracy2))
l1_max2=L1[max_index2]
l1_max2


# ## Question 20:

# In[102]:


R_irl2_max= irl(P, policy2, max(rf2.T.flatten()), l1_max2[0])
print(R_irl2_max)
R_irl2_max_t= R_irl2_max.reshape(10,10).T
plot_heatmap(R_irl2_max_t, 'Reward function 2 from solving the linear program')


# In[103]:


plot_heatmap(rf2, 'Reward Function 1')


# ## Question 21:

# In[104]:


Vs2_irl_max, policy2_irl_max = value_iteration(P, R_irl2_max)
Vs2_irl_max_t=Vs2_irl_max.reshape(10,10).T
plot_heatmap(Vs2_irl_max_t, 'Optimal state values from using extracted reward function 2')


# ## Question 22:

# Similarities: In both graph, right bottom corner has higher optimal state values and top middle right section has lower values. 
# 
# Differences: Heat map from Question 21 (extracted optimal values from inverse reinforcement learning) has more gentle value change from high area to low area and the "n" shape low value area is not as clear as seen on Quesiton 7.

# ## Question 23:

# In[105]:


policy2_irl_max_t=policy2_irl_max.reshape(10,10).T
policy2_irl_max_ta= replace_index_to_arrow(policy2_irl_max_t)
plot_table(policy2_irl_max_ta)


# ## Question 24:

# Similarities: In both optimal policy, agent move towards right bottom corner, which has higher state value than other areas. And agent tends to avoid top middle right area, which has with lower state value.
# 
# Differences: Optimal policy from Question 23 (extracted optimal policy from inverse reinforcement learning) takes more steps to move to the very right bottom corner or eventually stay at the bottom without moving to the right bottom corner. Agent tend to waste time wondering around and not directly moving towards the goal. Because the change in state value is more gentle in Question 23, the reward for agent moving is not very high when agent arrive bottom section. 

# ## Question 25:

# In[185]:


def value_iteration2(P, R, n_states=100, n_actions=4, discount_factor=0.8, threshold=0.01):
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
                Vsp[sp] = R[sp] + P[a][s][sp]* (discount_factor*temp[sp]) # this is modified logic
            Va[a]=np.sum(Vsp) 
        #print(Va)
        policy[s] =np.argmax(Va) #return the index of maximum Va
        #print(policy[s])
    return Vs, policy   


# In[188]:


L1=np.arange(0, 5, 5/500)
accuracy3=np.zeros(500)
i=0
for l in L1:
    R_irl2= irl(P, policy2, max(rf2.T.flatten()), l)
    Vs3_irl, policy3_irl = value_iteration2(P, R_irl2)
    m=0   
    for s in range(n_states):
        if (policy3_irl[s]==policy2[s]):       
            m+=1
    
    accuracy3[i]=m/n_states
    i+=1
print(accuracy3)


# In[192]:


print(max(accuracy3))
plot_accuracy(L1, accuracy3)

