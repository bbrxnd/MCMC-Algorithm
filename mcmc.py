import sorobn as hh
import pandas as pd
import random as rand

bn = hh.BayesNet(
    ('C', ['S', 'R']),
    ('S', 'W'),
    ('R', 'W'))
bn.P['C'] = pd.Series({True: 0.5, False:0.5})
bn.P['S'] = pd.Series({
    (True, True): 0.1, (True, False): 0.9,
    (False, True): 0.5, (False, False): 0.5})
bn.P['R'] = pd.Series({
    (True, True): 0.8, (True, False): 0.2,
    (False, True): 0.2, (False, False): 0.8})
bn.P['W'] = pd.Series({
    (True, True, True): 0.99, (True, True, False): 0.01,
    (True, False, True): 0.9, (True, False, False): 0.1,
    (False, True, True): 0.95, (False, True, False): 0.05,
    (False, False, True): 0.05, (False, False, False): 0.95})

bn.prepare()
p = bn.query('C', event={'S': False, 'W': True})

# Part A. The sampling probabilities

partA_res1 = bn.query('C', event={'S' : False, 'R' : True})
partA_res2 = bn.query('C', event={'S' : False, 'R' : False})
partA_res3 = bn.query('R', event={'C' : True, 'S' : False, 'W' : True})
partA_res4 = bn.query('R', event={'C' : False, 'S' : False, 'W' : True})

print("Part A. The sampling probabilities")
print(
    f"P(C|-s,r)   = <{partA_res1.loc[True]:.4f}, {partA_res1.loc[False]:.4f}>\n"
    f"P(C|-s,-r)  = <{partA_res2.loc[True]:.4f}, {partA_res2.loc[False]:.4f}>\n"
    f"P(R|c,-s,w) = <{partA_res3.loc[True]:.4f}, {partA_res3.loc[False]:.4f}>\n"
    f"P(R|-c,-s,w)= <{partA_res4.loc[True]:.4f}, {partA_res4.loc[False]:.4f}>\n"
)

# Part B. The transition probability matrix

transition_probability_matrix = [
    [0.93215, 0.00685, 0.061, 0],   # state 1
    [0.49315, 0.162, 0, 0.34485],   # state 2
    [0.439, 0, 0.4701, 0.0909],     # state 3
    [0, 0.15515, 0.4091, 0.43575]   # state 4
]

df = pd.DataFrame(transition_probability_matrix, index=["S1", "S2", "S3", "S4"], columns=["S1", "S2", "S3", "S4"])

print("Part B. The transition probability matrix")
print(df.to_string(float_format="{:.4f}".format))

# Part C. The probability for the query P(C|-s,w)

print("\nPart C. The probability for the query P(C|-s,w)")
print(f"Exact probability: <{p.loc[True]:.4f}, {p.loc[False]:.4f}>")

true_state = {0, 1}
for k in [3, 4, 5, 6]:
    n = 10 ** k
    state = rand.randint(0,3)
    visit_count = [0, 0] # state visits: True/False

    for i in range(n):
        # if the state is true, increment true count, else increment false count
        if state in true_state:
            visit_count[0] += 1
        else:
            visit_count[1] += 1
        
        # transition state using CDF of transition probability matrix, choose to stay or move to the next state
        r = rand.random()
        accumulated_probability = 0.0
        for index, probability in enumerate(transition_probability_matrix[state]):
           accumulated_probability += probability
           if r < accumulated_probability:
              state = index
              break
           
    phat_true = visit_count[0]/n # estimated true
    phat_false = 1 - phat_true   # estimated false
    error = abs(p.loc[True] - phat_true) / p.loc[True] * 100 # calculate error
    print(f"n = 10 ^ {k}: <{phat_true:.4f}, {phat_false:.4f}>, error = {error:.2f} %")