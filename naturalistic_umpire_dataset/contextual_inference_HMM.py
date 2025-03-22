
import numpy as np
import pandas as pd
from hmmlearn.hmm import CategoricalHMM

filename = 'Umpire_6/context_inference_umpire_6.csv'
df = pd.read_csv(filename, dtype={'ball_strike_count': str,'obs': str})

#to view observation set to categorical mapping
temp = dict( zip( df['code'],df['obs']  ))

#prepare data for context inference using HMM
subset_list = ['code','game_id']

subset_df = df[subset_list]
games = subset_df.game_id.unique()

training_seq =[]
lengths = []
for game in games:
    temp_df = subset_df.loc[(subset_df['game_id'] == game)]
    temp_df = temp_df.iloc[:,:-1].to_numpy()
    training_seq.extend(temp_df)
    lengths.append(len(temp_df))
training_seq = np.asarray(training_seq)

#HMM
n_hidden_states = 12 
best_score = best_model = None
n_fits = 1
np.random.seed(13)
i_seed = 16
for idx in range(n_fits):
    model = CategoricalHMM(n_components=n_hidden_states, random_state=i_seed,tol=1e-4,n_iter=200)  
    model.fit(training_seq,lengths)
    score = model.score(training_seq,lengths)
    print(f'Model #{idx}\tScore: {score}')
    if best_score is None or score > best_score:
        best_model = model
        best_score = score

# given the model
states = best_model.predict(training_seq,lengths)
posterior = best_model.predict_proba(training_seq,lengths)
# Print trained parameters
transition_prob = best_model.transmat_
emission_prob = best_model.emissionprob_
start_prob = best_model.startprob_


#save posterior prob to the dataframe
A = ['post_s'+str(id) for id in range(0,n_hidden_states) ]
counter=0
for col in A:
    df[col] = posterior[:,counter]
    counter+=1
#save decoded contexts to the dataframe
df['dec_states'] = states
