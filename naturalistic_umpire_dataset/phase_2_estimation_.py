import numpy as np
import pandas as pd
from scipy.optimize import minimize
import itertools
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import warnings

warnings.filterwarnings('ignore')
class EMAlgorithm:
    def __init__(self, df, plot_flag, plot_threshold,seed_, n_hidden_states=12, max_iterations=200000, convergence_value=0.001, temperature=1):
        self.n_hidden_states = n_hidden_states
        self.max_iterations = max_iterations
        self.convergence_value = convergence_value
        self.Temp_ = temperature
        self.plot_flag = plot_flag
        self.threshold = plot_threshold
        self.df = df
        
        # Process data
        self.posteriors = df[[f'post_s{id}' for id in range(n_hidden_states)]].to_numpy()
        # Observations
        self.z = df[["pitch_category_breaking_pitch", "pitch_category_offspeed_pitch",
                            "pitch_category_fastballs", "pitch_location"]].to_numpy()
        self.F1 = self.z.shape[1]
        self.F2 = self.F1
        self.F3 = 1

        self.outcome = df[['error_in_decision']].to_numpy().ravel()
        
        # Initialize parameters
        np.random.seed(seed_)
        self.init_b = np.random.random(self.F1 + self.F2)
        self.init_bias = np.zeros(2)
        np.random.seed(seed_)
        self.init_c = np.random.random(self.F3 * n_hidden_states)
        
        #set initial weights, bias, theta
        self.weights = np.array([self.init_b[:self.F1], self.init_b[self.F1:self.F1+self.F2]])
        self.theta = self.init_c
        self.bias = self.init_bias
        self.constant_thetas = np.zeros(n_hidden_states)
        
        self.best_weights = None
        self.best_theta = None
        self.best_bias = None
        
        self.K = 2
        self.N = len(self.z)
        self.log_likelihoods = [100]
        
    def logistic_prob_batch(self, X, weights, bias):
        x_dot_weights = np.dot(X, weights) + bias
        return 1 / (1 + np.exp(-x_dot_weights))

    def compute_policy_batch(self, thetas, posterior_probs):
        exp_R0 = np.exp(self.constant_thetas / self.Temp_)
        exp_R1 = np.exp(thetas / self.Temp_)
        
        pi_a1_given_s = exp_R1 / (exp_R1 + exp_R0)
        pi_a0_given_s = exp_R0 / (exp_R1 + exp_R0)
        
        pi_a1_matrix = np.tile(pi_a1_given_s, (self.N, 1))
        pi_a0_matrix = np.tile(pi_a0_given_s, (self.N, 1))
        
        c1_sums = np.sum(pi_a1_matrix * posterior_probs, axis=1)
        c0_sums = np.sum(pi_a0_matrix * posterior_probs, axis=1)
        
        return np.column_stack((c0_sums, c1_sums))

    def expectationStep_batch(self):
        mu_0 = self.logistic_prob_batch(self.z, self.weights[0], self.bias[0])
        mu_0 = mu_0 * self.outcome + (1 - mu_0) * (1 - self.outcome)
        
        mu_1 = self.logistic_prob_batch(self.z, self.weights[1], self.bias[1])
        mu_1 = mu_1 * self.outcome + (1 - mu_1) * (1 - self.outcome)
        
        policy = self.compute_policy_batch(self.theta, self.posteriors)
        
        denominator0 = policy[:, 0] * mu_0
        denominator1 = policy[:, 1] * mu_1
        sum_denominator = denominator0 + denominator1
        
        gamma = np.zeros((self.N, self.K))
        gamma[:, 0] = denominator0 / sum_denominator
        gamma[:, 1] = denominator1 / sum_denominator
        return gamma

    def maximizationStep_batch(self, gamma):
        def objective(params):
            new_weights, new_bias, new_theta = params[:self.F1+self.F2], params[self.F1+self.F2:self.F1+self.F2+2], params[self.F1+self.F2+2:]
            weights = [new_weights[:self.F1], new_weights[self.F1:self.F1+self.F2]]
            bias = new_bias
            thetas = new_theta
            mu_0 = self.logistic_prob_batch(self.z, weights[0], bias[0])
            mu_0 = mu_0 * self.outcome + (1 - mu_0) * (1 - self.outcome)
            mu_1 = self.logistic_prob_batch(self.z, weights[1], bias[1])
            mu_1 = mu_1 * self.outcome + (1 - mu_1) * (1 - self.outcome)
            
            policy = self.compute_policy_batch(thetas, self.posteriors)
            
            with np.errstate(divide='ignore', invalid='ignore'):
                inner0 = np.log(policy[:, 0]) + np.log(mu_0)
                inner0 = inner0 * gamma[:, 0]
                inner1 = np.log(policy[:, 1]) + np.log(mu_1)
                inner1 = inner1 * gamma[:, 1]
            
            return -np.sum(inner0 + inner1)
        
        result = minimize(fun=objective,
                          x0=np.concatenate([self.init_b, self.init_bias, self.init_c]),
                          method='SLSQP')
        
        beta_k_new, bias_new, C_new = result.x[:self.F1+self.F2], result.x[self.F1+self.F2:self.F1+self.F2+2], result.x[self.F1+self.F2+2:]
        return beta_k_new, bias_new, C_new

    def logLikelihoodCalculation_batch(self):
        mu_0 = self.logistic_prob_batch(self.z, self.weights[0], self.bias[0])
        mu_0 = mu_0 * self.outcome + (1 - mu_0) * (1 - self.outcome)
        mu_1 = self.logistic_prob_batch(self.z, self.weights[1], self.bias[1])
        mu_1 = mu_1 * self.outcome + (1 - mu_1) * (1 - self.outcome)
        
        policy = self.compute_policy_batch(self.theta, self.posteriors)
        
        likelihood = np.zeros((self.N, self.K))
        likelihood[:, 0] = mu_0 * policy[:, 0]
        likelihood[:, 1] = mu_1 * policy[:, 1]
        
        log_likelihood = np.sum(np.log(likelihood[:, 0] + likelihood[:, 1]))
        return log_likelihood

    def run_em_algorithm(self):
        best_log_likelihood = float('-inf')  # Start with negative infinity
        worse_count = 0
        
        for i in range(self.max_iterations):
            gamma = self.expectationStep_batch()
            new_weights, new_bias, new_theta = self.maximizationStep_batch(gamma)
            
            self.weights = [new_weights[:self.F1], new_weights[self.F1:self.F1+self.F2]]
            self.theta = new_theta
            self.bias = new_bias
            
            current_log_likelihood = self.logLikelihoodCalculation_batch()
            self.log_likelihoods.append(current_log_likelihood)
            
            if np.isnan(current_log_likelihood):
                print("Stopped due to NaN in log-likelihood.")
                break
            
            if current_log_likelihood > best_log_likelihood:
                best_log_likelihood = current_log_likelihood
                self.best_weights = self.weights
                self.best_theta = self.theta
                self.best_bias = self.bias
                worse_count = 0
            else:
                worse_count += 1
            
            if i % 100 == 0:
                print(f"Iteration {i + 1}: Log likelihood = {current_log_likelihood}")
                print('Weights:', self.weights)
                print('Bias:',self.bias)
                print('Theta:', self.theta)
            
            if abs(self.log_likelihoods[-1] - self.log_likelihoods[-2]) < self.convergence_value:
                print("Converged.")
                break
            if worse_count >= 100:
                print("Stopped due to 10 consecutive worsening iterations.")
                break
            
    def get_empirical_accuracy(self,threshold=0.1):
        df_filter = self.df[(self.df['pitch_location']<=threshold)]   
        accuracy_per_state = df_filter.groupby('dec_states')['error_in_decision'].mean()
        return accuracy_per_state
    
    def generate_probabilities(self, select_obs,threshold):
        post_matrix = np.eye(12)
        # Generate all valid (z1, z2, z3, z4) combinations
        z_options = [(z1, z2, z3, threshold) for z1, z2, z3 in itertools.product([0, 1], repeat=3) if z1 + z2 + z3 <= 1]
        z_matrix = np.array(z_options)
        
        # Create DataFrame
        columns = [f'post_{i}' for i in range(post_matrix.shape[1])] + [f'z_{i+1}' for i in range(z_matrix.shape[1])]
        df = pd.DataFrame([list(post) + list(z) for post in post_matrix for z in z_matrix], 
            columns=columns)
        
        N = len(z_matrix)
        prob = []
        
        for j in range(len(post_matrix)):
            for n in range(N):
                # Compute probabilities for each component
                mu_1 = self.logistic_prob_batch(z_matrix[n], self.best_weights[0], self.best_bias[0])
                mu_2 = self.logistic_prob_batch(z_matrix[n], self.best_weights[1], self.best_bias[1])
                
                # Combine with mixing coefficients
                policy_c = self.compute_policy_batch(self.best_theta, post_matrix[j])
                prob_1 = policy_c[:, 0] * mu_1
                prob_2 = policy_c[:, 1] * mu_2
                
                total_prob = prob_1 + prob_2
                total_prob = total_prob[0]
                prob.append(total_prob)
        
        probabilities = np.array(prob).reshape(len(post_matrix), N)
        return probabilities[:, select_obs]

    def plot_logistic_probabilities(self):
        n_samples = 100
        min_pitch = self.df['pitch_location'].min()
        max_pitch = self.df['pitch_location'].max()
        # Generate a continuous variable
        pitch_loc = np.linspace(start=min_pitch, stop=max_pitch, num=n_samples).reshape(n_samples, 1)
    
        # Define pitch category representation
        arr = np.array([1, 0, 0])
        repeated = np.tile(arr, n_samples).reshape(n_samples, 3)
        
        # Combine categorical and continuous variables
        z = np.concatenate((repeated, pitch_loc), axis=1)
    
        # Compute logistic probabilities
        mu_0 = self.logistic_prob_batch(z, self.best_weights[0], self.best_bias[0])
        mu_1 = self.logistic_prob_batch(z, self.best_weights[1], self.best_bias[1])
    
        # Plot results
        fig, ax = plt.subplots(figsize=(10, 5))
        if self.plot_flag == 0:
            ax.plot(z[:, -1], mu_0, color="red", linewidth=4,alpha=0.8,
                    label="Model with Low Attention (a=0)")
            ax.plot(z[:, -1], mu_1, color="green", linewidth=4,alpha=0.8,
                    label="Model with High Attention (a=1)")
        elif self.plot_flag == 1:
            ax.plot(z[:, -1], mu_1, color="red", linewidth=4,alpha=0.8,
                    label="Model with Low Attention (a=0)")
            ax.plot(z[:, -1], mu_0, color="green", linewidth=4,alpha=0.8,
                    label="Model with High Attention (a=1)")
    
        # Customize plot
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(axis='both', labelsize=16)
        plt.xlabel("Distance from Strike Zone Boundary",fontsize=16)
        plt.ylabel("P(u = correct call)",fontsize=16)
        plt.grid(True)
        plt.legend()
        
        # Show plot
        plt.show()
    
        return 0
    
    def plot_context_probabilities(self):
        new_indices_theta = np.array([4, 10, 1, 8, 9, 11, 5, 2, 3, 7, 12, 6]) - 1
    
        # Rearrange theta values
        thetas = self.best_theta
        rearranged_thetas = thetas[new_indices_theta]

        # Compute exponentiated values
        exp_R0 = np.exp(self.constant_thetas / self.Temp_)
        exp_R1 = np.exp(rearranged_thetas / self.Temp_)  # Shape (num_states,)
    
        # Compute probabilities
        pi_a1_given_s = exp_R1 / (exp_R1 + exp_R0)
        pi_a0_given_s = exp_R0 / (exp_R1 + exp_R0)
    
        # Rearrange probabilities and true values
        true_overall = self.get_empirical_accuracy().to_numpy()

        prob_overall= self.generate_probabilities(1,self.threshold)

        rearranged_prob = prob_overall[new_indices_theta]
        rearranged_true = true_overall[new_indices_theta]
    
        # Context groups and colors
        context_groups = {
            "A": ["$s_1$", "$s_2$"],
            "B": ["$s_3$", "$s_4$"],
            "C": ["$s_5$", "$s_6$"],
            "D": ["$s_7$", "$s_8$", "$s_9$"],
            "E": ["$s_{10}$", "$s_{11}$", "$s_{12}$"]
        }
    
        # Generate grayscale shades for groups A-D
        greyscale_cmap = cm.get_cmap("Greys")
        grey_shades = {
            "A": greyscale_cmap(0.3),  
            "B": greyscale_cmap(0.5),  
            "C": greyscale_cmap(0.7),  
            "D": greyscale_cmap(0.9)  
        }
        
        # Define blue for Type E
        type_E_color = "#1f77b4"
        
        # Assign colors
        group_colors = {**grey_shades, "E": type_E_color}
    
        # Define contexts
        contexts = [r"$s_1$", r"$s_2$", r"$s_3$", r"$s_4$", r"$s_5$", r"$s_6$", r"$s_7$", r"$s_8$", r"$s_9$", r"$s_{10}$", r"$s_{11}$", r"$s_{12}$"]
        
        probabilities = pi_a1_given_s  # Estimated probabilities
        if self.plot_flag == 1:
            probabilities = pi_a0_given_s  # Estimated probabilities

    
        # Assign colors dynamically
        bar_colors = []
        for ctx in contexts:
            for group, group_contexts in context_groups.items():
                if ctx in group_contexts:
                    bar_colors.append(group_colors[group])
                    break  
    
        # Sorting the data by probability in descending order
        sorted_data = sorted(zip(probabilities, contexts, bar_colors,
                                 rearranged_prob, rearranged_true
                                 ), key=lambda x: x[0], reverse=True)
    
        # Unpack sorted values
        probabilities, contexts, bar_colors, rearranged_prob, rearranged_true = zip(*sorted_data)
        #probabilities, contexts, bar_colors = zip(*sorted_data)
    
        # Convert tuples to lists
        probabilities = list(probabilities)
        contexts = list(contexts)
        bar_colors = list(bar_colors)
        #standard_errors = list(standard_errors)
        rearranged_prob = list(rearranged_prob)
        rearranged_true = list(rearranged_true)
    
        # Plot
        plt.figure(figsize=(14, 8))
        plt.gcf().set_facecolor("white")
    
        # Bar plot
        plt.bar(contexts, probabilities,
                #yerr=standard_errors,
                capsize=5, color=bar_colors, alpha=0.9)
    
        # # Estimated probability line (Hotpink)
        est_prob_line, = plt.plot(contexts, rearranged_prob, marker='o', linestyle=':', color='crimson',markersize=12, alpha=0.9)
    
        # # Error bars
        # plt.errorbar(contexts, rearranged_prob, yerr=standard_errors, fmt="none", ecolor="crimson", capsize=5, capthick=1,
        #              alpha=0.9, linestyle="None")
    
        # # True probability line (Slateblue)
        true_prob_line, = plt.plot(contexts, rearranged_true, marker='o', markersize=12, linestyle=':', color='slateblue')
    
        # Labels and title
        plt.xlabel("Game Context", size=20)
        plt.ylim(0.45, 1.0)
        plt.grid(axis="y", linestyle="--", alpha=0.4)
        plt.grid(axis="x", linestyle="--", alpha=0.4)
    
        # Legend for context types
        legend_patches = [plt.Line2D([0], [0], marker='s', color='w', markerfacecolor=group_colors[group], markersize=16, 
                                     label=f"Type {group}") for group in group_colors.keys()]
        legend1 = plt.legend(handles=legend_patches, title="Context Type", fontsize=16, title_fontsize=18)
    
        # Legend for estimated & true probabilities
        plt.legend([est_prob_line, true_prob_line], ["Estimated Accuracy", "Observed Accuracy"], loc="upper left", fontsize=16)
        plt.gca().add_artist(legend1)
    
        # Increase tick label size
        plt.tick_params(axis='x', labelsize=16)
        plt.tick_params(axis='y', labelsize=16)
    
        # Show plot
        plt.show()
        
        return 0