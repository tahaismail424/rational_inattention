# rational_inattention

This repo contains all code for training and evaluating models for the monkey gambling task!

src contains all scripts, which at the moment is only taskgym.py - which defines the gym environment for this task

the gym environment currently has the following modifiable variables:

offer_probs - a tuple containing the probabilities of being rewarded for each offer type
offer_amounts - tuple defining the amount of reward given for being rewarded for a particular offer
penalty_no_choice - the penalty for not making a choice during the choice period
penalty_break_fixation - penalty for making a choice during the fixation period

^^^ more environment variables can be added as needed!

### environment specs
the observation space is a 1D array (Box) consisting of binary flags for the currently partially observable space
indices 0-2 and 4-6 one hot encode which offer types are on screen
the value of these indices indicate the probability of receiving reward
indices 3 and 7 are simply (1 - probability of reward)

the action space is a single int (Discrete) space of options 0, 1, 2 with following definitions:
0: fixate
1: pick left offer
2: pick right offer


run_recurrent_PPO.ipynb contains an example notebook for training and evaluating a model from stable baseline on our gym environment - the gym environment should be compatible with any model from sb3 and typical reinforcement training loop!