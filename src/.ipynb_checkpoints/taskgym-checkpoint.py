import numpy as np
import gymnasium as gym
from gymnasium import spaces
from numpy.random import default_rng

class HaydenRiskTrial(gym.Env):
    """
    task structure:
    - ITI: 800ms
    - first offer (left or right): 400ms
        - 4D: [small, med, large, none]
            - reward probability, p ~ U(0,1)
            - stakes: small (125, 12.5%), medium (165, 43.75%), large (240, 43.75%)
            - so the offer cue is 4D, indicating stake and reward prob
                - where "none" the fourth state, with 1-p, if p is the reward probability
    - pause: 600ms
    - second offer (other side): 400ms
    - pause: 600ms
    - fixation cue: 100ms, animal must choose to fixate
    - both offers simultaneously: animal can choose
    - choice
    - feedback: 250ms
        - e.g., if unrewarded, [0 0 0 1] on the offer chosen
        - e.g., if rewarded for medium stake, [0 1 0 0]

    observation space: 9D including two 5D offer cues, fixation cue, and previous reward
    action space: 3D (choose left, fixate, choose right)
    time discretization: 50ms
    """
    def __init__(self, offer_probs=(0.125, 0.4375, 0.4375), offer_amounts=(1,2,3), penalty_no_choice=-0.1, penalty_break_fixation=-0.1, penalty_hyperactive=-0.1, reward_choice_made=0.0):
        self.observation_space = spaces.Box(low=0, high=1, shape=(9,), dtype=np.float32) # offer1 (4D), offer2 (4D), fixation
        self.action_space = spaces.Discrete(3) # fixate, choose left, choose right
        self.offer_probs = offer_probs # probability of each stake, per offer
        self.offer_amounts = offer_amounts # reward amounts of each stake
        # durations (assuming 50ms)
        self.penalty_no_choice = penalty_no_choice
        self.penalty_break_fixation = penalty_break_fixation
        self.penalty_hyperactive = penalty_hyperactive
        self.reward_choice_made = reward_choice_made
        self.epoch_durations = {'iti': 16, 'offer1': 8, 'offer2': 8, 'pause1': 12, 'pause2': 12, 'fixation': 2, 'choice': 5, 'feedback': 5}
        self.epoch_order = ['iti', 'offer1', 'pause1', 'offer2', 'pause2', 'fixation', 'choice', 'feedback']
        self.rng = default_rng() # random number generator
        self.choice_made = False
        self.state = None
    
    def _make_offer(self):
        p_offer = self.rng.random()
        offer_index = self.rng.choice([0,1,2], p=self.offer_probs) # small, med, large
        is_rewarding = self.rng.random() < p_offer # we sample success ahead of time
        r_if_chosen = self.offer_amounts[offer_index] * float(is_rewarding)
        return {'p_offer': p_offer, 'offer_index': offer_index, 'r_if_chosen': r_if_chosen}

    def reset(self, **kwargs):
        offer1 = self._make_offer()
        offer2 = self._make_offer()
        self.state = {'t': 0, 'epoch_index': 0, 't_epoch': 0, 'offer1': offer1, 'offer2': offer2, 'choice': None, 'r': None}
        return self._get_obs(), self._get_info()

    def _get_offer_obs(self, offer_name):
        obs = np.zeros(self.observation_space.shape[0])
        i_offset = 0 if offer_name == 'offer1' else 4
        offer = self.state[offer_name]
        obs[i_offset + offer['offer_index']] = offer['p_offer']
        obs[i_offset + 3] = 1-offer['p_offer']
        return obs

    def _get_obs(self):
        """
        returns indicator of whether or not we are in the ISI
        """
        obs = np.zeros(self.observation_space.shape[0])
        if self.state['epoch_index'] < len(self.epoch_order):
            cur_epoch = self.epoch_order[self.state['epoch_index']]
            if cur_epoch in ['iti', 'pause1', 'pause2']:
                # null observation
                pass
            elif cur_epoch in ['offer1', 'offer2']:
                # observe offer identity, probability, and 1-probability
                obs = self._get_offer_obs(cur_epoch)
            elif cur_epoch == 'fixation':
                # fixation cross
                obs[-1] = 1.
            elif cur_epoch == 'choice':
                # choice cue is both offers
                obs = self._get_offer_obs('offer1') + self._get_offer_obs('offer2')
            elif cur_epoch == 'feedback':
                r = self.state['r']
                a = self.state['choice']
                i_offset = 0 if a == 1 else 4
                if a: # agent chose an offer
                    offer = self.state['offer{}'.format(a)]
                    if r > 0: # reward
                        obs[i_offset + offer['offer_index']] = 1.
                    else: # no reward
                        obs[i_offset + 3] = 1.
        return obs

    def _get_info(self):
        return self.state.copy()
    
    def _get_reward(self, action):
        """
        updates reward when choice is made during choice period
        returns reward only when in feedback period
        """
        cur_epoch = self.epoch_order[self.state['epoch_index']]
        if cur_epoch == 'choice' and action != 0:
            # get the reward from the chosen offer
            self.state['choice'] = action
            self.state['r'] = self.state['offer{}'.format(action)]['r_if_chosen']
            self.choice_made = True
            return self.reward_choice_made
        elif cur_epoch == 'feedback':
            if self.state['r'] is None: # no action was chosen during choice period
                self.state['r'] = self.penalty_no_choice
                self.state['choice'] = 0
            return self.state['r']
        elif cur_epoch == 'fixation':
            if action != 0:
                return self.penalty_break_fixation
        elif cur_epoch == 'fixation':
            if action != 0:
                return self.penalty_break_fixation
        elif cur_epoch not in ['fixation', 'choice'] and action != 0:
            return self.penalty_hyperactive
        return 0

    def _update_state(self):
        """
        updates epoch and checks whether we're done yet
        """
        self.state['t'] += 1
        self.state['t_epoch'] += 1
        if self.state['t_epoch'] >= self.epoch_durations[self.epoch_order[self.state['epoch_index']]]:
            self.state['epoch_index'] += 1
            self.state['t_epoch'] = 0
        elif self.state['epoch_index'] == len(self.epoch_order) - 2 and self.choice_made:
            self.state['epoch_index'] += 1
            self.state['t_epoch'] = 0
        done = self.state['epoch_index'] >= len(self.epoch_order)
        return done
    
    def step(self, action, **kwargs):
        reward = self._get_reward(action)
        done = self._update_state()
        obs = self._get_obs()
        info = self._get_info()
        return obs, reward, done, False, info
