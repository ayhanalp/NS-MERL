import os
import sys

from core.utils import str2bool
from config_settings import ConfigSettings

import argparse


def get_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('-algorithm', type=str, help='#MERL', default='MERL')
    parser.add_argument('-quant', type=int, help='#Quantization: 1, 2, or 3', default=1)
    parser.add_argument('-entropy_on', type=str2bool, help='Is ENTROPY REWARD ON?', default=True)
    parser.add_argument('-popsize', type=int, help='#Evo Population size', default=10)
    parser.add_argument('-rollsize', type=int, help='#Rollout size for agents', default=50)
    parser.add_argument('-env', type=str, help='Env to test on?', default='rover_tight')
    parser.add_argument('-config', type=str, help='World Setting?', default='6_4_4_s')
    parser.add_argument('-matd3', type=str2bool, help='Use_MATD3?', default=False)
    parser.add_argument('-maddpg', type=str2bool, help='Use_MADDPG?', default=False)
    parser.add_argument('-reward', type=str, help='Reward Structure? 1. mixed 2. global', default='mixed')
    parser.add_argument('-frames', type=float, help='Frames in millions?', default=40)
    parser.add_argument('-global_w', type=float, help='Global Weight?', default=10)

    parser.add_argument('-filter_c', type=int, help='Prob multiplier for evo experiences absorbtion into buffer?',
                             default=1)
    parser.add_argument('-evals', type=int, help='#Evals to compute a fitness', default=1)  # default=1
    parser.add_argument('-seed', type=int, help='#Seed', default=2019)
    parser.add_argument('-algo', type=str, help='SAC Vs. TD3?', default='TD3')
    parser.add_argument('-savetag', help='Saved tag', default='')
    parser.add_argument('-gradperstep', type=float, help='gradient steps per frame', default=1.0)
    parser.add_argument('-pr', type=float, help='Prioritization?', default=0.0)
    parser.add_argument('-use_gpu', type=str2bool, help='USE_GPU?', default=True)  # default=False
    parser.add_argument('-alz', type=str2bool, help='Actualize?', default=True)  # default=False
    parser.add_argument('-scheme', type=str, help='Scheme?', default='multipoint')  # default='standard'
    parser.add_argument('-cmd_vel', type=str2bool, help='Switch to Velocity commands?', default=True)
    parser.add_argument('-ps', type=str,
                             help='Parameter Sharing Scheme: 1. none (heterogenous) 2. full (homogeneous) 3. trunk (shared trunk - similar to multi-headed)?',
                             default='trunk')

    return parser


class Params:
    def __init__(self):
        self.args = get_parser().parse_args()

        self.entropy_on = self.args.entropy_on

        self.quant = self.args.quant

        self.algorithm = self.args.algorithm

        # Transitive Algo Params
        self.popn_size = self.args.popsize
        self.rollout_size = self.args.rollsize
        self.num_evals = self.args.evals
        self.frames_bound = int(self.args.frames * 1000000)
        self.actualize = self.args.alz
        self.priority_rate = self.args.pr
        self.use_gpu = self.args.use_gpu
        self.seed = self.args.seed
        self.ps = self.args.ps
        self.is_matd3 = self.args.matd3
        self.is_maddpg = self.args.maddpg
        assert self.is_maddpg * self.is_matd3 == 0  # Cannot be both True

        # Env domain
        self.config = ConfigSettings(self.args, self.popn_size)

        # Fairly Stable Algo params
        self.hidden_size = 100
        self.algo_name = self.args.algo
        self.actor_lr = 5e-5
        self.critic_lr = 1e-5
        self.tau = 1e-5
        self.init_w = True
        self.gradperstep = self.args.gradperstep
        self.gamma = 0.5
        self.batch_size = 512
        self.buffer_size = 100000
        self.filter_c = self.args.filter_c
        self.reward_scaling = 10.0

        self.action_loss = False
        self.policy_ups_freq = 2
        self.policy_noise = True
        self.policy_noise_clip = 0.4

        # SAC
        self.alpha = 0.2
        self.target_update_interval = 1

        # NeuroEvolution stuff
        self.popn_size = self.args.popsize
        self.scheme = self.args.scheme  # 'multipoint' vs 'standard'
        self.crossover_prob = 0.1
        self.mutation_prob = 0.9
        self.extinction_prob = 0.005  # Probability of extinction event
        self.extinction_magnitude = 0.5  # Probabilty of extinction for each genome, given an extinction event
        self.weight_clamp = 1000000
        self.mut_distribution = 1  # 1-Gaussian, 2-Laplace, 3-Uniform
        self.lineage_depth = 10
        self.ccea_reduction = "leniency"
        self.num_anchors = 5
        self.num_elites = 4
        self.num_blends = int(0.15 * self.popn_size)

        # Dependents
        if self.config.env_choice == 'rover_loose' or self.config.env_choice == 'rover_tight' or self.config.env_choice == 'rover_trap' :  # Rover Domain
            self.state_dim = int(720 / self.config.angle_res) + 1
            if self.config.cmd_vel : self.state_dim += 2
            self.action_dim = 2
        else :
            sys.exit('Unknown Environment Choice')

        if self.config.env_choice == 'motivate' :
            self.hidden_size = 100
            self.buffer_size = 100000
            self.batch_size = 128
            self.gamma = 0.9
            self.num_anchors = 7

        self.num_test = 10
        self.test_gap = 5

        # Save Filenames
        self.savetag = self.args.savetag + \
                       'pop' + str(self.popn_size) + \
                       '_roll' + str(self.rollout_size) + \
                       '_env' + str(self.config.env_choice) + '_' + str(self.config.config) + \
                       '_ps' + str(self.ps) + \
                       '_seed' + str(self.seed) + \
                       ('_alz' if self.actualize else '') + \
                       ('_lsg' if self.config.is_lsg else '') + \
                       ('_cmdvel' if self.config.cmd_vel else '') + \
                       ('_gsl' if self.config.is_gsl else '') + \
                       ('_multipoint' if self.scheme == 'multipoint' else '')
        # '_pr' + str(self.priority_rate)
        # '_algo' + str(self.algo_name) + \
        # '_evals' + str(self.num_evals) + \
        # '_seed' + str(SEED)
        # '_filter' + str(self.filter_c)

        self.save_foldername = 'R_MERL/'
        if not os.path.exists(self.save_foldername) : os.makedirs(self.save_foldername)
        self.metric_save = self.save_foldername + 'metrics/'
        self.model_save = self.save_foldername + 'models/'
        self.aux_save = self.save_foldername + 'auxiliary/'
        if not os.path.exists(self.save_foldername) : os.makedirs(self.save_foldername)
        if not os.path.exists(self.metric_save) : os.makedirs(self.metric_save)
        if not os.path.exists(self.model_save) : os.makedirs(self.model_save)
        if not os.path.exists(self.aux_save) : os.makedirs(self.aux_save)

        self.critic_fname = 'critic_' + self.savetag
        self.actor_fname = 'actor_' + self.savetag
        self.log_fname = 'reward_' + self.savetag
        self.best_fname = 'best_' + self.savetag
