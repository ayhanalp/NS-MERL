from core.agent import Agent, TestAgent
from core.runner import rollout_worker
from params import Params as p
from core import utils

import sys, threading, random, torch

from torch.multiprocessing import Process, Pipe

RANDOM_BASELINE = False


class MERL :
    '''
    Policy Gradient Algorithm main object which carries out off-policy learning using policy gradient
           Encodes all functionalities for 1. TD3 2. DDPG 3.Trust-region TD3/DDPG 4. Advantage TD3/DDPG

                Parameters:
                    args (int): Parameter class with all the parameters
    '''

    def __init__(self, args) :
        self.params = p()
        self.args = args

        ######### Initialize the Multiagent Team of agents ########
        if self.args.ps == 'full' or self.args.ps == 'trunk':
            self.agents = [Agent(self.args, id)]

        elif self.args.ps == 'none' :
            self.agents = [Agent(self.args, id) for id in range(self.args.config.num_agents)]

        else :
            sys.exit('Incorrect PS choice')

        self.test_agent = TestAgent(self.args, 991)

        ######### Buffer and Model Bucket as references to the corresponding agent's attributes ####
        if args.ps == "trunk":
            self.buffer_bucket = [buffer.tuples for buffer in self.agents[0].buffer]
        else :
            self.buffer_bucket = [ag.buffer.tuples for ag in self.agents]

        self.popn_bucket = [ag.popn for ag in self.agents]

        self.rollout_bucket = [ag.rollout_actor for ag in self.agents]

        self.test_bucket = self.test_agent.rollout_actor

        ######### EVOLUTIONARY WORKERS ############
        if self.args.popn_size > 0 :
            self.evo_task_pipes = [Pipe() for _ in range(args.popn_size * args.num_evals)]
            self.evo_result_pipes = [Pipe() for _ in range(args.popn_size * args.num_evals)]
            self.evo_workers = [Process(target=rollout_worker, args=(
                self.args, i, 'evo', self.evo_task_pipes[i][1], self.evo_result_pipes[i][0],
                self.buffer_bucket, self.popn_bucket, True, RANDOM_BASELINE)) for i in
                                range(args.popn_size * args.num_evals)]

            for worker in self.evo_workers:
                worker.start()

        ######### POLICY GRADIENT WORKERS ############
        if self.args.rollout_size > 0 :
            self.pg_task_pipes = Pipe()
            self.pg_result_pipes = Pipe()

            self.pg_workers = [
                Process(target=rollout_worker,
                        args=(self.args, 0, 'pg', self.pg_task_pipes[1], self.pg_result_pipes[0],
                              self.buffer_bucket, self.rollout_bucket,
                              self.args.rollout_size > 0, RANDOM_BASELINE))]

            for worker in self.pg_workers:
                worker.start()

        ######### TEST WORKERS ############
        self.test_task_pipes = Pipe()
        self.test_result_pipes = Pipe()

        self.test_workers = [Process(target=rollout_worker,
                                        args=(self.args, 0, 'test', self.test_task_pipes[1], self.test_result_pipes[0],
                                            None, self.test_bucket, False, RANDOM_BASELINE))]

        for worker in self.test_workers:
            worker.start()

        #### STATS AND TRACKING WHICH ROLLOUT IS DONE ######
        self.best_score = -999
        self.total_frames = 0
        self.gen_frames = 0
        self.test_trace = []

    def make_teams(self, num_agents, popn_size, num_evals):

        temp_inds = []

        for _ in range(num_evals):
            temp_inds += list(range(popn_size))

        all_inds = [temp_inds[:] for _ in range(num_agents)]

        for entry in all_inds:
            random.shuffle(entry)

        teams = [[entry[i] for entry in all_inds] for i in range(popn_size * num_evals)]

        return teams

    def train(self, gen, test_tracker, test_tracker_I=None, test_tracker_MI=None):
        """
        Main training loop to do rollouts and run policy gradients

            Parameters:
                gen (int): Current epoch of training

            Returns:
                None
        """
        # Test Rollout
        if gen % self.args.test_gap == 0:
            self.test_agent.make_champ_team(self.agents)  # Sync the champ policies into the TestAgent
            self.test_task_pipes[0].send("START")

        # Figure out teams for Coevolution
        if self.args.ps == 'full' or self.args.ps == 'trunk' or self.args.ps == 'rtrunk':
            teams = [[i] for i in list(range(
                self.params.popn_size))]  # Homogeneous case is just the popn as a list of lists to maintain compatibility

        else :
            teams = self.make_teams(self.params.config.num_agents, self.params.popn_size,
                                    self.params.num_evals)  # Heterogeneous Case

        ########## START EVO ROLLOUT ##########
        if self.args.popn_size > 0:
            for pipe, team in zip(self.evo_task_pipes, teams):
                pipe[0].send(team)

        ########## START POLICY GRADIENT ROLLOUT ##########
        if self.args.rollout_size > 0 and not RANDOM_BASELINE :

            # Synch pg_actors to its corresponding rollout_bucket
            for agent in self.agents:
                agent.update_rollout_actor()

            # Start rollouts using the rollout actors
            self.pg_task_pipes[0].send('START')  # Index 0 for the Rollout bucket

            ############ POLICY GRADIENT UPDATES #########
            # Spin up threads for each agent
            threads = [threading.Thread(target=agent.update_parameters, args=()) for agent in self.agents]

            # Start threads
            for thread in threads:
                thread.start()

            # Join threads
            for thread in threads:
                thread.join()

        all_fits = []

        ####### JOIN EVO ROLLOUTS ########
        if self.args.popn_size > 0 :
            for pipe in self.evo_result_pipes :
                entry = pipe[1].recv()
                team = entry[0]
                fitness = entry[1][0]
                frames = entry[2]

                for agent_id, popn_id in enumerate(team):
                    self.agents[agent_id].fitnesses[popn_id].append(utils.list_mean(fitness))  # Assign

                all_fits.append(utils.list_mean(fitness))
                self.total_frames += frames

        ####### JOIN PG ROLLOUTS ########
        pg_fits = []

        if self.args.rollout_size > 0 and not RANDOM_BASELINE :
            entry = self.pg_result_pipes[1].recv()
            pg_fits = entry[1][0]
            self.total_frames += entry[2]

        ####### JOIN TEST ROLLOUTS ########
        test_fits = []

        if gen % self.args.test_gap == 0 :
            entry = self.test_result_pipes[1].recv()

            test_fits = entry[1][0]

            test_tracker.update([utils.list_mean(test_fits)], self.total_frames)
            self.test_trace.append(utils.list_mean(test_fits))

        # Evolution Step
        for agent in self.agents :
            if not agent.adv:
                agent.evolve()

        # Save models periodically
        if gen % 20 == 0 :
            for id, test_actor in enumerate(self.test_agent.rollout_actor) :
                torch.save(test_actor.state_dict(), self.args.model_save + str(id) + '_' + self.args.actor_fname)
            print("Models Saved")

        print()

        return all_fits, pg_fits, test_fits
