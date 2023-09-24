import numpy as np, time, torch
import random

from core import utils

from core.utils import pprint
from params import Params

from MERL.MERL import MERL

TAG_KEYS = ['seed']


if __name__ == '__main__':
    for run_i in range(2):
        params = Params()

        test_tracker = utils.Tracker(params.metric_save, [params.log_fname], '_' + str(run_i) + '.csv')
        torch.manual_seed(params.args.seed)
        np.random.seed(params.args.seed)
        random.seed(params.args.seed)

        if params.args.algorithm == 'MERL':
            # INITIALIZE THE MAIN AGENT CLASS
            ai = MERL(params)
            print('Running ', params.config.env_choice, 'with config ', params.config.config, ' State_dim:', params.state_dim,
                  'Action_dim', params.action_dim)

            time_start = time.time()

            ###### TRAINING LOOP ########
            for gen in range(1, 2006):

                print("GEN: ", gen)

                # ONE EPOCH OF TRAINING
                popn_fits, pg_fits, test_fits = ai.train(gen, test_tracker)

                # PRINT PROGRESS
                print('Ep:/Frames', gen, '/', ai.total_frames, 'Popn stat:', utils.list_stat(popn_fits), 'PG_stat:',
                      utils.list_stat(pg_fits),
                      'Test_trace:', [pprint(i) for i in ai.test_trace[-5 :]], 'FPS:',
                      pprint(ai.total_frames / (time.time() - time_start)), 'Evo', params.scheme, 'PS:', params.ps
                      )

                if gen % 5 == 0 :
                    print()
                    print('Test_stat:', utils.list_stat(test_fits), 'SAVETAG:  ', params.savetag)
                    print('Weight Stats: min/max/average', pprint(ai.test_bucket[0].get_norm_stats()))
                    print('Buffer Lens:',
                            [ag.buffer[0].__len__() for ag in ai.agents] if params.ps == 'trunk' else [ag.buffer.__len__() for
                                                                                                       ag in
                                                                                                       ai.agents])
                    print()

                if gen % 10 == 0 and params.rollout_size > 0 :
                    print()
                    print('Q', pprint(ai.agents[0].algo.q))
                    print('Q_loss',
                          (ai.agents[0].algo.q_loss))
                    print('Policy', pprint(ai.agents[0].algo.policy_loss))

                    if params.algo_name == 'TD3' :
                        print('Alz_Score', pprint(ai.agents[0].algo.alz_score))
                        print('Alz_policy', pprint(ai.agents[0].algo.alz_policy))

                    if params.algo_name == 'SAC' :
                        print('Val', pprint(ai.agents[0].algo.val))
                        print('Val_loss', pprint(ai.agents[0].algo.value_loss))
                        print('Mean_loss', pprint(ai.agents[0].algo.mean_loss))
                        print('Std_loss', pprint(ai.agents[0].algo.std_loss))

                    # Buffer Stats
                    if params.ps != 'trunk' and params.ps != 'rtrunk':
                        print('R_mean:', [agent.buffer.rstats['mean'] for agent in ai.agents])
                        print('G_mean:', [agent.buffer.gstats['mean'] for agent in ai.agents])

                    print('########################################################################')

                if ai.total_frames > params.frames_bound :
                    break

                # Kill all processes
            try :
                for p in ai.pg_task_pipes :
                    p[0].send('TERMINATE')

                for p in ai.test_task_pipes :
                    p[0].send('TERMINATE')

                for p in ai.evo_task_pipes :
                    p[0].send('TERMINATE')

            except :
                None

            print(params.args.config)
            print("DONE")
