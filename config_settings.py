import sys

class ConfigSettings:
    def __init__(self, parser, popnsize) :
        self.env_choice = parser.env
        config = parser.config
        self.config = config
        self.reward_scheme = parser.reward
        self.global_w = parser.global_w
        self.quant = parser.quant

        # Reward types
        self.state_only = False
        self.n_s_D = False
        self.D = False

        # Global subsumes local or vice-versa?
        ####################### NIPS EXPERIMENTS SETUP #################
        if popnsize > 0 :  #######MERL or EA
            self.is_lsg = False
            self.is_proxim_rew = True

        else :  #######TD3 or MADDPG
            if self.reward_scheme == 'mixed' :
                self.is_lsg = True
                self.is_proxim_rew = True
            elif self.reward_scheme == 'global' :
                self.is_lsg = True
                self.is_proxim_rew = False
            else :
                sys.exit('Incorrect Reward Scheme')

        self.is_gsl = False
        self.cmd_vel = parser.cmd_vel

        self.dist_only = False
        self.informed = True

        self.entropy = False  # For Particle Envs

        # ROVER DOMAIN
        if self.env_choice == 'rover_loose' or self.env_choice == 'rover_tight' or self.env_choice == 'rover_trap' :  # Rover Domain

            if config == 'two_test' :
                # Rover domain
                self.dim_x = self.dim_y = 10
                self.obs_radius = self.dim_x * 10
                self.act_dist = 2
                self.angle_res = 10
                self.num_poi = 6
                self.num_agents = 2
                self.ep_len = 50  # Default = 50
                self.poi_rand = 1
                self.coupling = 2
                self.rover_speed = 1
                self.sensor_model = 'closest'

            elif config == 'try' :
                # Rover domain
                self.poi_val_config = 5  # all ones
                self.entropy = False
                self.ent_alpha = 1.0

                self.dim_x = self.dim_y = 15
                self.obs_radius = 15
                self.act_dist = 2
                self.angle_res = 90
                self.num_poi = 2
                self.num_agents = 2
                self.ep_len = 10
                self.poi_rand = 1
                self.coupling = 2
                self.rover_speed = 1
                self.sensor_model = 'closest'

            elif config in ['6_4_1_s', '6_4_2_s', '6_4_3_s', '6_4_4_s'] :
                # Rover domain
                self.poi_val_config = 5  # all ones
                self.entropy = True
                self.ent_alpha = 1.0

                self.state_only = True

                self.dim_x = self.dim_y = 15
                self.obs_radius = 15
                self.act_dist = 2
                self.angle_res = 90
                self.num_poi = 4
                self.num_agents = 6
                self.ep_len = 50
                self.poi_rand = 1
                self.coupling = 1 if config == '6_4_1_s' else 2 if config == '6_4_2_s' else 3 if config == '6_4_3_s' else 4 if config == '6_4_4_s' else -1
                self.rover_speed = 1
                self.sensor_model = 'closest'

            elif config == '1layer_POIs_2c_V' :
                # Rover domain
                self.poi_val_config = '1layer_POIs_3coupling_ent'  # all ones
                self.entropy = True
                self.ent_alpha = 1.0

                self.V_only = True

                self.dim_x = self.dim_y = 20
                self.obs_radius = 20
                self.act_dist = 2
                self.angle_res = 90
                self.num_poi = 6
                self.num_agents = 6
                self.ep_len = 40
                self.poi_rand = 0
                self.coupling = 2
                self.rover_speed = 1
                self.sensor_model = 'closest'

            elif config == 'circular_9rovs_3c_V':
                # Rover domain
                self.poi_val_config = '1layer_POIs_3coupling_ent'  # all ones
                self.entropy = True
                self.ent_alpha = 1.0
                self.quant = 1

                self.V_only = True

                self.dim_x = self.dim_y = 20
                self.obs_radius = 20
                self.act_dist = 2
                self.angle_res = 90
                self.num_poi = 6
                self.num_agents = 9
                self.ep_len = 40
                self.poi_rand = 0
                self.coupling = 3
                self.rover_speed = 1
                self.sensor_model = 'closest'

            elif config == 'circular_12rovs_4c_V':
                # Rover domain
                self.poi_val_config = '1layer_POIs_3coupling_ent'  # all ones
                self.entropy = True
                self.ent_alpha = 1.0
                self.quant = 1

                self.V_only = True

                self.dim_x = self.dim_y = 20
                self.obs_radius = 20
                self.act_dist = 2
                self.angle_res = 90
                self.num_poi = 6
                self.num_agents = 12
                self.ep_len = 40
                self.poi_rand = 0
                self.coupling = 4
                self.rover_speed = 1
                self.sensor_model = 'closest'

            elif config == 'circular_15rovs_5c_V':
                # Rover domain
                self.poi_val_config = '1layer_POIs_3coupling_ent'  # all ones
                self.entropy = True
                self.ent_alpha = 1.0
                self.quant = 1

                self.V_only = True

                self.dim_x = self.dim_y = 20
                self.obs_radius = 20
                self.act_dist = 2
                self.angle_res = 90
                self.num_poi = 6
                self.num_agents = 15
                self.ep_len = 40
                self.poi_rand = 0
                self.coupling = 5
                self.rover_speed = 1
                self.sensor_model = 'closest'

            elif config == 'circular_18rovs_6c_V':
                # Rover domain
                self.poi_val_config = '1layer_POIs_3coupling_ent'  # all ones
                self.entropy = True
                self.ent_alpha = 1.0
                self.quant = 1

                self.V_only = True

                self.dim_x = self.dim_y = 20
                self.obs_radius = 20
                self.act_dist = 2
                self.angle_res = 90
                self.num_poi = 6
                self.num_agents = 18
                self.ep_len = 40
                self.poi_rand = 0
                self.coupling = 6
                self.rover_speed = 1
                self.sensor_model = 'closest'

            elif config == '8_1_dp_V' :
                # Rover domain
                self.poi_val_config = 'densePOIs'  # all ones
                self.entropy = True
                self.ent_alpha = 1.0

                self.V_only = True

                self.dim_x = self.dim_y = 20
                self.obs_radius = 20
                self.act_dist = 2
                self.angle_res = 90
                self.num_poi = 9
                self.num_agents = 8
                self.ep_len = 50
                self.poi_rand = 0
                self.coupling = 1
                self.rover_speed = 1
                self.sensor_model = 'closest'

            elif config == '8_3_dp_V' :
                # Rover domain
                self.poi_val_config = 'densePOIs'  # all ones
                self.entropy = True
                self.ent_alpha = 1.0

                self.V_only = True

                self.dim_x = self.dim_y = 20
                self.obs_radius = 20
                self.act_dist = 2
                self.angle_res = 90
                self.num_poi = 9
                self.num_agents = 8
                self.ep_len = 50
                self.poi_rand = 0
                self.coupling = 3
                self.rover_speed = 1
                self.sensor_model = 'closest'

            elif config == '8_5_dp_V' :
                # Rover domain
                self.poi_val_config = 'densePOIs'  # all ones
                self.entropy = True
                self.ent_alpha = 1.0

                self.V_only = True

                self.dim_x = self.dim_y = 20
                self.obs_radius = 20
                self.act_dist = 2
                self.angle_res = 90
                self.num_poi = 9
                self.num_agents = 8
                self.ep_len = 50
                self.poi_rand = 0
                self.coupling = 5
                self.rover_speed = 1
                self.sensor_model = 'closest'

            elif config == '8_7_dp_V' :
                # Rover domain
                self.poi_val_config = 'densePOIs'  # all ones
                self.entropy = True
                self.ent_alpha = 1.0

                self.V_only = True

                self.dim_x = self.dim_y = 20
                self.obs_radius = 20
                self.act_dist = 2
                self.angle_res = 90
                self.num_poi = 9
                self.num_agents = 8
                self.ep_len = 50
                self.poi_rand = 0
                self.coupling = 7
                self.rover_speed = 1
                self.sensor_model = 'closest'

            elif config == '6_4_2_s' :
                # Rover domain
                self.poi_val_config = 5  # all ones
                self.entropy = True
                self.ent_alpha = 1.0

                self.state_only = True

                self.dim_x = self.dim_y = 15
                self.obs_radius = 15
                self.act_dist = 2
                self.angle_res = 90
                self.num_poi = 4
                self.num_agents = 6
                self.ep_len = 50
                self.poi_rand = 1
                self.coupling = 2
                self.rover_speed = 1
                self.sensor_model = 'closest'

            elif config == '6_4_2_s_u' :
                # Rover domain
                self.poi_val_config = 5  # all ones
                self.entropy = True
                self.ent_alpha = 1.0

                self.informed = False

                self.state_only = True

                self.dim_x = self.dim_y = 15
                self.obs_radius = 15
                self.act_dist = 2
                self.angle_res = 90
                self.num_poi = 4
                self.num_agents = 6
                self.ep_len = 50
                self.poi_rand = 1
                self.coupling = 2
                self.rover_speed = 1
                self.sensor_model = 'closest'

            elif config == '6_4_1_merld':
                # Rover domain
                self.poi_val_config = 5  # all ones
                self.entropy = False
                self.ent_alpha = 1.0

                self.dist_only = True

                self.dim_x = self.dim_y = 15
                self.obs_radius = 15
                self.act_dist = 2
                self.angle_res = 90
                self.num_poi = 4
                self.num_agents = 6
                self.ep_len = 50
                self.poi_rand = 1
                self.coupling = 1
                self.rover_speed = 1
                self.sensor_model = 'closest'

            elif config == '6_4_2_merld':
                # Rover domain
                self.poi_val_config = 5  # all ones
                self.entropy = False
                self.ent_alpha = 1.0

                self.dist_only = True

                self.dim_x = self.dim_y = 15
                self.obs_radius = 15
                self.act_dist = 2
                self.angle_res = 90
                self.num_poi = 4
                self.num_agents = 6
                self.ep_len = 50
                self.poi_rand = 1
                self.coupling = 2
                self.rover_speed = 1
                self.sensor_model = 'closest'

            elif config == '6_4_3_s' :
                # Rover domain
                self.poi_val_config = 5  # all ones
                self.entropy = True
                self.ent_alpha = 1.0

                self.state_only = True

                self.dim_x = self.dim_y = 15
                self.obs_radius = 15
                self.act_dist = 2
                self.angle_res = 90
                self.num_poi = 4
                self.num_agents = 6
                self.ep_len = 50
                self.poi_rand = 1
                self.coupling = 3
                self.rover_speed = 1
                self.sensor_model = 'closest'

            elif config == '6_4_3_s_u' :
                # Rover domain
                self.poi_val_config = 5  # all ones
                self.entropy = True
                self.ent_alpha = 1.0

                self.informed = False

                self.state_only = True

                self.dim_x = self.dim_y = 15
                self.obs_radius = 15
                self.act_dist = 2
                self.angle_res = 90
                self.num_poi = 4
                self.num_agents = 6
                self.ep_len = 50
                self.poi_rand = 1
                self.coupling = 3
                self.rover_speed = 1
                self.sensor_model = 'closest'

            elif config == '6_4_3_merld':
                # Rover domain
                self.poi_val_config = 5  # all ones
                self.entropy = False
                self.ent_alpha = 1.0

                self.dist_only = True

                self.dim_x = self.dim_y = 15
                self.obs_radius = 15
                self.act_dist = 2
                self.angle_res = 90
                self.num_poi = 4
                self.num_agents = 6
                self.ep_len = 50
                self.poi_rand = 1
                self.coupling = 3
                self.rover_speed = 1
                self.sensor_model = 'closest'

            elif config == '8_1_dp_V' :
                # Rover domain
                self.poi_val_config = 'densePOIs'  # all ones
                self.entropy = True
                self.ent_alpha = 1.0

                self.V_only = True

                self.dim_x = self.dim_y = 20
                self.obs_radius = 20
                self.act_dist = 2
                self.angle_res = 90
                self.num_poi = 9
                self.num_agents = 8
                self.ep_len = 50
                self.poi_rand = 0
                self.coupling = 1
                self.rover_speed = 1
                self.sensor_model = 'closest'

            elif config == '8_3_dp_V' :
                # Rover domain
                self.poi_val_config = 'densePOIs'  # all ones
                self.entropy = True
                self.ent_alpha = 1.0

                self.V_only = True

                self.dim_x = self.dim_y = 20
                self.obs_radius = 20
                self.act_dist = 2
                self.angle_res = 90
                self.num_poi = 9
                self.num_agents = 8
                self.ep_len = 50
                self.poi_rand = 0
                self.coupling = 3
                self.rover_speed = 1
                self.sensor_model = 'closest'

            elif config == '8_5_dp_V' :
                # Rover domain
                self.poi_val_config = 'densePOIs'  # all ones
                self.entropy = True
                self.ent_alpha = 1.0

                self.V_only = True

                self.dim_x = self.dim_y = 20
                self.obs_radius = 20
                self.act_dist = 2
                self.angle_res = 90
                self.num_poi = 9
                self.num_agents = 8
                self.ep_len = 50
                self.poi_rand = 0
                self.coupling = 5
                self.rover_speed = 1
                self.sensor_model = 'closest'

            elif config == '8_7_dp_V' :
                # Rover domain
                self.poi_val_config = 'densePOIs'  # all ones
                self.entropy = True
                self.ent_alpha = 1.0

                self.V_only = True

                self.dim_x = self.dim_y = 20
                self.obs_radius = 20
                self.act_dist = 2
                self.angle_res = 90
                self.num_poi = 9
                self.num_agents = 8
                self.ep_len = 50
                self.poi_rand = 0
                self.coupling = 7
                self.rover_speed = 1
                self.sensor_model = 'closest'

            elif config == '8_1_dp_n_s_D' :
                # Rover domain
                self.poi_val_config = 'densePOIs'  # all ones
                self.entropy = True
                self.ent_alpha = 1.0

                self.n_s_D = True

                self.dim_x = self.dim_y = 20
                self.obs_radius = 20
                self.act_dist = 2
                self.angle_res = 90
                self.num_poi = 9
                self.num_agents = 8
                self.ep_len = 50
                self.poi_rand = 0
                self.coupling = 1
                self.rover_speed = 1
                self.sensor_model = 'closest'

            elif config == '8_3_dp_n_s_D' :
                # Rover domain
                self.poi_val_config = 'densePOIs'  # all ones
                self.entropy = True
                self.ent_alpha = 1.0

                self.n_s_D = True

                self.dim_x = self.dim_y = 20
                self.obs_radius = 20
                self.act_dist = 2
                self.angle_res = 90
                self.num_poi = 9
                self.num_agents = 8
                self.ep_len = 50
                self.poi_rand = 0
                self.coupling = 3
                self.rover_speed = 1
                self.sensor_model = 'closest'

            elif config == '8_5_dp_n_s_D' :
                # Rover domain
                self.poi_val_config = 'densePOIs'  # all ones
                self.entropy = True
                self.ent_alpha = 1.0

                self.n_s_D = True

                self.dim_x = self.dim_y = 20
                self.obs_radius = 20
                self.act_dist = 2
                self.angle_res = 90
                self.num_poi = 9
                self.num_agents = 8
                self.ep_len = 50
                self.poi_rand = 0
                self.coupling = 5
                self.rover_speed = 1
                self.sensor_model = 'closest'

            elif config == '8_7_dp_n_s_D' :
                # Rover domain
                self.poi_val_config = 'densePOIs'  # all ones
                self.entropy = True
                self.ent_alpha = 1.0

                self.n_s_D = True

                self.dim_x = self.dim_y = 20
                self.obs_radius = 20
                self.act_dist = 2
                self.angle_res = 90
                self.num_poi = 9
                self.num_agents = 8
                self.ep_len = 50
                self.poi_rand = 0
                self.coupling = 7
                self.rover_speed = 1
                self.sensor_model = 'closest'

            elif config == '8_1_dp_merl' :
                # Rover domain
                self.poi_val_config = 'densePOIs'  # all ones
                self.entropy = False
                self.ent_alpha = 1.0
                self.quant = 1

                self.dist_only = True

                self.dim_x = self.dim_y = 20
                self.obs_radius = 20
                self.act_dist = 2
                self.angle_res = 90
                self.num_poi = 9
                self.num_agents = 8
                self.ep_len = 50
                self.poi_rand = 0
                self.coupling = 1
                self.rover_speed = 1
                self.sensor_model = 'closest'

            elif config == '8_3_dp_merl' :
                # Rover domain
                self.poi_val_config = 'densePOIs'  # all ones
                self.entropy = False
                self.ent_alpha = 1.0
                self.quant = 1

                self.dist_only = True

                self.dim_x = self.dim_y = 20
                self.obs_radius = 20
                self.act_dist = 2
                self.angle_res = 90
                self.num_poi = 9
                self.num_agents = 8
                self.ep_len = 50
                self.poi_rand = 0
                self.coupling = 3
                self.rover_speed = 1
                self.sensor_model = 'closest'

            elif config == '8_5_dp_merl' :
                # Rover domain
                self.poi_val_config = 'densePOIs'  # all ones
                self.entropy = False
                self.ent_alpha = 1.0
                self.quant = 1

                self.dist_only = True

                self.dim_x = self.dim_y = 20
                self.obs_radius = 20
                self.act_dist = 2
                self.angle_res = 90
                self.num_poi = 9
                self.num_agents = 8
                self.ep_len = 50
                self.poi_rand = 0
                self.coupling = 5
                self.rover_speed = 1
                self.sensor_model = 'closest'

            elif config == '8_7_dp_merl' :
                # Rover domain
                self.poi_val_config = 'densePOIs'  # all ones
                self.entropy = False
                self.ent_alpha = 1.0
                self.quant = 1

                self.dist_only = True

                self.dim_x = self.dim_y = 20
                self.obs_radius = 20
                self.act_dist = 2
                self.angle_res = 90
                self.num_poi = 9
                self.num_agents = 8
                self.ep_len = 50
                self.poi_rand = 0
                self.coupling = 7
                self.rover_speed = 1
                self.sensor_model = 'closest'

            elif config == 'circular_3rovs_1c_s_Q1':
                # Rover domain
                self.poi_val_config = '1layer_POIs_3coupling_ent'  # all ones
                self.entropy = True
                self.ent_alpha = 1.0
                self.quant = 1

                self.state_only = True

                self.dim_x = self.dim_y = 20
                self.obs_radius = 20
                self.act_dist = 2
                self.angle_res = 90
                self.num_poi = 6
                self.num_agents = 3
                self.ep_len = 40
                self.poi_rand = 0
                self.coupling = 1
                self.rover_speed = 1
                self.sensor_model = 'closest'

            elif config == 'circular_3rovs_1c_s_cb':
                # Rover domain
                self.poi_val_config = '1layer_POIs_3coupling_ent'  # all ones
                self.entropy = True
                self.ent_alpha = 1.0
                self.quant = 1

                self.state_only = True
                self.count_based = True

                self.dim_x = self.dim_y = 20
                self.obs_radius = 20
                self.act_dist = 2
                self.angle_res = 90
                self.num_poi = 6
                self.num_agents = 3
                self.ep_len = 40
                self.poi_rand = 0
                self.coupling = 1
                self.rover_speed = 1
                self.sensor_model = 'closest'

            elif config == 'circular_6rov_2c_s_Q1':
                # Rover domain
                self.poi_val_config = '1layer_POIs_3coupling_ent'  # all ones
                self.entropy = True
                self.ent_alpha = 1.0
                self.quant = 1

                self.state_only = True

                self.dim_x = self.dim_y = 20
                self.obs_radius = 20
                self.act_dist = 2
                self.angle_res = 90
                self.num_poi = 6
                self.num_agents = 6
                self.ep_len = 40
                self.poi_rand = 0
                self.coupling = 2
                self.rover_speed = 1
                self.sensor_model = 'closest'

            elif config == 'circular_9rovs_3c_s_Q1':
                # Rover domain
                self.poi_val_config = '1layer_POIs_3coupling_ent'  # all ones
                self.entropy = True
                self.ent_alpha = 1.0
                self.quant = 1

                self.state_only = True

                self.dim_x = self.dim_y = 20
                self.obs_radius = 20
                self.act_dist = 2
                self.angle_res = 90
                self.num_poi = 6
                self.num_agents = 9
                self.ep_len = 40
                self.poi_rand = 0
                self.coupling = 3
                self.rover_speed = 1
                self.sensor_model = 'closest'

            elif config == 'circular_12rovs_4c_s_Q1':
                # Rover domain
                self.poi_val_config = '1layer_POIs_3coupling_ent'  # all ones
                self.entropy = True
                self.ent_alpha = 1.0
                self.quant = 1

                self.state_only = True

                self.dim_x = self.dim_y = 20
                self.obs_radius = 20
                self.act_dist = 2
                self.angle_res = 90
                self.num_poi = 6
                self.num_agents = 12
                self.ep_len = 40
                self.poi_rand = 0
                self.coupling = 4
                self.rover_speed = 1
                self.sensor_model = 'closest'

            elif config == 'circular_12rovs_4c_s_Alpha10':
                # Rover domain
                self.poi_val_config = '1layer_POIs_3coupling_ent'  # all ones
                self.entropy = True
                self.ent_alpha = 10.0
                self.quant = 1

                self.state_only = True

                self.dim_x = self.dim_y = 20
                self.obs_radius = 20
                self.act_dist = 2
                self.angle_res = 90
                self.num_poi = 6
                self.num_agents = 12
                self.ep_len = 40
                self.poi_rand = 0
                self.coupling = 4
                self.rover_speed = 1
                self.sensor_model = 'closest'

            elif config == 'circular_15rovs_5c_s_Q1':
                # Rover domain
                self.poi_val_config = '1layer_POIs_3coupling_ent'  # all ones
                self.entropy = True
                self.ent_alpha = 1.0
                self.quant = 1

                self.state_only = True

                self.dim_x = self.dim_y = 20
                self.obs_radius = 20
                self.act_dist = 2
                self.angle_res = 90
                self.num_poi = 6
                self.num_agents = 15
                self.ep_len = 40
                self.poi_rand = 0
                self.coupling = 5
                self.rover_speed = 1
                self.sensor_model = 'closest'

            elif config == 'circular_18rovs_6c_s_Q1':
                # Rover domain
                self.poi_val_config = '1layer_POIs_3coupling_ent'  # all ones
                self.entropy = True
                self.ent_alpha = 1.0
                self.quant = 1

                self.state_only = True

                self.dim_x = self.dim_y = 20
                self.obs_radius = 20
                self.act_dist = 2
                self.angle_res = 90
                self.num_poi = 6
                self.num_agents = 18
                self.ep_len = 40
                self.poi_rand = 0
                self.coupling = 6
                self.rover_speed = 1
                self.sensor_model = 'closest'

            elif config == 'circular_21rovs_7c_s_Q1':
                # Rover domain
                self.poi_val_config = '1layer_POIs_3coupling_ent'  # all ones
                self.entropy = True
                self.ent_alpha = 1.0
                self.quant = 1

                self.state_only = True

                self.dim_x = self.dim_y = 20
                self.obs_radius = 20
                self.act_dist = 2
                self.angle_res = 90
                self.num_poi = 6
                self.num_agents = 21
                self.ep_len = 40
                self.poi_rand = 0
                self.coupling = 7
                self.rover_speed = 1
                self.sensor_model = 'closest'

            elif config == 'circular_3rovs_1c_merl':
                # Rover domain
                self.poi_val_config = '1layer_POIs_3coupling_ent'  # all ones
                self.entropy = False
                self.ent_alpha = 1.0
                self.quant = 1

                self.dist_only = True

                self.dim_x = self.dim_y = 20
                self.obs_radius = 20
                self.act_dist = 2
                self.angle_res = 90
                self.num_poi = 6
                self.num_agents = 3
                self.ep_len = 40
                self.poi_rand = 0
                self.coupling = 1
                self.rover_speed = 1
                self.sensor_model = 'closest'

            elif config == 'circular_9rovs_3c_merl':
                # Rover domain
                self.poi_val_config = '1layer_POIs_3coupling_ent'  # all ones
                self.entropy = False
                self.ent_alpha = 1.0
                self.quant = 1

                self.dist_only = True

                self.dim_x = self.dim_y = 20
                self.obs_radius = 20
                self.act_dist = 2
                self.angle_res = 90
                self.num_poi = 6
                self.num_agents = 9
                self.ep_len = 40
                self.poi_rand = 0
                self.coupling = 3
                self.rover_speed = 1
                self.sensor_model = 'closest'

            elif config == 'circular_12rovs_4c_merl':
                # Rover domain
                self.poi_val_config = '1layer_POIs_3coupling_ent'  # all ones
                self.entropy = False
                self.ent_alpha = 1.0
                self.quant = 1

                self.dist_only = True

                self.dim_x = self.dim_y = 20
                self.obs_radius = 20
                self.act_dist = 2
                self.angle_res = 90
                self.num_poi = 6
                self.num_agents = 12
                self.ep_len = 40
                self.poi_rand = 0
                self.coupling = 4
                self.rover_speed = 1
                self.sensor_model = 'closest'

            elif config == 'circular_15rovs_5c_merl':
                # Rover domain
                self.poi_val_config = '1layer_POIs_3coupling_ent'  # all ones
                self.entropy = False
                self.ent_alpha = 1.0
                self.quant = 1

                self.dim_x = self.dim_y = 20
                self.obs_radius = 20
                self.act_dist = 2
                self.angle_res = 90
                self.num_poi = 6
                self.num_agents = 15
                self.ep_len = 40
                self.poi_rand = 0
                self.coupling = 5
                self.rover_speed = 1
                self.sensor_model = 'closest'

            elif config == 'circular_18rovs_6c_merl':
                # Rover domain
                self.poi_val_config = '1layer_POIs_3coupling_ent'  # all ones
                self.entropy = False
                self.ent_alpha = 1.0
                self.quant = 1

                self.dim_x = self.dim_y = 20
                self.obs_radius = 20
                self.act_dist = 2
                self.angle_res = 90
                self.num_poi = 6
                self.num_agents = 18
                self.ep_len = 40
                self.poi_rand = 0
                self.coupling = 6
                self.rover_speed = 1
                self.sensor_model = 'closest'

            elif config == '6_4_1_Q1_n_s_D' :
                # Rover domain
                self.poi_val_config = 5  # all ones
                self.entropy = True
                self.ent_alpha = 1.0
                self.quant = 1

                self.n_s_D = True

                self.dim_x = self.dim_y = 15
                self.obs_radius = 15
                self.act_dist = 2
                self.angle_res = 90
                self.num_poi = 4
                self.num_agents = 6
                self.ep_len = 50
                self.poi_rand = 1
                self.coupling = 1
                self.rover_speed = 1
                self.sensor_model = 'closest'

            elif config == '6_4_2_Q1_n_s_D' :
                # Rover domain
                self.poi_val_config = 5  # all ones
                self.entropy = True
                self.ent_alpha = 1.0
                self.quant = 1

                self.n_s_D = True

                self.dim_x = self.dim_y = 15
                self.obs_radius = 15
                self.act_dist = 2
                self.angle_res = 90
                self.num_poi = 4
                self.num_agents = 6
                self.ep_len = 50
                self.poi_rand = 1
                self.coupling = 2
                self.rover_speed = 1
                self.sensor_model = 'closest'

            elif config == '6_4_3_Q1_n_s_D' :
                # Rover domain
                self.poi_val_config = 5  # all ones
                self.entropy = True
                self.ent_alpha = 1.0
                self.quant = 1

                self.n_s_D = True

                self.dim_x = self.dim_y = 15
                self.obs_radius = 15
                self.act_dist = 2
                self.angle_res = 90
                self.num_poi = 4
                self.num_agents = 6
                self.ep_len = 50
                self.poi_rand = 1
                self.coupling = 3
                self.rover_speed = 1
                self.sensor_model = 'closest'

            elif config == '6_4_4_Q1_n_s_D' :
                # Rover domain
                self.poi_val_config = 5  # all ones
                self.entropy = True
                self.ent_alpha = 1.0
                self.quant = 1

                self.n_s_D = True

                self.dim_x = self.dim_y = 15
                self.obs_radius = 15
                self.act_dist = 2
                self.angle_res = 90
                self.num_poi = 4
                self.num_agents = 6
                self.ep_len = 50
                self.poi_rand = 1
                self.coupling = 4
                self.rover_speed = 1
                self.sensor_model = 'closest'

            elif config == '6_4_1_D' :
                # Rover domain
                self.poi_val_config = 5  # all ones
                self.entropy = True
                self.ent_alpha = 1.0
                self.quant = 1

                self.D = True

                self.dim_x = self.dim_y = 15
                self.obs_radius = 15
                self.act_dist = 2
                self.angle_res = 90
                self.num_poi = 4
                self.num_agents = 6
                self.ep_len = 50
                self.poi_rand = 1
                self.coupling = 1
                self.rover_speed = 1
                self.sensor_model = 'closest'

            elif config == '6_4_2_D' :
                # Rover domain
                self.poi_val_config = 5  # all ones
                self.entropy = True
                self.ent_alpha = 1.0
                self.quant = 1

                self.D = True

                self.dim_x = self.dim_y = 15
                self.obs_radius = 15
                self.act_dist = 2
                self.angle_res = 90
                self.num_poi = 4
                self.num_agents = 6
                self.ep_len = 50
                self.poi_rand = 1
                self.coupling = 2
                self.rover_speed = 1
                self.sensor_model = 'closest'

            elif config == '6_4_3_D' :
                # Rover domain
                self.poi_val_config = 5  # all ones
                self.entropy = True
                self.ent_alpha = 1.0
                self.quant = 1

                self.D = True

                self.dim_x = self.dim_y = 15
                self.obs_radius = 15
                self.act_dist = 2
                self.angle_res = 90
                self.num_poi = 4
                self.num_agents = 6
                self.ep_len = 50
                self.poi_rand = 1
                self.coupling = 3
                self.rover_speed = 1
                self.sensor_model = 'closest'

            elif config == '6_4_4_D' :
                # Rover domain
                self.poi_val_config = 5  # all ones
                self.entropy = True
                self.ent_alpha = 1.0
                self.quant = 1

                self.D = True

                self.dim_x = self.dim_y = 15
                self.obs_radius = 15
                self.act_dist = 2
                self.angle_res = 90
                self.num_poi = 4
                self.num_agents = 6
                self.ep_len = 50
                self.poi_rand = 1
                self.coupling = 4
                self.rover_speed = 1
                self.sensor_model = 'closest'

            elif config == '1layer_POIs_2c_D':
                # Rover domain
                self.poi_val_config = '1layer_POIs_3coupling_ent'  # all ones
                self.entropy = True
                self.ent_alpha = 1.0
                self.quant = 1

                self.D = True

                self.dim_x = self.dim_y = 20
                self.obs_radius = 20
                self.act_dist = 2
                self.angle_res = 90
                self.num_poi = 6
                self.num_agents = 6
                self.ep_len = 40
                self.poi_rand = 0
                self.coupling = 2
                self.rover_speed = 1
                self.sensor_model = 'closest'

            elif config == 'circular_9rovs_3c_D':
                # Rover domain
                self.poi_val_config = '1layer_POIs_3coupling_ent'  # all ones
                self.entropy = True
                self.ent_alpha = 1.0
                self.quant = 1

                self.D = True

                self.dim_x = self.dim_y = 20
                self.obs_radius = 20
                self.act_dist = 2
                self.angle_res = 90
                self.num_poi = 6
                self.num_agents = 9
                self.ep_len = 40
                self.poi_rand = 0
                self.coupling = 3
                self.rover_speed = 1
                self.sensor_model = 'closest'

            elif config == 'circular_12rovs_4c_D':
                # Rover domain
                self.poi_val_config = '1layer_POIs_3coupling_ent'  # all ones
                self.entropy = True
                self.ent_alpha = 1.0
                self.quant = 1

                self.D = True

                self.dim_x = self.dim_y = 20
                self.obs_radius = 20
                self.act_dist = 2
                self.angle_res = 90
                self.num_poi = 6
                self.num_agents = 12
                self.ep_len = 40
                self.poi_rand = 0
                self.coupling = 4
                self.rover_speed = 1
                self.sensor_model = 'closest'

            elif config == 'circular_15rovs_5c_D':
                # Rover domain
                self.poi_val_config = '1layer_POIs_3coupling_ent'  # all ones
                self.entropy = True
                self.ent_alpha = 1.0
                self.quant = 1

                self.D = True

                self.dim_x = self.dim_y = 20
                self.obs_radius = 20
                self.act_dist = 2
                self.angle_res = 90
                self.num_poi = 6
                self.num_agents = 15
                self.ep_len = 40
                self.poi_rand = 0
                self.coupling = 5
                self.rover_speed = 1
                self.sensor_model = 'closest'

            elif config == '1layer_POIs_2c_Q1_n_s_D':
                # Rover domain
                self.poi_val_config = '1layer_POIs_3coupling_ent'  # all ones
                self.entropy = True
                self.ent_alpha = 1.0
                self.quant = 1

                self.n_s_D = True

                self.dim_x = self.dim_y = 20
                self.obs_radius = 20
                self.act_dist = 2
                self.angle_res = 90
                self.num_poi = 6
                self.num_agents = 6
                self.ep_len = 40
                self.poi_rand = 0
                self.coupling = 2
                self.rover_speed = 1
                self.sensor_model = 'closest'

            elif config == '1layer_POIs_3c_Q1_n_s_D':
                # Rover domain
                self.poi_val_config = '1layer_POIs_3coupling_ent'  # all ones
                self.entropy = True
                self.ent_alpha = 1.0
                self.quant = 1

                self.n_s_D = True

                self.dim_x = self.dim_y = 20
                self.obs_radius = 20
                self.act_dist = 2
                self.angle_res = 90
                self.num_poi = 6
                self.num_agents = 6
                self.ep_len = 40
                self.poi_rand = 0
                self.coupling = 3
                self.rover_speed = 1
                self.sensor_model = 'closest'

            elif config == '1layer_POIs_4c_Q1_n_s_D':
                # Rover domain
                self.poi_val_config = '1layer_POIs_3coupling_ent'  # all ones
                self.entropy = True
                self.ent_alpha = 1.0
                self.quant = 1

                self.n_s_D = True

                self.dim_x = self.dim_y = 20
                self.obs_radius = 20
                self.act_dist = 2
                self.angle_res = 90
                self.num_poi = 6
                self.num_agents = 6
                self.ep_len = 40
                self.poi_rand = 0
                self.coupling = 4
                self.rover_speed = 1
                self.sensor_model = 'closest'

            elif config == 'circular_3rovs_1c_n_s_D_Q1':
                # Rover domain
                self.poi_val_config = '1layer_POIs_3coupling_ent'  # all ones
                self.entropy = True
                self.ent_alpha = 1.0
                self.quant = 1

                self.n_s_D = True

                self.dim_x = self.dim_y = 20
                self.obs_radius = 20
                self.act_dist = 2
                self.angle_res = 90
                self.num_poi = 6
                self.num_agents = 3
                self.ep_len = 40
                self.poi_rand = 0
                self.coupling = 1
                self.rover_speed = 1
                self.sensor_model = 'closest'

            elif config == 'circular_9rovs_3c_n_s_D_Q1':
                # Rover domain
                self.poi_val_config = '1layer_POIs_3coupling_ent'  # all ones
                self.entropy = True
                self.ent_alpha = 1.0
                self.quant = 1

                self.n_s_D = True

                self.dim_x = self.dim_y = 20
                self.obs_radius = 20
                self.act_dist = 2
                self.angle_res = 90
                self.num_poi = 6
                self.num_agents = 9
                self.ep_len = 40
                self.poi_rand = 0
                self.coupling = 3
                self.rover_speed = 1
                self.sensor_model = 'closest'

            elif config == 'circular_12rovs_4c_n_s_D_Q1':
                # Rover domain
                self.poi_val_config = '1layer_POIs_3coupling_ent'  # all ones
                self.entropy = True
                self.ent_alpha = 1.0
                self.quant = 1

                self.n_s_D = True

                self.dim_x = self.dim_y = 20
                self.obs_radius = 20
                self.act_dist = 2
                self.angle_res = 90
                self.num_poi = 6
                self.num_agents = 12
                self.ep_len = 40
                self.poi_rand = 0
                self.coupling = 4
                self.rover_speed = 1
                self.sensor_model = 'closest'

            elif config == 'circular_15rovs_5c_n_s_D_Q1':
                # Rover domain
                self.poi_val_config = '1layer_POIs_3coupling_ent'  # all ones
                self.entropy = True
                self.ent_alpha = 1.0
                self.quant = 1

                self.n_s_D = True

                self.dim_x = self.dim_y = 20
                self.obs_radius = 20
                self.act_dist = 2
                self.angle_res = 90
                self.num_poi = 6
                self.num_agents = 15
                self.ep_len = 40
                self.poi_rand = 0
                self.coupling = 5
                self.rover_speed = 1
                self.sensor_model = 'closest'

            elif config == 'circular_18rovs_6c_n_s_D_Q1':
                # Rover domain
                self.poi_val_config = '1layer_POIs_3coupling_ent'  # all ones
                self.entropy = True
                self.ent_alpha = 1.0
                self.quant = 1

                self.n_s_D = True

                self.dim_x = self.dim_y = 20
                self.obs_radius = 20
                self.act_dist = 2
                self.angle_res = 90
                self.num_poi = 6
                self.num_agents = 18
                self.ep_len = 40
                self.poi_rand = 0
                self.coupling = 6
                self.rover_speed = 1
                self.sensor_model = 'closest'

            elif config == 'circular_21rovs_7c_n_s_D_Q1':
                # Rover domain
                self.poi_val_config = '1layer_POIs_3coupling_ent'  # all ones
                self.entropy = True
                self.ent_alpha = 1.0
                self.quant = 1

                self.n_s_D = True

                self.dim_x = self.dim_y = 20
                self.obs_radius = 20
                self.act_dist = 2
                self.angle_res = 90
                self.num_poi = 6
                self.num_agents = 21
                self.ep_len = 40
                self.poi_rand = 0
                self.coupling = 7
                self.rover_speed = 1
                self.sensor_model = 'closest'

            elif config == '6_4_4_s' :
                # Rover domain
                self.poi_val_config = 5  # all ones
                self.entropy = True
                self.ent_alpha = 1.0

                self.state_only = True

                self.dim_x = self.dim_y = 15
                self.obs_radius = 15
                self.act_dist = 2
                self.angle_res = 90
                self.num_poi = 4
                self.num_agents = 6
                self.ep_len = 50
                self.poi_rand = 1
                self.coupling = 4
                self.rover_speed = 1
                self.sensor_model = 'closest'

            elif config == '6_4_4_s_u' :
                # Rover domain
                self.poi_val_config = 5  # all ones
                self.entropy = True
                self.ent_alpha = 1.0

                self.informed = False

                self.state_only = True

                self.dim_x = self.dim_y = 15
                self.obs_radius = 15
                self.act_dist = 2
                self.angle_res = 90
                self.num_poi = 4
                self.num_agents = 6
                self.ep_len = 50
                self.poi_rand = 1
                self.coupling = 4
                self.rover_speed = 1
                self.sensor_model = 'closest'

            elif config == '8_1_dp_s_u' :
                # Rover domain
                self.poi_val_config = 'densePOIs'  # all ones
                self.entropy = True
                self.ent_alpha = 1.0

                self.informed = False

                self.state_only = True

                self.dim_x = self.dim_y = 20
                self.obs_radius = 20
                self.act_dist = 2
                self.angle_res = 90
                self.num_poi = 9
                self.num_agents = 8
                self.ep_len = 50
                self.poi_rand = 0
                self.coupling = 1
                self.rover_speed = 1
                self.sensor_model = 'closest'

            elif config == '8_3_dp_s_u' :
                # Rover domain
                self.poi_val_config = 'densePOIs'  # all ones
                self.entropy = True
                self.ent_alpha = 1.0

                self.informed = False

                self.state_only = True

                self.dim_x = self.dim_y = 20
                self.obs_radius = 20
                self.act_dist = 2
                self.angle_res = 90
                self.num_poi = 9
                self.num_agents = 8
                self.ep_len = 50
                self.poi_rand = 0
                self.coupling = 3
                self.rover_speed = 1
                self.sensor_model = 'closest'

            elif config == '8_5_dp_s_u' :
                # Rover domain
                self.poi_val_config = 'densePOIs'  # all ones
                self.entropy = True
                self.ent_alpha = 1.0

                self.informed = False

                self.state_only = True

                self.dim_x = self.dim_y = 20
                self.obs_radius = 20
                self.act_dist = 2
                self.angle_res = 90
                self.num_poi = 9
                self.num_agents = 8
                self.ep_len = 50
                self.poi_rand = 0
                self.coupling = 5
                self.rover_speed = 1
                self.sensor_model = 'closest'

            elif config == '8_7_dp_s_u' :
                # Rover domain
                self.poi_val_config = 'densePOIs'  # all ones
                self.entropy = True
                self.ent_alpha = 1.0

                self.informed = False

                self.state_only = True

                self.dim_x = self.dim_y = 20
                self.obs_radius = 20
                self.act_dist = 2
                self.angle_res = 90
                self.num_poi = 9
                self.num_agents = 8
                self.ep_len = 50
                self.poi_rand = 0
                self.coupling = 7
                self.rover_speed = 1
                self.sensor_model = 'closest'

            elif config == '6_4_4_sa' :
                # Rover domain
                self.poi_val_config = 5  # all ones
                self.entropy = True
                self.ent_alpha = 1.0

                self.state_action = True

                self.dim_x = self.dim_y = 15
                self.obs_radius = 15
                self.act_dist = 2
                self.angle_res = 90
                self.num_poi = 4
                self.num_agents = 6
                self.ep_len = 50
                self.poi_rand = 1
                self.coupling = 4
                self.rover_speed = 1
                self.sensor_model = 'closest'

            elif config == '6_4_4_merld':
                # Rover domain
                self.poi_val_config = 5  # all ones
                self.entropy = False
                self.ent_alpha = 1.0

                self.dist_only = True

                self.dim_x = self.dim_y = 15
                self.obs_radius = 15
                self.act_dist = 2
                self.angle_res = 90
                self.num_poi = 4
                self.num_agents = 6
                self.ep_len = 50
                self.poi_rand = 1
                self.coupling = 4
                self.rover_speed = 1
                self.sensor_model = 'closest'

            elif config == '6_4_4_merld_cb':
                # Rover domain
                self.poi_val_config = 5  # all ones
                self.entropy = False
                self.ent_alpha = 1.0

                self.dist_only = True
                self.count_based = True

                self.dim_x = self.dim_y = 15
                self.obs_radius = 15
                self.act_dist = 2
                self.angle_res = 90
                self.num_poi = 4
                self.num_agents = 6  # Correct
                self.ep_len = 50  # Correct
                self.poi_rand = 1
                self.coupling = 4
                self.rover_speed = 1
                self.sensor_model = 'closest'

            elif config == '1layer_POIs_2c_s':
                # Rover domain
                self.poi_val_config = '1layer_POIs_3coupling_ent'  # all ones
                self.entropy = True
                self.ent_alpha = 1.0

                self.state_only = True

                self.dim_x = self.dim_y = 20
                self.obs_radius = 20
                self.act_dist = 2
                self.angle_res = 90
                self.num_poi = 6
                self.num_agents = 6
                self.ep_len = 40
                self.poi_rand = 0
                self.coupling = 2
                self.rover_speed = 1
                self.sensor_model = 'closest'

            elif config == '1layer_POIs_2c_merl':
                # Rover domain
                self.poi_val_config = '1layer_POIs_3coupling_ent'  # all ones
                self.entropy = False
                self.ent_alpha = 1.0

                self.dist_only = True

                self.dim_x = self.dim_y = 20
                self.obs_radius = 20
                self.act_dist = 2
                self.angle_res = 90
                self.num_poi = 6
                self.num_agents = 6
                self.ep_len = 40
                self.poi_rand = 0
                self.coupling = 2
                self.rover_speed = 1
                self.sensor_model = 'closest'

            elif config == '1layer_POIs_3c_s':
                # Rover domain
                self.poi_val_config = '1layer_POIs_3coupling_ent'  # all ones
                self.entropy = True
                self.ent_alpha = 1.0

                self.state_only = True

                self.dim_x = self.dim_y = 20
                self.obs_radius = 20
                self.act_dist = 2
                self.angle_res = 90
                self.num_poi = 6
                self.num_agents = 6
                self.ep_len = 40
                self.poi_rand = 0
                self.coupling = 3
                self.rover_speed = 1
                self.sensor_model = 'closest'

            elif config == '1layer_POIs_3c_merl':
                # Rover domain
                self.poi_val_config = '1layer_POIs_3coupling_ent'  # all ones
                self.entropy = False
                self.ent_alpha = 1.0

                self.dist_only = True

                self.dim_x = self.dim_y = 20
                self.obs_radius = 20
                self.act_dist = 2
                self.angle_res = 90
                self.num_poi = 6
                self.num_agents = 6
                self.ep_len = 40
                self.poi_rand = 0
                self.coupling = 3
                self.rover_speed = 1
                self.sensor_model = 'closest'

            # TEAMING
            elif config == 'teaming_2teams_4c_s':
                # Rover domain
                self.poi_val_config = 'teaming'  # all ones
                self.entropy = True
                self.ent_alpha = 1.0
                self.quant = 1

                self.state_only = True

                self.dim_x = self.dim_y = 20
                self.obs_radius = 20
                self.act_dist = 2
                self.angle_res = 90
                self.num_poi = 2
                self.num_agents = 8
                self.ep_len = 30
                self.poi_rand = 0
                self.coupling = 4
                self.rover_speed = 1
                self.sensor_model = 'closest'

            elif config == 'teaming_2teams_4c_n_s_D':
                # Rover domain
                self.poi_val_config = 'teaming'  # all ones
                self.entropy = True
                self.ent_alpha = 1.0
                self.quant = 1

                self.n_s_D = True

                self.dim_x = self.dim_y = 20
                self.obs_radius = 20
                self.act_dist = 2
                self.angle_res = 90
                self.num_poi = 2
                self.num_agents = 8
                self.ep_len = 30
                self.poi_rand = 0
                self.coupling = 4
                self.rover_speed = 1
                self.sensor_model = 'closest'

            elif config == 'teaming_2teams_4c_merl':
                # Rover domain
                self.poi_val_config = 'teaming'  # all ones
                self.entropy = False
                self.ent_alpha = 1.0
                self.quant = 1

                self.dim_x = self.dim_y = 20
                self.obs_radius = 20
                self.act_dist = 2
                self.angle_res = 90
                self.num_poi = 2
                self.num_agents = 8
                self.ep_len = 30
                self.poi_rand = 0
                self.coupling = 4
                self.rover_speed = 1
                self.sensor_model = 'closest'

            elif config == 'teaming_3teams_4c_s':
                # Rover domain
                self.poi_val_config = 'teaming'  # all ones
                self.entropy = True
                self.ent_alpha = 1.0
                self.quant = 1

                self.state_only = True

                self.dim_x = self.dim_y = 20
                self.obs_radius = 20
                self.act_dist = 2
                self.angle_res = 90
                self.num_poi = 3
                self.num_agents = 12
                self.ep_len = 30
                self.poi_rand = 0
                self.coupling = 4
                self.rover_speed = 1
                self.sensor_model = 'closest'

            elif config == 'teaming_3teams_4c_n_s_D':
                # Rover domain
                self.poi_val_config = 'teaming'  # all ones
                self.entropy = True
                self.ent_alpha = 1.0
                self.quant = 1

                self.n_s_D = True

                self.dim_x = self.dim_y = 20
                self.obs_radius = 20
                self.act_dist = 2
                self.angle_res = 90
                self.num_poi = 3
                self.num_agents = 12
                self.ep_len = 30
                self.poi_rand = 0
                self.coupling = 4
                self.rover_speed = 1
                self.sensor_model = 'closest'

            elif config == 'teaming_3teams_4c_merl':
                # Rover domain
                self.poi_val_config = 'teaming'  # all ones
                self.entropy = False
                self.ent_alpha = 1.0
                self.quant = 1

                self.dim_x = self.dim_y = 20
                self.obs_radius = 20
                self.act_dist = 2
                self.angle_res = 90
                self.num_poi = 3
                self.num_agents = 12
                self.ep_len = 30
                self.poi_rand = 0
                self.coupling = 4
                self.rover_speed = 1
                self.sensor_model = 'closest'

            elif config == 'teaming_4teams_4c_s':
                # Rover domain
                self.poi_val_config = 'teaming'  # all ones
                self.entropy = True
                self.ent_alpha = 1.0
                self.quant = 1

                self.state_only = True

                self.dim_x = self.dim_y = 20
                self.obs_radius = 20
                self.act_dist = 2
                self.angle_res = 90
                self.num_poi = 4
                self.num_agents = 16
                self.ep_len = 30
                self.poi_rand = 0
                self.coupling = 4
                self.rover_speed = 1
                self.sensor_model = 'closest'

            elif config == 'teaming_4teams_4c_n_s_D':
                # Rover domain
                self.poi_val_config = 'teaming'  # all ones
                self.entropy = True
                self.ent_alpha = 1.0
                self.quant = 1

                self.n_s_D = True

                self.dim_x = self.dim_y = 20
                self.obs_radius = 20
                self.act_dist = 2
                self.angle_res = 90
                self.num_poi = 4
                self.num_agents = 16
                self.ep_len = 30
                self.poi_rand = 0
                self.coupling = 4
                self.rover_speed = 1
                self.sensor_model = 'closest'

            elif config == 'teaming_4teams_4c_merl':
                # Rover domain
                self.poi_val_config = 'teaming'  # all ones
                self.entropy = False
                self.ent_alpha = 1.0
                self.quant = 1

                self.dim_x = self.dim_y = 20
                self.obs_radius = 20
                self.act_dist = 2
                self.angle_res = 90
                self.num_poi = 4
                self.num_agents = 16
                self.ep_len = 30
                self.poi_rand = 0
                self.coupling = 4
                self.rover_speed = 1
                self.sensor_model = 'closest'

            elif config == 'simple' :
                # Rover domain
                self.poi_val_config = 5  # all ones
                self.entropy = True
                self.ent_alpha = 1.0

                self.dim_x = self.dim_y = 10
                self.obs_radius = 2
                self.act_dist = 2
                self.angle_res = 10
                self.num_poi = 6
                self.num_agents = 3
                self.ep_len = 50
                self.poi_rand = 1
                self.coupling = 2
                self.rover_speed = 1
                self.sensor_model = 'closest'

            elif config == 'nav' :
                # Rover domain
                self.dim_x = self.dim_y = 30
                self.obs_radius = self.dim_x * 10
                self.act_dist = 2
                self.rover_speed = 1
                self.sensor_model = 'closest'
                self.angle_res = 10
                self.num_poi = 10
                self.num_agents = 1
                self.ep_len = 50
                self.poi_rand = 1
                self.coupling = 1

            elif config == '4_4' :
                # Rover domain
                self.dim_x = self.dim_y = 30
                self.obs_radius = self.dim_x * 10
                self.act_dist = 3
                self.rover_speed = 1
                self.sensor_model = 'closest'
                self.angle_res = 10
                self.num_poi = 4
                self.num_agents = 4
                self.ep_len = 50
                self.poi_rand = 1
                self.coupling = 4

            elif config == '6_3':
                # Rover domain
                self.poi_val_config = 5  # all ones
                self.entropy = False
                self.ent_alpha = 1.0

                self.dim_x = self.dim_y = 30
                self.obs_radius = self.dim_x * 10
                self.act_dist = 3
                self.rover_speed = 1
                self.sensor_model = 'closest'
                self.angle_res = 90
                self.num_poi = 4
                self.num_agents = 6
                self.ep_len = 50
                self.poi_rand = 1
                self.coupling = 3

            elif config == '6_4' :
                # Rover domain
                self.dim_x = self.dim_y = 30
                self.obs_radius = self.dim_x * 10
                self.act_dist = 3
                self.rover_speed = 1
                self.sensor_model = 'closest'
                self.angle_res = 10
                self.num_poi = 4
                self.num_agents = 6
                self.ep_len = 50
                self.poi_rand = 1
                self.coupling = 4

            elif config == '6_5' :
                # Rover domain
                self.dim_x = self.dim_y = 30
                self.obs_radius = self.dim_x * 10
                self.act_dist = 3
                self.rover_speed = 1
                self.sensor_model = 'closest'
                self.angle_res = 10
                self.num_poi = 4
                self.num_agents = 6
                self.ep_len = 50
                self.poi_rand = 1
                self.coupling = 5

            elif config == '4_4_plenty' :
                # Rover domain
                self.dim_x = self.dim_y = 30
                self.obs_radius = self.dim_x * 10
                self.act_dist = 3
                self.rover_speed = 1
                self.sensor_model = 'closest'
                self.angle_res = 10
                self.num_poi = 20
                self.num_agents = 4
                self.ep_len = 50
                self.poi_rand = 1
                self.coupling = 4

            elif config == '4_1_plenty' :
                # Rover domain
                self.dim_x = self.dim_y = 30
                self.obs_radius = self.dim_x * 10
                self.act_dist = 3
                self.rover_speed = 1
                self.sensor_model = 'closest'
                self.angle_res = 10
                self.num_poi = 16
                self.num_agents = 4
                self.ep_len = 30
                self.poi_rand = 1
                self.coupling = 1

            elif config == '4_1' :
                # Rover domain
                self.dim_x = self.dim_y = 30
                self.obs_radius = self.dim_x * 10
                self.act_dist = 3
                self.rover_speed = 1
                self.sensor_model = 'closest'
                self.angle_res = 10
                self.num_poi = 4
                self.num_agents = 4
                self.ep_len = 25
                self.poi_rand = 1
                self.coupling = 1

            elif config == '4_2' :
                # Rover domain
                self.dim_x = self.dim_y = 30
                self.obs_radius = self.dim_x * 10
                self.act_dist = 3
                self.rover_speed = 1
                self.sensor_model = 'closest'
                self.angle_res = 10
                self.num_poi = 2
                self.num_agents = 4
                self.ep_len = 30
                self.poi_rand = 1
                self.coupling = 1

            elif config == '3_1' :
                # Rover domain
                self.dim_x = self.dim_y = 30
                self.obs_radius = self.dim_x * 10
                self.act_dist = 3
                self.rover_speed = 1
                self.sensor_model = 'closest'
                self.angle_res = 10
                self.num_poi = 3
                self.num_agents = 3
                self.ep_len = 50
                self.poi_rand = 1
                self.coupling = 1

            else :
                sys.exit('Unknown Config')
            # Fix Harvest Period and coupling given some config choices

            if self.env_choice == "rover_trap" :
                self.harvest_period = 3
            else :
                self.harvest_period = 1

            if self.env_choice == "rover_loose" : self.coupling = 1  # Definiton of a Loosely coupled domain
