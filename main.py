import os
from config import get_config, show_config, save_config, load_config
from data import Generator
from base import BasicScenario
from solver import REGISTRY


def run(config):
    print(f"\n{'-' * 20}    Start     {'-' * 20}\n")

    # just to Load solver info: environment and solver class from config.solver_name
    solver_info = REGISTRY.get(config.solver_name)
    print("-----------------------------1-----------------------------------------")

    #print("____________solver_info_____________",config.solver_name,"\n\n")
    #print("____________solver_info (classe slover+ environment)_____________",solver_info,"\n\n")
    #print("-----------------------------1-----------------------------------------")

    Env, Solver = solver_info['env'], solver_info['solver']
    
    #print("--------------------------2--------------------------------------------")
    #print("____________solver_info_____________",Solver,"\n\n")
    #print("____________env_____________",Env,"\n\n")
    #print("____________config_____________",config,"\n\n")

    #print("---------------------------2-------------------------------------------")

    print(f'Use {config.solver_name} Solver (Type = {solver_info["type"]})...\n')

    scenario = BasicScenario.from_config(Env, Solver, config)
    scenario.run()

    
    #print("--------------------------3--------------------------------------------")
    print("____________scenario_____________",scenario,"\n\n")
 

    #print("---------------------------3------------------------------------------")


    print(f"\n{'-' * 20}   Complete   {'-' * 20}\n")



if __name__ == '__main__':
    # Please refer to `base.loader` to obtain all available solvers

    # 1. Get Config
    # The key settings are controlled with config.py
    # while other advanced settings are listed in settings/*.yaml
    config = get_config()

    # You can modify some settings directly here.
    # An example: #a3c_gcn_seq2seq
    config.solver_name = 'ddpg' # modify the algorithm of the solver
    # config.shortest_method = 'mcf'  # modify the shortest path algorithm to Multi-commodity Flow
    config.num_train_epochs = 5   # modify the number of trainning epochs

    config.v_sim_setting_num_v_nets=100  # me
    config.v_sim_setting['num_v_nets']=100
    config.total_steps=3
    config.k_shortest=3
    config.v_sim_setting_aver_lifetime=500
    config.allow_parallel=True
    config.batch_size=32

    config.v_sim_setting['v_net_size']['high']=4

    config.p_net_setting['num_nodes']=50


    # 2. Generate Dataset
    # Although we do not generate a static dataset,
    # the environment will automatically produce a random dataset.
    p_net, v_net_simulator = Generator.generate_dataset(
        config, 
        p_net=False, 
        v_nets=False, 
        save=False) # Here, no dataset will be generated and saved.
    #print("p_net, v_net_simulator ",p_net, v_net_simulator )

    # 3. Start to Run
    # A scenario with an environment and a solver will be create following provided config.
    # The interaction between the environment and the solver will happen in this scenario.
    run(config)
