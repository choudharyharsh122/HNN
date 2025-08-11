To run the code first run the data generation scripts: 

1) For the 2D systems
$ python data_gen_2.py --dynamics_name "double_well" --q_range -2.0 2.0 --p_range -2.0 2.0 --sim_len 20 --time_step 0.01

2) For the 4D system(Henon-Heiles)
$ python data_gen_chaotic.py

The data range for Henon-Heiles system is defined within the file. We can't generate data in an arbitrary range as HH system doesn't have closed form solutions in the entire phase space.

The default noise level is 0.01 but could be changed within the data_gen_2.py file.



To train HNN and save a trained model run the following code:

python run_generic_adjoint.py --dynamics_name "double_well" --data_folder "double_well_20" --gt_res 0.01 --hid_layers "[16, 32, 16]" --solver_res 0.01 --pred True --sim_len 7 --solver "im"


This model saves the final trained models in the directory models/ and the notebook: plot_gen_data_main is used to generate the plots in the paper.

You will find the baselines in the Folder Baselines

To reproduce the results in Table 1, you have to go to notebook : NSSNN/plot_NSSNN_baseline.ipynb, rest is self explanatory, just load the model you need
to see the results for and you're good to go. Same for SHNN go to SHNN/plot_SHNN_baseline.ipynb



