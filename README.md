# learning_better_dale
Learning better with Dale's law: A Spectral Perspective

To run the code, first make the environment

`cd shell_scripts`

`source make_env.sh`

If env is already made, load environment

`source load_env.sh`

After the env is ready, run specific experiments using script in `shell_scripts/`

`sbatch specific_exp_scripts`

All visualizations and figures are generated using jupyter notebooks, see `analysis_in_notebooks/`

Code base is located in `lib/`; python scripts for experiments are in `exps/`

All Figures (except Fig.1) can be found in `analysis_in_notebooks/`

Figure 2 can be found in `analysis_in_notebooks/Exp_learning_curve.ipynb` and `Spectrum_Sweep_SeqMNIST_1layer.ipynb`

Figure 3 in `analysis_in_notebooks/Spectrum_simulation.ipynb`

Figure 4 in `analysis_in_notebooks/SVD_transplant.ipynb`

Figure 5,6 in `analysis_in_notebooks/Exp_learning_curve.ipynb` and `analysis_in_notebooks/SVD_entropy_analysis.ipynb`

Figure 7 in `analysis_in_notebooks/Entropy_predictive`

