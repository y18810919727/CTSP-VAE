#!/usr/bin/env bash

cd ..
screen -L -dmS ode_10 -Logfile ./screen_log/ode_10.log python run_models.py --niters 500 -n 1000 -l 10 --dataset periodic  --latent-ode --viz -s 0.1 --noise-weight 0.01 --gpu 3
screen -L -dmS ode_20 -Logfile ./screen_log/ode_20.log python run_models.py --niters 500 -n 1000 -l 10 --dataset periodic  --latent-ode --viz -s 0.2 --noise-weight 0.01 --gpu 3
screen -L -dmS ode_30 -Logfile ./screen_log/ode_30.log python run_models.py --niters 500 -n 1000 -l 10 --dataset periodic  --latent-ode --viz -s 0.3 --noise-weight 0.01 --gpu 3
screen -L -dmS ode_50 -Logfile ./screen_log/ode_50.log python run_models.py --niters 500 -n 1000 -l 10 --dataset periodic  --latent-ode --viz -s 0.5 --noise-weight 0.01 --gpu 3

