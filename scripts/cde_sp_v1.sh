#!/usr/bin/env bash

cd ..
#screen -L -dmS sp_vae_10 -Logfile ./screen_log/sp_vae_10.log python run_models.py --niters 500 -n 1000 -l 10 --dataset periodic  --sp_vae --viz -s 0.1 --noise-weight 0.01 --gpu 3 --prior gaa
#screen -L -dmS sp_vae_20 -Logfile ./screen_log/sp_vae_20.log python run_models.py --niters 500 -n 1000 -l 10 --dataset periodic  --sp_vae --viz -s 0.2 --noise-weight 0.01 --gpu 3 --prior gaa
#screen -L -dmS sp_vae_30 -Logfile ./screen_log/sp_vae_30.log python run_models.py --niters 500 -n 1000 -l 10 --dataset periodic  --sp_vae --viz -s 0.3 --noise-weight 0.01 --gpu 3 --prior gaa
screen -L -dmS sp_vae_50 -Logfile ./screen_log/sp_vae_50_1l_gac.log python run_models.py --niters 500 -n 1000 -l 10 --dataset periodic  --sp_vae --viz -s 0.5 --noise-weight 0.01 --gpu 3 --prior gac

