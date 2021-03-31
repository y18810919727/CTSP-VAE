# CT Stochastic Process VAE - Experimental code

代码基本上从latent_ode照搬过来，构建CT-SP-VAE模型需要额外实现的代码包括：
- ```lib/stochastic_process_prior.py```高斯随机过程先验
- ```lib/create_spvae_model.py```: 创建CT-SP-VAE模型及参数设定(已经实现)
- ```lib/sp_vae.py```: CT-SP-VAE核心代码，包括如何计算多个ODE，计算loss等等，工作量较大
- ```lib/ode_func.py```: 里面新加了CDE(Controlled Differential Equations)，用来构建posterior模型，基础框架已经实现，还差微分项的计算和三次样条差值

- 详细实验规划见文档：http://202.204.54.58:82/project/60501e2319abe40074d778cc


## SPVAE-0331版本
* Toy dataset(sin curve) of 1d periodic functions for ct-sp-vae
```bash
python run_models.py --niters 500 -n 1000 -l 10 --dataset periodic  --sp_vae --viz -s 0.5 --noise-weight 0.01 --gpu 3 --prior gac
```
