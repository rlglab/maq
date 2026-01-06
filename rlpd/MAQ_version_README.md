You can follow these instructions based on the MAQ's container, make sure that your **CUDA version is newer than 12.0** 
```bash
   1  cd rlpd
   3  cd ..
   4  pip install -r rlpd/requirements.txt 
   5  pip show jax
   6  pip install jax==0.4.14 jaxlib==0.4.14+cuda11.cudnn86 -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
   7  pip isntall optax
   8  pip install flax
   9  pip install git+https://github.com/ikostrikov/dmcgym.git
   10  pip uninstall mujoco-py
   11  pip install git+https://github.com/ikostrikov/dmcgym.git
   12  pip install git+https://github.com/ikostrikov/dmcgym.git
   13  export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/root/.mujoco/mjpro150/bin
   14  pip install git+https://github.com/ikostrikov/dmcgym.git
```

We have make small changes on loading dataset in RLPD, and here is a simple example for training RLPD on MAQ container
```bash
    XLA_PYTHON_CLIENT_PREALLOCATE=false python train_finetuning.py --env_name=<your env> \
    --utd_ratio=20 \
    --start_training 10000 \
    --max_steps 1000000 \
    --config=configs/rlpd_config.py \
    --project_name=rlpd_locomotion \
    --seed=<your see> \
    --checkpoint_model=true \
    --log_dir=/workspace/rlpd/<your log> \
    --training_dataset=<your training dataset stored in the ./offline_data>
```
