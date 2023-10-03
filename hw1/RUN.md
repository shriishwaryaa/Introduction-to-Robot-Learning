### Section 1 (Behavior Cloning)
Command for part 2:

```
python rob831/scripts/run_hw1.py --expert_policy_file rob831/policies/experts/Ant.pkl --env_name Ant-v2 --exp_name bc_ant --n_iter 1 --expert_data rob831/expert_data/expert_data_Ant-v2.pkl --video_log_freq -1 --eval_batch_size 5000 --n_layers 1
```

```
python rob831/scripts/run_hw1.py --expert_policy_file rob831/policies/experts/Humanoid.pkl --env_name Humanoid-v2 --exp_name bc_Humanoid --n_iter 1 --expert_data rob831/expert_data/expert_data_Humanoid-v2.pkl --video_log_freq -1 --eval_batch_size 5000 --n_layers 1
```

```
python rob831/scripts/run_hw1.py --expert_policy_file rob831/policies/experts/Walker2d.pkl --env_name Walker2d-v2 --exp_name bc_Walker2d --n_iter 1 --expert_data rob831/expert_data/expert_data_Walker2d-v2.pkl --video_log_freq -1 --eval_batch_size 5000 --n_layers 1
```

```
python rob831/scripts/run_hw1.py --expert_policy_file rob831/policies/experts/Hopper.pkl --env_name Hopper-v2 --exp_name bc_Hopper --n_iter 1 --expert_data rob831/expert_data/expert_data_Hopper-v2.pkl --video_log_freq -1 --eval_batch_size 5000 --n_layers 1
```

```
python rob831/scripts/run_hw1.py --expert_policy_file rob831/policies/experts/HalfCheetah.pkl --env_name HalfCheetah-v2 --exp_name bc_HalfCheetah --n_iter 1 --expert_data rob831/expert_data/expert_data_HalfCheetah-v2.pkl --video_log_freq -1 --eval_batch_size 5000 --n_layers 1
```

Command for part 3 can be found above for the Ant-v2 and the Hopper-v2 environments. 

### Section 2 (DAgger)
Command for part 2:

```
python rob831/scripts/run_hw1.py --expert_policy_file rob831/policies/experts/Ant.pkl --env_name Ant-v2 --exp_name dagger_ant --n_iter 10 --expert_data rob831/expert_data/expert_data_Ant-v2.pkl --video_log_freq -1 --eval_batch_size 5000 --do_dagger --n_layers 1
```

```
python rob831/scripts/run_hw1.py --expert_policy_file rob831/policies/experts/Hopper.pkl --env_name Hopper-v2 --exp_name dagger_hopper --n_iter 10 --expert_data rob831/expert_data/expert_data_Hopper-v2.pkl --video_log_freq -1 --eval_batch_size 5000 --do_dagger --n_layers 1
```