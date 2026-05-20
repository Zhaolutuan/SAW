# ToolRL-GDPO

Tool-use RL training on top of [verl](https://github.com/volcengine/verl), with four
advantage estimators for decoupled correctness / format rewards.

## Installation
Please install torch, vllm and ray according to your own environment configuration.
```
# install torch
pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu121
# install vllm
pip install vllm==0.6.3
pip install ray
```

Please further install the verl in the current project and flash attention.
```
# verl
cd verl
pip install -e .

# flash attention 2
pip install flash-attn --no-build-isolation
```

## Dataset

Clone the [ToolRL](https://github.com/qiancheng0/ToolRL) repo and copy its dataset
into this project:

```
git clone git@github.com:qiancheng0/ToolRL.git
cp -r ToolRL/dataset toolrl/
```

This puts the data under `toolrl/dataset/` (e.g. `toolrl/dataset/rlla_4k`).



## Training

Training is launched via `train_gdpo.sh`. Before running, edit the paths at the top
of the script to **absolute paths** on your machine:

```bash
export DATA_DIR="/abs/path/to/toolrl/dataset/rlla_4k"   # the copied dataset
export BASE_MODEL="Qwen/Qwen2.5-1.5B-Instruct"             # HF id or local abs path
export CKPT_DIR="/abs/path/to/toolrl/results/gdpo"      # where checkpoints go
```

Then run:

```bash
bash train_gdpo.sh
```

### Advantage estimators

The method comes in **four variants**, selected with `algorithm.adv_estimator` in
`train_gdpo.sh`:

```bash
python3 -u -m verl.trainer.main_ppo \
    algorithm.adv_estimator=gdpo \   # <-- change this to one of: grpo | gdpo | grpo-h | gdpo-h
    ...
```


To run a different variant, just change the `algorithm.adv_estimator=` line in
`train_gdpo.sh` to `grpo`, `gdpo`, `grpo-h`, or `gdpo-h`.
