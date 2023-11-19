import sys
sys.path.append('/data/huangrm/audioLM/musicgen_trainer/compare_model/v2a_base_beatnet_addnoise_lossweight')

from train_prompt_vocal_to_beat_accompaniment import train

import argparse
import faulthandler
import torch
# 在import之后直接添加以下启用代码即可
# faulthandler.enable()
parser = argparse.ArgumentParser()
# /data/huangrm/audioLM/raw_data
# /data/huangrm/audioLM/musicgen_trainer/archivo/opencpop
# /data/huangrm/audioLM/musicgen_trainer/archivo/opencpop_overfit
# /data/huangrm/audioLM/musicgen_trainer/MAGICDATA68
# /data/huangrm/audioLM/musicgen_trainer/archivo/accompaniment
# torch.distributed.launch
# parser.add_argument('--dataset_path', type=str, required=False,default='/data/huangrm/audioLM/musicgen_trainer/archivo/keyword/__pycache__/music_emotion_data/data.txt')
parser.add_argument('--dataset_path', type=str, required=False,default='/data/huangrm/audioLM/musicgen_trainer/archivo/accompaniment')
# parser.add_argument('--dataset_path', type=str, required=False,default='/data/huangrm/audioLM/musicgen_trainer/MAGICDATA68')
# parser.add_argument('--dataset_path', type=str, required=False,default='/data/huangrm/audioLM/musicgen_trainer/archivo/opencpop_overfit')
# parser.add_argument('--dataset_path', type=str, required=False,default='/data/huangrm/audioLM/musicgen_trainer/archivo/opencpop')
# parser.add_argument('--dataset_path', type=str, required=False,default='/data/huangrm/audioLM/raw_data')
parser.add_argument('--model_id', type=str, required=False, default='small')
parser.add_argument('--lr', type=float, required=False, default=1e-6)
parser.add_argument('--epochs', type=int, required=False, default=10)
parser.add_argument('--use_wandb', type=int, required=False, default=0)
parser.add_argument('--save_step', type=int, required=False, default=2000)
parser.add_argument('--val_step', type=int, required=False, default=2000)
parser.add_argument('--accum_step', type=int, required=False, default=2)
parser.add_argument('--warmup_steps', type=int, required=False, default=2000)
parser.add_argument('--local_rank',type=int)
#使用whisper参数：cosrestart,0.05-0.02，cgd优化器，2.5
#不使用whisper：
# python -m torch.distributed.launch \
#        --nnodes 1 \
#        --nproc_per_node=2 \
#        run.py

args = parser.parse_args()

train(
    dataset_path=args.dataset_path,
    model_id=args.model_id,
    lr=args.lr,
    epochs=args.epochs,
    use_wandb=args.use_wandb,
    save_step=args.save_step,
    val_step=args.val_step,
    accum_step=args.accum_step,
    warmup_steps=args.warmup_steps
)