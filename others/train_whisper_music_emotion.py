import yaml
from pathlib import Path 
import multimodal_whisper
import torch

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
# from memory_profiler import profile

import sys
sys.path.append(str(Path(__file__).resolve().absolute().parents[2]))
# from whisper_finetune_combine_dataset_original.dataset import load_data_list
from whisper_finetune.model import WhisperModelModule#3.68it/s
# from whisper_finetune.dataset import load_data_list
# from whisper_finetune.model import WhisperModelModule
import os
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import GPUtil
import resource
# gpus = GPUtil.getGPUs()
# if len(gpus) > 0:
#     gpu = gpus[0]
#     for i in range(1, len(gpus)):
#         if gpus[i].memoryFree > gpu.memoryFree:
#             gpu = gpus[i]
#     os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu.id)
#     print(f"Using GPU {gpu.id} ({gpu.name}) with {gpu.memoryFree}MB free memory")
# else:
#     print("No GPUs found")
# resource.setrlimit(resource.RLIMIT_DATA, (5.9* 1024 * 1024 * 1024, 5.9 * 1024 * 1024 * 1024))

#vision_features/目录下印存一个字典{0:image_tensorlist,1:image_tensorlist,...}
# @profile
def train():
    # load config 
    config_path = Path("whisper_finetune/config_original.yaml")
    config = yaml.load(open(config_path, "r"), Loader=yaml.FullLoader)
    
    # dirs and paths
    in_data_dir = Path(config["path"]["preprocessed"])
    #"./preprocessed/jsut_ver1.1"=data/processed_1117/
    out_log_dir = Path(config["path"]["log"])
    #"./log"{}
    checkpoint_dir = Path(config["path"]["checkpoint"])
    #"./checkpoint" # dir to save model
    with_timestamps = bool(config["data"]["timestamps"])
    #False无时间戳
    device = "gpu" if torch.cuda.is_available() else "cpu"
    #创建模型文件夹
    out_log_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # tools
    #指定生成任务，参数：语言“ja”,是否需要时间戳
    whisper_options = multimodal_whisper.DecodingOptions(
        language=config["data"]["lang"], without_timestamps=not with_timestamps,task=config["data"]["task"]
    )
        #mel分词器，指定语言，任务：转录
    whisper_tokenizer = multimodal_whisper.tokenizer.get_tokenizer(
        True, language=config["data"]["lang"], task=whisper_options.task
    )
    #
    # list：训练集路径
    train_list = []#load_data_list(in_data_dir / "train.txt")
    val_list = [] #oad_data_list(in_data_dir / "val.txt")

    # logger日志记录？参数：日志名whisper，版本1
    tflogger = TensorBoardLogger(
        save_dir=out_log_dir,
        name=config["train_name"],
        version=config["train_id"],
    )
    if not os.path.exists('checkpoint/checkpoint'):
        os.mkdir('checkpoint/checkpoint')
    # callback回调函数，参数：模型路径checkpoint/checkpoint/checkpoint-0003
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir / "checkpoint_original_del_30s",
        filename="checkpoint-{epoch:04d}",
        save_top_k=-1, # all model save
        monitor='val/loss'
    )#['train/loss', 'val/loss', 'val/loss_epoch', 'val/cer', 'val/cer_epoch', 'val/wer', 'val/wer_epoch', 'epoch', 'step']

    callback_list = [
        checkpoint_callback, LearningRateMonitor(logging_interval="epoch")]
    #模型
    setname='open_images_train_v6_localized_narratives-00000-of-00010'
    model = WhisperModelModule(setname,config['d_model'],config['patch_dim'],config['data']['img_type'],config['path']['audio_features_dir'],config["train"],whisper_options,config["model_name"],train_list,val_list)
    # model.load_state_dict(torch.load('checkpoint/checkpoint/checkpoint-epoch=0000.ckpt')['state_dict'])
    trainer = Trainer(
        precision=16,
        accelerator=device,#gpu
        max_epochs=config["train"]["num_train_epochs"],
        accumulate_grad_batches=config["train"]["gradient_accumulation_steps"],
        logger=tflogger,
        callbacks=callback_list,
        devices=1,#1->gpu
        # gpus=[0,1]#use 2gpus
        auto_select_gpus=True
        # gpus=os.environ['CUDA_VISIBLE_DEVICES'],
        # resume_from_checkpoint='checkpoint/checkpoint/checkpoint-epoch=0000.ckpt'
        )

    trainer.fit(model)

if __name__ == "__main__":
    train()