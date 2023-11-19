# from tensorboardX import SummaryWriter
# writer = SummaryWriter('opencpop_log')
# Code referenced from https://gist.github.com/gyglim/1f8dfb1b5c82627ae3efcfbbadb9f514
import tensorflow as tf
import numpy as np
import scipy.misc 
import random
import multimodal_whisper
from torch.optim.lr_scheduler import CosineAnnealingLR,CosineAnnealingWarmRestarts,StepLR
try:
    from StringIO import StringIO  # Python 2.7
except ImportError:
    from io import BytesIO         # Python 3.x
import os
import torch

# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
# os.environ["TORCH_USE_CUDA_DSA"] = "1"
if not os.path.exists('opencpop_log'):
    os.mkdir('opencpop_log')
if not os.path.exists('wangyi_log'):
    os.mkdir('wangyi_log')
class Logger(object):
    
    def __init__(self, log_dir):
        """Create a summary writer logging to log_dir."""
        # self.writer = tf.summary.FileWriter(log_dir)
        self.writer =tf.summary.create_file_writer(log_dir)
    # def scalar_summary(self, tag, value, step):
    #     """Log a scalar variable."""
    #     summary = tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=value)])
    #     self.writer.add_summary(summary, step)
    def scalar_summary(self, tag, value, step):
        """Log a scalar variable."""
        #summary = tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=value)])
        #self.writer.add_summary(summary, step)
        with self.writer.as_default():
            tf.summary.scalar(tag, value, step=step)
            self.writer.flush()
    def image_summary(self, tag, images, step):
        """Log a list of images."""

        img_summaries = []
        for i, img in enumerate(images):
            # Write the image to a string
            try:
                s = StringIO()
            except:
                s = BytesIO()
            scipy.misc.toimage(img).save(s, format="png")

            # Create an Image object
            img_sum = tf.Summary.Image(encoded_image_string=s.getvalue(),
                                       height=img.shape[0],
                                       width=img.shape[1])
            # Create a Summary value
            img_summaries.append(tf.Summary.Value(tag='%s/%d' % (tag, i), image=img_sum))

        # Create and write Summary
        summary = tf.Summary(value=img_summaries)
        self.writer.add_summary(summary, step)
        
    def histo_summary(self, tag, values, step, bins=1000):
        """Log a histogram of the tensor of values."""

        # Create a histogram using numpy
        counts, bin_edges = np.histogram(values, bins=bins)

        # Fill the fields of the histogram proto
        hist = tf.HistogramProto()
        hist.min = float(np.min(values))
        hist.max = float(np.max(values))
        hist.num = int(np.prod(values.shape))
        hist.sum = float(np.sum(values))
        hist.sum_squares = float(np.sum(values**2))

        # Drop the start of the first bin
        bin_edges = bin_edges[1:]

        # Add bin edges and counts
        for edge in bin_edges:
            hist.bucket_limit.append(edge)
        for c in counts:
            hist.bucket.append(c)

        # Create and write Summary
        summary = tf.Summary(value=[tf.Summary.Value(tag=tag, histo=hist)])
        self.writer.add_summary(summary, step)
        self.writer.flush()
import sys
import torchaudio
# XFORMERS_MORE_DETAILS=1
from audiocraft.models import MusicGen
# import torch
import torch.nn as nn
from torch.utils.data import DataLoader,ConcatDataset
from torch.optim import AdamW
from tqdm import tqdm
from torch.utils.data import Dataset
import numpy as np
from audiocraft.modules.conditioners import (
    ClassifierFreeGuidanceDropout
)
import timm
import timm.optim
import timm.scheduler
import os
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR, ReduceLROnPlateau
from xformers.components import attention as xf_attention
import torch.utils.data as data_utils
from sklearn.model_selection import train_test_split
from torch.nn.parallel import DataParallel
import torch.distributed as dist
import torch.multiprocessing as mp
import time
import pynvml

from torch.nn.parallel import DistributedDataParallel
cuda_device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
torch.cuda.set_device(cuda_device)
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
gpu_id=0

def monitor_gpu_temperature(gpu_id:int):
    pynvml.nvmlInit()
    gpu_count = pynvml.nvmlDeviceGetCount()
        
    while True:
        temperature_exceeds_threshold = False
        i=gpu_id
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        temperature = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
        print(f't={temperature}')
        if temperature > 62:
            temperature_exceeds_threshold = True
            

        if temperature_exceeds_threshold:
            # 如果有GPU温度超过70度
            print("GPU温度超过70度，暂停训练一分钟")
            pause_training()
        else:
            break

def pause_training():
    time.sleep(0.5)  # 暂停训
class  MusicEmotion(Dataset):
    def __init__(self, 
                data_path
                ):
        self.data_dir = data_path
        self.data_map = []
        with open(data_path,'r') as file:
            for line in file.readlines():
                line_parts = line.strip().split("\t")
                path = line_parts[0]
                emotions = line_parts[1]        
                if os.path.exists(path):
                    self.data_map.append({
                        "audio": path,
                        "label": emotions
                    })
                else:
                    raise ValueError(f'No label file for {path}')
                
    def __len__(self):
        return len(self.data_map)
    
    def __getitem__(self, idx):
        data = self.data_map[idx]
        audio = data['audio']
        label = data['label']

        return audio, label
	
class AudioDataset(Dataset):
    def __init__(self, 
                data_dir
                ):
        self.data_dir = data_dir
        self.data_map = []

        dir_map = os.listdir(os.path.join(data_dir,'cutwav'))
        for d in dir_map:
            name, ext = os.path.splitext(d)
            if ext == '.wav':
                if os.path.exists(os.path.join(data_dir, 'cutlrc',name + '.txt')):
                    self.data_map.append({
                        "audio": os.path.join(data_dir,'cutwav',d),
                        "label": os.path.join(data_dir,'cutlrc', name + '.txt')
                    })
                else:
                    raise ValueError(f'No label file for {name}')
                
    def __len__(self):
        return len(self.data_map)
    
    def __getitem__(self, idx):
        data = self.data_map[idx]
        audio = data['audio']
        label = data['label']

        return audio, label
class  Opencpop(Dataset):
    def __init__(self, 
                data_dir,datatype
                ):
        self.data_dir = data_dir
        self.data_map = []

        dir_map = os.listdir(os.path.join(data_dir,f'{datatype}_cutwav'))
        for d in dir_map:
            name, ext = os.path.splitext(d)
            if ext == '.wav':
                if os.path.exists(os.path.join(data_dir, f'{datatype}_cutlrc',name + '.txt')):
                    self.data_map.append({
                        "audio": os.path.join(data_dir,f'{datatype}_cutwav',d),
                        "label": os.path.join(data_dir,f'{datatype}_cutlrc', name + '.txt')
                    })
                else:
                    raise ValueError(f'No label file for {name}')
                
    def __len__(self):
        return len(self.data_map)
    
    def __getitem__(self, idx):
        data = self.data_map[idx]
        audio = data['audio']
        label = data['label']

        return audio, label
	
class MAGICDATA68(Dataset):
    def __init__(self, 
                data_dir,datatype
                ):
        self.data_dir = data_dir
        self.data_map = []
        datatype='dev' if datatype=='val' else 'train'
        trans_path=os.path.join(data_dir,f'{datatype}','TRANS.txt')
        with open(trans_path, 'r') as f:
            lines = f.readlines()#训练集585274，其中573480有对应文件；开发集11794->11793
            wavfiles = [line.strip().split('\t')[0] for line in lines][1:]
            subsets = [line.strip().split('\t')[1] for line in lines][1:]
        for wavfile,subset in zip(wavfiles,subsets):
            txtfile=wavfile.replace('.wav','.txt')
            if os.path.exists(os.path.join(data_dir,f'{datatype}',subset,txtfile)) and os.path.exists(os.path.join(data_dir,f'{datatype}',subset,wavfile)):
                self.data_map.append({
                    "audio": os.path.join(data_dir,f'{datatype}',subset,wavfile),
                    "label": os.path.join(data_dir,f'{datatype}',subset,txtfile)
                })
            else:
                print(f'No label file for {wavfile}')
                # raise ValueError(f'No label file for {wavfile}')
                
    def __len__(self):
        return len(self.data_map)
    
    def __getitem__(self, idx):
        data = self.data_map[idx]
        audio = data['audio']
        label = data['label']

        return audio, label

class Accompaniment(Dataset):
    def __init__(self, 
                data_dir,datatype
                ):
        self.data_dir = data_dir
        self.data_map = []
        dir_map = os.listdir(os.path.join(data_dir,f'{datatype}_cutlrc'))
        for d in dir_map:
            name, ext = os.path.splitext(d)
            if ext == '.txt':
                if os.path.exists(os.path.join(data_dir+'/cutwav',name , 'accompaniment.wav')):
                    self.data_map.append({
                        "audio": os.path.join(data_dir+'/cutwav',name , 'accompaniment.wav'),
                        "label": os.path.join(data_dir,f'{datatype}_cutlrc', name + '.txt')
                    })
                else:
                    raise ValueError(f'No label file for {name}')
                
    def __len__(self):
        return len(self.data_map)
    
    def __getitem__(self, idx):
        data = self.data_map[idx]
        audio = data['audio']
        label = data['label']

        return audio, label
class Piano(Dataset):
    def __init__(self, 
                data_dir,datatype
                ):
        self.data_dir = data_dir
        self.data_map = []
        dir_map = os.listdir(os.path.join(data_dir,f'{datatype}_cutlrc_piano'))
        for d in dir_map:
            name, ext = os.path.splitext(d)
            if ext == '.txt':
                if os.path.exists(os.path.join(data_dir+'/pianowav' , f'{name}.wav')):
                    self.data_map.append({
                        "audio": os.path.join(data_dir+'/pianowav' , f'{name}.wav'),
                        "label": os.path.join(data_dir,f'{datatype}_cutlrc_piano', name + '.txt')
                    })
                else:
                    print(f'No label file for {name}')
                
    def __len__(self):
        return len(self.data_map)
    
    def __getitem__(self, idx):
        data = self.data_map[idx]
        audio = data['audio']
        label = data['label']

        return audio, label

class TestAudioDataset(Dataset):
    def __init__(self, 
                data_dir
                ):
        self.data_dir = data_dir
        self.data_map = {'audio':[np.random.rand(1, 16000) for i in range(6)],
                         'label':['label' for i in range(6)]}#6

                
    def __len__(self):
        return len(self.data_map)
    
    def __getitem__(self, idx):
        data = self.data_map[idx]
        audio = data['audio']
        label = data['label']

        return audio, label

def count_nans(tensor):
    nan_mask = torch.isnan(tensor)
    num_nans = torch.sum(nan_mask).item()
    return num_nans

def preprocess_audio(audio_path, model: MusicGen, duration: int = 6):
    wav, sr = torchaudio.load(audio_path)
    # wav=wav.cpu()
    # wav=wav.to(torch.float16) 
    wav = torchaudio.functional.resample(wav, sr, model.sample_rate)
    wav = wav.mean(dim=0, keepdim=True)
    end_sample = int(model.sample_rate * duration)
    if wav.shape[1] <= model.sample_rate * duration:
        mask_len=model.sample_rate * duration-wav.shape[1]
        wav = torch.nn.functional.pad(wav, (0,model.sample_rate * duration-wav.shape[1] ), mode='constant', value=0.)
        mask= torch.ones_like(wav, dtype=torch.bool)
        mask[:,-mask_len:]=False
    elif wav.shape[1] > model.sample_rate * duration:
        wav = wav[:, :end_sample] 
        mask= torch.ones_like(wav, dtype=torch.bool)
    assert wav.shape[0] == 1
    assert wav.shape[1] == model.sample_rate * duration

    wav = wav
    wav = wav.unsqueeze(1)
    return wav,mask
def wav_to_codes(wav,model:MusicGen):
    with torch.no_grad():
        gen_audio = model.compression_model.encode(wav)

    codes, scale = gen_audio

    assert scale is None

    return codes

def fixnan(tensor: torch.Tensor):
    nan_mask = torch.isnan(tensor)
    result = torch.where(nan_mask, torch.zeros_like(tensor), tensor)
    
    return result

def one_hot_encode_one_sample(tensor, num_classes=2048):
    shape = tensor.shape
    one_hot = torch.zeros((shape[0], shape[1], num_classes))

    for i in range(shape[0]):
        for j in range(shape[1]):
            index = tensor[i, j].item()
            one_hot[i, j, index] = 1

    return one_hot
def one_hot_encode(tensor, num_classes=2048):
    shape = tensor.shape
    one_hot = torch.zeros((shape[0], shape[1], shape[2],num_classes))

    for i in range(shape[0]):
        for j in range(shape[1]):
            for k in range(shape[2]):
                index = tensor[i, j,k].item()
                one_hot[i, j,k,index] = 1

    return one_hot

def setup_seed(seed):
   torch.manual_seed(seed)
#    torch.cuda.manual_seed_all(seed)
   np.random.seed(seed)
   random.seed(seed)
   torch.backends.cudnn.deterministic = True
# 设置随机数种子
def build_condition_tensors(text_embs,mask):
    condition_tensors={
        'description':(text_embs.cuda(),mask.cuda())
    }
    return condition_tensors
def get_emb(texts,tokenizer,whisper_model,model):
    text_embs=torch.tensor([]).cuda()
    emb_list=[]
    max_len=0
    len_list=[]
    for txt in texts:
        txt = [tokenizer.sot] + tokenizer.encode(txt)#<sot><lang><task><notimestamps><text>
        txt=torch.tensor(txt).cuda()
        # text=model.decoder(text,text)
        with torch.no_grad():
            text_emb=whisper_model.decoder.get_emb(txt)
        emb_list.append(text_emb)
        len_list.append(len(text_emb))
        if len(text_emb)>max_len:
            max_len=len(text_emb) 
    mask=torch.ones([2*len(texts),max_len])
    for i,length in enumerate(len_list):
        mask[i,length:]=False
    mask[len(texts):,:]=False
    for text_emb in emb_list:
        pad_emb=torch.nn.functional.pad(text_emb,(0,0,0,max_len-len(text_emb)),mode='constant',value=0).unsqueeze(0)
        text_embs=torch.cat((text_embs,pad_emb),dim=0)
    text_embs=torch.nn.functional.pad(text_embs,(0,0,0,0,0,len(texts)),mode='constant',value=0.)
    with torch.set_grad_enabled(model.lm.condition_provider.conditioners['description'].finetune), model.lm.condition_provider.conditioners['description'].autocast:
        text_embs = model.lm.condition_provider.conditioners['description'].output_proj(text_embs.to(model.lm.condition_provider.conditioners['description'].output_proj.weight))
    condition_tensors=build_condition_tensors(text_embs,mask)
    return condition_tensors
# with torch.set_grad_enabled(self.finetune), self.autocast:
def train(
        dataset_path: str,
        model_id: str,
        lr: float,
        epochs: int,
        use_wandb: bool,
        save_step: int = None,
        val_step: int=None,
        accum_step=1,
        warmup_steps=1000
):
    # torch.set_default_tensor_type()
    monitor_gpu_temperature(gpu_id)
    setup_seed(3407)
    # linear=nn.linear(512,1024)
    # tokenizer = multimodal_whisper.tokenizer.get_tokenizer(True, language='zh', task='transcribe')
    # whisper_model = multimodal_whisper.load_model('base')
    # whisper_model.eval()
    tf_logger = Logger(os.path.join(sys.path[0]+'/ac_log/'))
    # tf_logger = Logger(os.path.join(sys.path[0]+'/opencpop_log/'))
    if use_wandb is True:
        import wandb
        run = wandb.init(project='audiocraft')

    model = MusicGen.get_pretrained(model_id)

    state_dict = torch.load(f"models/accompaniment/lm_41999.pt")
    # # state_dict.pop('condition_provider.conditioners.description.output_proj.weight', None)
    # # state_dict.pop('condition_provider.conditioners.description.output_proj.bias', None)
    model.lm.load_state_dict(state_dict,strict=False)
    model.lm = model.lm.to(torch.float32) #important
    # model.compression_model = model.compression_model.to(torch.float16) #important
    model.compression_model=model.compression_model.eval()
    model.lm=model.lm.train()
    
    # model.lm = nn.DataParallel(model.lm)
    # 设置分布式训练参数
    
    # backend = 'nccl'  # 或者选择 'gloo'
    # world_size = torch.cuda.device_count()
    # rank = torch.distributed.get_rank()
    # dist.init_process_group(backend=backend)
    
    # # 将模型放置在多个 GPU 上
    # model = DistributedDataParallel(model)    
    # model.lm = DistributedDataParallel(model.lm)    
    # model.compression_model = DistributedDataParallel(model.compression_model)    
    
    # model.lm.condition_provider.conditioners['description'].output_proj = nn.Linear(in_features=512, out_features=1024, bias=True).cuda()
    #musicemotion
    # dataset = MusicEmotion(dataset_path)
    # train_dataset, val_dataset = train_test_split(dataset, test_size=50, random_state=42)
    train_dataset1 = Accompaniment(dataset_path,'train')
    train_dataset = Piano(dataset_path,'train')
    train_dataset = ConcatDataset([train_dataset1, train_dataset])
    # train_dataset = MAGICDATA68(dataset_path,'train')
    # train_dataset = Opencpop(dataset_path,'train')
    # train_dataset = AudioDataset(dataset_path)
    # split_index = 10000
    # train_dataset = data_utils.Subset(train_dataset, range(split_index))
    train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True,num_workers=5)
    

    val_dataset1 = Accompaniment(dataset_path,'val')
    val_dataset = Piano(dataset_path,'val')
    val_dataset = ConcatDataset([val_dataset1, val_dataset])
    
    # val_dataset = MAGICDATA68(dataset_path,'val')
    # val_dataset = Opencpop(dataset_path,'val')
    # val_dataset = AudioDataset(dataset_path)
    # split_index = 300
    # val_dataset = data_utils.Subset(val_dataset, range(split_index))

    val_dataloader = DataLoader(val_dataset, batch_size=8, shuffle=False,num_workers=5)
    
    learning_rate = lr
    scaler = torch.cuda.amp.GradScaler()

    #from paper
    optimizer = AdamW(model.lm.parameters(), lr=learning_rate, betas=(0.9, 0.95), weight_decay=0.1)
    # optimizer = torch.optim.SGD(model.lm.parameters(), lr=learning_rate)
    lr_scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-7)
    warmup_scheduler = LambdaLR(optimizer, lr_lambda=lambda s: min(s / warmup_steps, 1))
    warmup_cos_scheduler=timm.scheduler.CosineLRScheduler(optimizer=optimizer,
                                                          t_initial=epochs,
                                                          lr_min=1e-8,
                                                          warmup_t=0.1,
                                                          warmup_lr_init=1e-8)
    cos_scheduler = CosineAnnealingWarmRestarts(optimizer,T_0=1,T_mult=2,eta_min=0)
    # ema = torch.optim.swa_utils.MovingAverage(model.parameters(), average_decay=0.99)

    criterion = nn.CrossEntropyLoss()
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    num_epochs = epochs

    save_step = save_step
    save_models = False if save_step is None else True

    save_path = "models/"+dataset_path.split('/')[-1]
    
    os.makedirs(save_path, exist_ok=True)

    current_step = 0
    for epoch in range(num_epochs):

        for batch_idx, (audiopath, labels) in enumerate(tqdm(train_dataloader)):
            monitor_gpu_temperature(gpu_id)
            # warmup_cos_scheduler.step(epoch+batch_idx/len(train_dataloader))        
            # cos_scheduler.step((epoch+batch_idx/len(train_dataloader))*15/2)  #t0=2/15 epoch      
            cos_scheduler.step((epoch*len(train_dataloader)+batch_idx)/2000) #t0为2000step
            optimizer.zero_grad()
            #音频编码器
            wavs=torch.tensor([])
            audio_mask=torch.tensor([])
            for audio,label in zip(audiopath, labels):
                wav,pad_mask = preprocess_audio(audio, model) #gen_code: torch.Size([1, 4, 1500]) returns tensor,B,K,T
                wavs = torch.cat((wavs, wav), dim=0)
                audio_mask = torch.cat((audio_mask, pad_mask), dim=0)
            audio_mask=audio_mask.cuda()
            wavs=wavs.cuda()
            
            audio=wav_to_codes(wavs,model)
            audio.cuda()#.to(torch.float16) 
            
            text =[ open(label,'r').read().strip() for label in labels]

            attributes, _ = model._prepare_tokens_and_attributes(text, None)

            conditions = attributes
            null_conditions = ClassifierFreeGuidanceDropout(p=1.0)(conditions)
            conditions = conditions + null_conditions
            # condition_tensors=get_emb(text,tokenizer,whisper_model,model)
            tokenized = model.lm.condition_provider.tokenize(conditions)
            cfg_conditions = model.lm.condition_provider(tokenized)
            condition_tensors = cfg_conditions#text_emb

            codes = torch.cat([audio, audio], dim=0)#[2B,K,T]
            # codes =audio#[2B,K,T]
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                lm_output = model.lm.compute_predictions(
                    codes=codes,
                    conditions=[],
                    condition_tensors=condition_tensors
                )

                codes = audio#[B,K,T]
                mask = lm_output.mask[:int(lm_output.logits.shape[0]/2), ...]
                cond_logits, uncond_logits = lm_output.logits.split(int(lm_output.logits.shape[0]/2), dim=0)  # [B, K, T, card]
                logits = uncond_logits + (cond_logits - uncond_logits)* 0.5
                # logits = cond_logits
                # codes = one_hot_encode(codes, num_classes=2048)#[B,K,T]->[B,K,T,2048]便于计算交叉熵
                codes = codes.cuda()
                logits = logits.cuda()#torch.Size([4, 1500, 2048])
                mask = mask.cuda()
                #由于不够30s导致的mask
                pad_mask=torch.ones_like(mask)
                for i in range(pad_mask.shape[-1]):
                    for j in range(pad_mask.shape[0]):
                        pad_mask[j,:,i]=audio_mask[j,int(32000/50*i)]
                end_id=[sum(pad_m[0]) for pad_m in pad_mask]
                # tri_mask=mask[0][:,-3:]
                # for b_id in range(len(pad_mask)):
                #     pad_mask[b_id,:,end_id[b_id]-3:end_id[b_id]]=tri_mask
                for b_id in range(len(pad_mask)):
                    pad_mask[b_id,:,end_id[b_id]-4:end_id[b_id]]=False
                mask=mask*pad_mask
                
                masked_logits = logits[mask].view(-1, 2048)
                # masked_codes = codes[mask].view(-1, 2048)
                
                masked_codes=codes[mask]
                masked_codes = masked_codes.view(-1)

                loss = criterion(masked_logits,masked_codes)

            assert count_nans(masked_logits) == 0
            
            scaler.scale(loss).backward()#autocast+GradScaler混合精度训练

            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.lm.parameters(), 1.0)
            # if (current_step+1)%accum_step==0:
            scaler.step(optimizer)
            scaler.update()
            # if current_step < warmup_steps:
            #     warmup_scheduler.step()
            # else:
            #     lr_scheduler.step()   
            print(f"Epoch: {epoch+1}/{num_epochs}, Batch: {batch_idx+1}/{len(train_dataloader)}, Loss: {loss.item()}")

            if use_wandb is True:
                run.log({
                    "loss": loss.item(),
                    "step": current_step,
                })
            # writer.add_scalar('Train/Loss', loss.item(), global_step=current_step)
            tf_logger.scalar_summary('Train/Loss', loss.item(),current_step)
            lr = optimizer.param_groups[0]['lr']
            tf_logger.scalar_summary('Train/Learning rate', lr,current_step)
            if save_models:
                if (current_step+1) % save_step == 0  or lr==0.01:
                    torch.save(model.lm.state_dict(), f"{save_path}/lm_{current_step}.pt")
            if (current_step+1)%val_step==0 or lr==0.01:
                val_avg_loss=0.
                with torch.no_grad():
                    model.lm.eval()
                    for batch_idx, (audiopath, labels) in enumerate(tqdm(val_dataloader)):
                        wavs=torch.tensor([])
                        audio_mask=torch.tensor([])
                        for audio,label in zip(audiopath, labels):
                            wav,pad_mask = preprocess_audio(audio, model) #gen_code: torch.Size([1, 4, 1500]) returns tensor,B,K,T
                            wavs = torch.cat((wavs, wav), dim=0)
                            audio_mask = torch.cat((audio_mask, pad_mask), dim=0)
                        audio_mask=audio_mask.cuda()
                        wavs=wavs.cuda()
                        audio=wav_to_codes(wavs,model)
                        text =[ open(label, 'r').read().strip() for label in labels]

                        attributes, _ = model._prepare_tokens_and_attributes(text, None)

                        conditions = attributes
                        null_conditions = ClassifierFreeGuidanceDropout(p=1.0)(conditions)
                        conditions = conditions + null_conditions
                        # condition_tensors=get_emb(text,tokenizer,whisper_model,model)
                        tokenized = model.lm.condition_provider.tokenize(conditions)
                        cfg_conditions = model.lm.condition_provider(tokenized)
                        condition_tensors = cfg_conditions#text_emb

                        codes = torch.cat([audio, audio], dim=0)#[2B,K,T]
                        # codes =audio#[2B,K,T]
                        with torch.autocast(device_type="cuda", dtype=torch.float16):
                            lm_output = model.lm.compute_predictions(
                                codes=codes,
                                conditions=[],
                                condition_tensors=condition_tensors
                            )

                            codes = audio#[B,K,T]
                            mask = lm_output.mask[:int(lm_output.logits.shape[0]/2), ...]
                            cond_logits, uncond_logits = lm_output.logits.split(int(lm_output.logits.shape[0]/2), dim=0)  # [B, K, T, card]
                            # logits = uncond_logits + (cond_logits - uncond_logits) * 3
                            logits = cond_logits
                            # codes = one_hot_encode(codes, num_classes=2048)#[B,K,T]->[B,K,T,2048]便于计算交叉熵
                            codes = codes.cuda()
                            logits = logits.cuda()#torch.Size([4, 1500, 2048])
                            mask = mask.cuda()
                            #由于不够30s导致的mask
                            pad_mask=torch.ones_like(mask)
                            for i in range(pad_mask.shape[-1]):
                                for j in range(pad_mask.shape[0]):
                                    pad_mask[j,:,i]=audio_mask[j,int(32000/50*i)]
                            end_id=[sum(pad_m[0]) for pad_m in pad_mask]
                            # tri_mask=mask[0][:,-3:]
                            # for b_id in range(len(pad_mask)):
                            #     pad_mask[b_id,:,end_id[b_id]-3:end_id[b_id]]=tri_mask
                            for b_id in range(len(pad_mask)):
                                pad_mask[b_id,:,end_id[b_id]-4:end_id[b_id]]=False
                            mask=mask*pad_mask
                            
                            masked_logits = logits[mask].view(-1, 2048)
                            # masked_codes = codes[mask].view(-1, 2048)
                            
                            masked_codes=codes[mask]
                            masked_codes = masked_codes.view(-1)

                            val_loss = criterion(masked_logits,masked_codes)
                                
                                
                        
                        # wavs=torch.tensor([])
                        # audio_mask=torch.tensor([])
                        # for audio,label in zip(audiopath, labels):
                        #     wav,pad_mask = preprocess_audio(audio, model) #gen_code: torch.Size([1, 4, 1500]) returns tensor,B,K,T
                        #     wavs = torch.cat((wavs, wav), dim=0)
                        #     audio_mask = torch.cat((audio_mask, pad_mask), dim=0)
                        # audio_mask=audio_mask.cuda()
                        # wavs=wavs.cuda()
                        # audio=wav_to_codes(wavs,model)
                        # text =[ open(label, 'r').read().strip() for label in labels]

                        # attributes, _ = model._prepare_tokens_and_attributes(text, None)

                        # conditions = attributes
                        # null_conditions = ClassifierFreeGuidanceDropout(p=1.0)(conditions)
                        # conditions = conditions + null_conditions
                        # # conditions = [[condition] + [null_condition] for (condition, null_condition) in zip(conditions, null_conditions)]
                        # tokenized = model.lm.condition_provider.tokenize(conditions)
                        # # tokenizeds=[model.lm.condition_provider.tokenize(condition) for condition in conditions ]
                        # cfg_conditions = model.lm.condition_provider(tokenized)
                        # # cfg_conditions=[ model.lm.condition_provider(tokenized) for tokenized in tokenizeds ]
                        # condition_tensors = cfg_conditions#text_emb

                        # codes = torch.cat([audio, audio], dim=0)#[2B,K,T]
                        # # codes =audio#[2B,K,T]
                        # with torch.autocast(device_type="cuda", dtype=torch.float16):
                        #     lm_output = model.lm.compute_predictions(
                        #         codes=codes,
                        #         conditions=[],
                        #         condition_tensors=condition_tensors
                        #     )

                        #     codes = audio#[B,K,T]
                        #     # logits = lm_output.logits[0]#[B,K,T,2048]
                        #     # mask = lm_output.mask[0]
                            
                        #     # indices = torch.arange( lm_output.logits.shape[0])
                        #     # mask_indices = indices % 2 == 0
                        #     # selected_indices = torch.nonzero(mask_indices).squeeze(1)
                        #     # logits = lm_output.logits[selected_indices, ...]
                        #     # mask = lm_output.mask[selected_indices, ...]
                        #     # logits = lm_output.logits[:int(lm_output.logits.shape[0]/2), ...]
                        #     mask = lm_output.mask[:int(lm_output.logits.shape[0]/2), ...]
                        #     cond_logits, uncond_logits = lm_output.logits.split(int(lm_output.logits.shape[0]/2), dim=0)  # [B, K, T, card]
                        #     # logits = uncond_logits + (cond_logits - uncond_logits) * 3
                        #     logits = cond_logits
                        #     codes = one_hot_encode(codes, num_classes=2048)#[B,K,T]->[B,K,T,2048]便于计算交叉熵
                        #     codes = codes.cuda()
                        #     logits = logits.cuda()#torch.Size([4, 1500, 2048])
                        #     mask = mask.cuda()
                        #     #由于不够30s导致的mask
                        #     pad_mask=torch.ones_like(mask)
                        #     for i in range(1500):
                        #         for j in range(pad_mask.shape[0]):
                        #             pad_mask[j,:,i]=audio_mask[j,int(32000/50*i)]
                        #     mask=mask*pad_mask
                            
                        #     masked_logits = logits[mask].view(-1, 2048)
                        #     masked_codes = codes[mask].view(-1, 2048)
                        #     val_loss = 10000*criterion(masked_logits,masked_codes)

                        
                        
                        
                        
                                    
                            val_avg_loss+=val_loss.item()/len(val_dataloader)
                        assert count_nans(masked_logits) == 0
                    
                        print(f"Epoch: {epoch+1}/{num_epochs}, Batch: {batch_idx+1}/{len(train_dataloader)}, Val_Loss: {val_loss.item()}")
                    tf_logger.scalar_summary('Val/Loss', val_avg_loss,current_step)
                        # writer.add_scalar('Val/Loss', val_loss.item(), global_step=current_step)
                model.lm.train()
            current_step += 1
        # lr_scheduler.step(epoch)
    torch.save(model.lm.state_dict(), f"{save_path}/lm_final.pt")
    # dist.destroy_process_group()