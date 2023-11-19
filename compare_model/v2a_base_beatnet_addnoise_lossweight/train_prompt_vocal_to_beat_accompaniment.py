import sys
sys.path.append('/data/huangrm/audioLM/musicgen_trainer')

# from tensorboardX import SummaryWriter
# writer = SummaryWriter('opencpop_log')
# Code referenced from https://gist.github.com/gyglim/1f8dfb1b5c82627ae3efcfbbadb9f514
import tensorflow as tf
import numpy as np
import scipy.misc 
import random
from torch.optim.lr_scheduler import CosineAnnealingLR,CosineAnnealingWarmRestarts,StepLR
try:
    from StringIO import StringIO  # Python 2.7
except ImportError:
    from io import BytesIO         # Python 3.x
import os
import torch
#cuda:0能用
# cuda_device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
# torch.cuda.set_device(cuda_device)
# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
# os.environ["TORCH_USE_CUDA_DSA"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
gpu_id=1
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
import torch
import torch.nn as nn
from torch.utils.data import DataLoader,ConcatDataset
from torch.optim import AdamW
from tqdm import tqdm
from torch.utils.data import Dataset
import numpy as np
import timm
import timm.optim
import timm.scheduler
import os
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR, ReduceLROnPlateau
import time
import pynvml

def monitor_gpu_temperature(gpu_id:int):
    pynvml.nvmlInit()
    gpu_count = pynvml.nvmlDeviceGetCount()
        
    while True:
        temperature_exceeds_threshold = False
        i=gpu_id
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        temperature = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
        print(f't={temperature}')
        if temperature > 66:
            temperature_exceeds_threshold = True
            

        if temperature_exceeds_threshold:
            # 如果有GPU温度超过70度
            print("GPU温度超过70度，暂停训练一分钟")
            pause_training()
        else:
            break

def pause_training():
    time.sleep(0.5)  # 暂停训


def va2wav_trim(audio_path,audio_path2, model: MusicGen, duration: int ):
    wav, sr = torchaudio.load(audio_path)
    noise = torch.randn(wav.shape)
    # 调整白噪音的幅度（可以根据需要调整）
    noise_amplitude = random.uniform(0.05, 0.3) # 调整白噪音的幅度
    wav = wav + noise_amplitude * noise
    # wav=wav.cpu()
    # wav=wav.to(torch.float16) 
    wav = torchaudio.functional.resample(wav, sr, model.sample_rate)
    wav = wav.mean(dim=0, keepdim=True)
    #trim
    if wav.shape[1] > model.sample_rate * duration:
        max_value = wav.shape[1] - int(model.sample_rate * duration)-1  # 你的上限值
        start_sample=random.randint(0, max_value)
        end_sample =start_sample+ int(model.sample_rate * duration)
        wav = wav[:,start_sample :end_sample] 
        assert wav.shape[1] == model.sample_rate * duration
    assert wav.shape[0] == 1
    assert wav.shape[1] <= model.sample_rate * duration
    wav = wav.unsqueeze(1)
    
    wav2, sr = torchaudio.load(audio_path2)
    # wav2=wav2.cpu()
    # wav2=wav2.to(torch.float16) 
    wav2 = torchaudio.functional.resample(wav2, sr, model.sample_rate)
    wav2 = wav2.mean(dim=0, keepdim=True)
    #trim
    if wav2.shape[1] > model.sample_rate * duration:
        wav2 = wav2[:,start_sample :end_sample] 
        assert wav2.shape[1] == model.sample_rate * duration
    assert wav2.shape[0] == 1
    assert wav2.shape[1] <= model.sample_rate * duration
    wav2 = wav2.unsqueeze(1)
    assert wav2.shape==wav.shape
    return wav,wav2
def wav_to_codes(wav,model:MusicGen):
    with torch.no_grad():
        gen_audio = model.compression_model.encode(wav)

    codes, scale = gen_audio

    assert scale is None

    return codes
def vocal_accompaniment_to_codes(vocal_paths,accopaniment_paths,model:MusicGen,duration=6):
    codes,masks=torch.tensor([]).to(torch.int64).cuda(),torch.tensor([]).cuda().to(torch.bool)
    for vocal_path,accopaniment_path in zip(vocal_paths,accopaniment_paths):
        vocal,accopaniment=va2wav_trim(vocal_path,accopaniment_path, model, duration)
        # vocal,start_sample=audio2wav_trim_2(vocal_path, model, duration)
        vocal=wav_to_codes(vocal.cuda(),model)
        vocal_mask=torch.ones_like(vocal, dtype=torch.bool)
        vocal_mask[...]=False
        # accopaniment,start_sample=audio2wav_trim_2(accopaniment_path, model, duration,start_sample)
        accopaniment=wav_to_codes(accopaniment.cuda(),model)
        accopaniment_mask=torch.ones_like(vocal, dtype=torch.bool)
        accopaniment_mask[...]=True
        #pad_mask
        b,K,T=vocal.shape
        pad_shape=[b,K,1]
        pad_token=torch.zeros(pad_shape).to(torch.int64)#B,K,K
        mask_token=torch.zeros(pad_shape).to(torch.bool)#B,K,K
        pad_token[...]=2048
        mask_token[...]=False
        
        # mid_pad_token=pad_token.repeat(1,1,vocal.shape[-2]).cuda()
        # end_pad_token=pad_token.repeat(1,1,duration*model.frame_rate*2-vocal.shape[-1]*2).cuda()
        # mid_mask_token=mask_token.repeat(1,1,vocal.shape[-2]).cuda()
        # end_mask_token=mask_token.repeat(1,1,duration*model.frame_rate*2-vocal.shape[-1]*2).cuda()#300-2*len
        
        
        mid_pad_token=pad_token.repeat(1,1,vocal.shape[-2]+duration*model.frame_rate-vocal.shape[-1]).cuda()#4+300-229
        end_pad_token=pad_token.repeat(1,1,duration*model.frame_rate-vocal.shape[-1]).cuda()#300-229
        mid_mask_token=mask_token.repeat(1,1,vocal.shape[-2]+duration*model.frame_rate-vocal.shape[-1]).cuda()
        end_mask_token=mask_token.repeat(1,1,duration*model.frame_rate-vocal.shape[-1]).cuda()#300-2*len

        #cat mask
        code=torch.cat([vocal,mid_pad_token,accopaniment,end_pad_token],dim=-1)
        mask=torch.cat([vocal_mask,mid_mask_token,accopaniment_mask,end_mask_token],dim=-1)
        masks=torch.cat([masks,mask],dim=0)
        codes=torch.cat([codes,code],dim=0)
    assert masks.shape[-1]==2*duration*model.frame_rate+4
    return codes,masks

def count_nans(tensor):
    nan_mask = torch.isnan(tensor)
    num_nans = torch.sum(nan_mask).item()
    return num_nans
def fixnan(tensor: torch.Tensor):
    nan_mask = torch.isnan(tensor)
    result = torch.where(nan_mask, torch.zeros_like(tensor), tensor)
    
    return result
def setup_seed(seed):
   torch.manual_seed(seed)
#    torch.cuda.manual_seed_all(seed)
   np.random.seed(seed)
   random.seed(seed)
   torch.backends.cudnn.deterministic = True
def load_txt(txt_path):
    with open(txt_path,'r') as f:
        a=f.read()
    return a
class Accompaniment(Dataset):
    def __init__(self, 
                data_dir,datatype
                ):
        self.data_dir = data_dir
        va_dir='/data/huangrm/audioLM/raw_data'
        self.data_map = []
        random.seed(42)
        dir_map = os.listdir(os.path.join(data_dir,f'{datatype}_cutlrc'))
        random.shuffle(dir_map)
        if datatype=='val':
            dir_map=dir_map[:100]
        for d in dir_map:
            name, ext = os.path.splitext(d)
            if ext == '.txt':
                if os.path.exists(os.path.join(data_dir+'/cutmp3',name , 'accompaniment.mp3')) and os.path.exists( os.path.join(va_dir+'/cutwav',f'{name}.wav')):
                    self.data_map.append({
                        "vocal": os.path.join(data_dir+'/cutmp3',name , 'vocals.mp3'),
                        "audio": os.path.join(data_dir+'/cutmp3',name , 'accompaniment.mp3'),
                        "va": os.path.join(va_dir+'/cutwav',f'{name}.wav'),
                        "label": os.path.join(data_dir,f'{datatype}_cutlrc', name + '.txt')
                        
                    })
                else:
                    raise ValueError(f'No label file for {name}')
                
    def __len__(self):
        return len(self.data_map)
    
    def __getitem__(self, idx):
        data = self.data_map[idx]
        vocal = data['vocal']
        audio = data['audio']
        va = data['va']
        label = data['label']

        return audio, label,vocal,va
def calculate_loss(codes,logits,mask,criterion):
    loss_code=[]
    for i in range(4):
        # masked_logits = logits[:,i,:,:][mask[:,i,:]].view(-1, 2048)   
        # masked_codes=codes[:,i,:][mask[:,i,:]].view(-1)        
        loss_code.append( criterion( logits[:,i,:,:][mask[:,i,:]].view(-1, 2048)  ,codes[:,i,:][mask[:,i,:]].view(-1)   ))
    return loss_code
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
    seed_i=1
    setup_seed(seed_i)
    off_step=0
    tf_logger = Logger('/data/huangrm/audioLM/musicgen_trainer/compare_model/v2a_log/v2a_beatnet_addnoise_lossweight_e_6')
    if use_wandb is True:
        import wandb
        run = wandb.init(project='audiocraft')
    save_path = "/data/huangrm/audioLM/musicgen_trainer/compare_model/models_test/v2a_beatnet_addnoise_lossweight_e_6"
    model = MusicGen.get_pretrained_beatnet(model_id)
    # state_dict = torch.load(f'{save_path}/lm_{off_step}.pt')
    # model.lm.load_state_dict(state_dict,strict=False)
    model.lm = model.lm.to(torch.float32) #important
    model.compression_model=model.compression_model.eval()
    model.lm=model.lm.train()
    train_dataset = Accompaniment(dataset_path,'train')
    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True,num_workers=5)
    val_dataset = Accompaniment(dataset_path,'val')
    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False,num_workers=5)
    learning_rate = lr
    scaler = torch.cuda.amp.GradScaler()

    #from paper
    optimizer = AdamW(model.lm.parameters(), lr=learning_rate, betas=(0.9, 0.95), weight_decay=0.1)
    cos_scheduler = CosineAnnealingWarmRestarts(optimizer,T_0=1,T_mult=2,eta_min=0)
    criterion = nn.CrossEntropyLoss()
    num_epochs = epochs
    save_step = save_step
    save_models = False if save_step is None else True
    os.makedirs(save_path, exist_ok=True)
    current_step = 0
    for epoch in range(num_epochs):
        seed_i+=1
        setup_seed(seed_i)
        for batch_idx, (audiopath, labels,vocalpath,va) in enumerate(tqdm(train_dataloader)):
            if current_step<off_step:
                current_step+=1
                continue
            monitor_gpu_temperature(gpu_id)
            cos_scheduler.step((epoch*len(train_dataloader)+batch_idx)/2000) #t0为2000step
            optimizer.zero_grad()
            #text
            text =[ load_txt(label).strip() for label in labels]
            attributes, _ = model._prepare_tokens_and_attributes(text, None)
            conditions = attributes
            #训练无条件生成和有条件生成
            # null_conditions = ClassifierFreeGuidanceDropout(p=1.0)(conditions)
            # conditions = conditions + null_conditions
            #只训练有条件生成
            tokenized = model.lm.condition_provider.tokenize(conditions)
            cfg_conditions = model.lm.condition_provider(tokenized)
            condition_tensors = cfg_conditions#text_emb
            # audio
            codes,pad_mask  =vocal_accompaniment_to_codes(vocalpath,audiopath,model)   
            #train  
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                lm_output, beat_logits,beat= model.lm.compute_predictions_beatnet(
                    codes=codes,
                    conditions=[],
                    condition_tensors=condition_tensors,
                    audiopaths=va
                )

                mask = lm_output.mask#[:int(lm_output.logits.shape[0]), ...]
                cond_logits = lm_output.logits#.split(int(lm_output.logits.shape[0]/2), dim=0)  # [B, K, T, card]
                # logits = uncond_logits + (cond_logits - uncond_logits) * 3
                logits = cond_logits
                logits = logits.cuda()#torch.Size([4, 1500, 2048])
                mask = mask.cuda()
                end_id=[sum(pad_m[0])+4+300 for pad_m in pad_mask]
                for b_id in range(len(pad_mask)):
                    pad_mask[b_id,:,end_id[b_id]-4:end_id[b_id]]=False
                mask=pad_mask
                
                # masked_logits = logits[mask].view(-1, 2048)
                masked_beat_logits = beat_logits[mask[:,0,:]].view(-1, 5)
                
                # masked_codes=codes[mask].view(-1)
                masked_beat =beat[mask[:,0,:]].view(-1)
                loss_code=calculate_loss(codes,logits,mask,criterion)
                loss_code=loss_code[0]*0.4+loss_code[1]*0.3+loss_code[2]*0.2+loss_code[3]*0.1
                # loss1 = criterion(masked_logits,masked_codes)
                loss_beat = criterion(masked_beat_logits,masked_beat)
                loss=loss_code+loss_beat
            loss.backward()
            optimizer.step()
            torch.nn.utils.clip_grad_norm_(model.lm.parameters(),0.5)
            print(f"Epoch: {epoch+1}/{num_epochs}, Batch: {batch_idx+1}/{len(train_dataloader)}, Loss: {loss_code.item()}")

            if use_wandb is True:
                run.log({
                    "loss": loss.item(),
                    "step": current_step,
                })
            # writer.add_scalar('Train/Loss', loss.item(), global_step=current_step)
            tf_logger.scalar_summary('Train/Loss', loss_code.item(),current_step)
            tf_logger.scalar_summary('Train/Loss_beat', loss_beat.item(),current_step)
            lr = optimizer.param_groups[0]['lr']
            tf_logger.scalar_summary('Train/Learning rate', lr,current_step)
            if save_models:
                if (current_step+1) % save_step == 0  or lr==0.01:
                    torch.save(model.lm.state_dict(), f"{save_path}/lm_{current_step}.pt")
                    print(f'save {save_path}')
            if (current_step+1)%val_step==0 or lr==0.01:
                val_avg_loss=0.
                val_avg_loss_beat=0.
                with torch.no_grad():
                    model.lm.eval()
                    for batch_idx, (audiopath, labels,vocalpath,va) in enumerate(tqdm(val_dataloader)):
                        monitor_gpu_temperature(gpu_id)
                        cos_scheduler.step((epoch*len(train_dataloader)+batch_idx)/2000) #t0为2000step
                        optimizer.zero_grad()
                        #text
                        text =[ load_txt(label).strip() for label in labels]
                        attributes, _ = model._prepare_tokens_and_attributes(text, None)
                        conditions = attributes
                        #训练无条件生成和有条件生成
                        # null_conditions = ClassifierFreeGuidanceDropout(p=1.0)(conditions)
                        # conditions = conditions + null_conditions
                        #只训练有条件生成
                        tokenized = model.lm.condition_provider.tokenize(conditions)
                        cfg_conditions = model.lm.condition_provider(tokenized)
                        condition_tensors = cfg_conditions#text_emb
                        # audio
                        codes,pad_mask  =vocal_accompaniment_to_codes(vocalpath,audiopath,model)   
                        #train  
                        with torch.autocast(device_type="cuda", dtype=torch.float16):
                            lm_output, beat_logits,beat= model.lm.compute_predictions_beatnet(
                                codes=codes,
                                conditions=[],
                                condition_tensors=condition_tensors,
                                audiopaths=va
                            )

                            mask = lm_output.mask#[:int(lm_output.logits.shape[0]), ...]
                            cond_logits = lm_output.logits#.split(int(lm_output.logits.shape[0]/2), dim=0)  # [B, K, T, card]
                            # logits = uncond_logits + (cond_logits - uncond_logits) * 3
                            logits = cond_logits
                            logits = logits.cuda()#torch.Size([4, 1500, 2048])
                            mask = mask.cuda()
                            end_id=[sum(pad_m[0])+4+300 for pad_m in pad_mask]
                            for b_id in range(len(pad_mask)):
                                pad_mask[b_id,:,end_id[b_id]-4:end_id[b_id]]=False
                            mask=pad_mask
                            
                            # masked_logits = logits[mask].view(-1, 2048)
                            masked_beat_logits = beat_logits[mask[:,0,:]].view(-1, 5)
                            
                            # masked_codes=codes[mask].view(-1)
                            masked_beat =beat[mask[:,0,:]].view(-1)
                            loss1=calculate_loss(codes,logits,mask,criterion)
                            loss1=loss_code[0]*0.4+loss_code[1]*0.3+loss_code[2]*0.2+loss_code[3]*0.1
                            # loss1 = criterion(masked_logits,masked_codes)
                            loss_beat = criterion(masked_beat_logits,masked_beat)
                            
                            val_loss=loss1+loss_beat
                            val_avg_loss+=loss1.item()/len(val_dataloader)
                            val_avg_loss_beat+=loss_beat.item()/len(val_dataloader)
                        # assert count_nans(masked_logits) == 0
                    
                        print(f"Epoch: {epoch+1}/{num_epochs}, Batch: {batch_idx+1}/{len(train_dataloader)}, Val_Loss: {loss1.item()}")
                    tf_logger.scalar_summary('Val/Loss', val_avg_loss,current_step)
                    tf_logger.scalar_summary('Val/Loss_beat', val_avg_loss_beat,current_step)
                    print(f'step={current_step},Val/Loss={val_avg_loss}')
                        # writer.add_scalar('Val/Loss', val_loss.item(), global_step=current_step)
                model.lm.train()
            current_step += 1
        # lr_scheduler.step(epoch)
    torch.save(model.lm.state_dict(), f"{save_path}/lm_final.pt")
    # dist.destroy_process_group()