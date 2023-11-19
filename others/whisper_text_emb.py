import multimodal_whisper
import torch
cuda_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.cuda.set_device(cuda_device)

tokenizer = multimodal_whisper.tokenizer.get_tokenizer(True, language='zh', task='transcribe')
model = multimodal_whisper.load_model('base')
text=''
text = [tokenizer.sot] + tokenizer.encode(text)#<sot><text>
text=torch.tensor(text).cuda()
# text=model.decoder(text,text)
text_emb=model.decoder.get_emb(text)
text_emb=model.decoder(text.unsqueeze(0),torch.tensor([]).cuda())
# text_emb=model.decoder.get_self_at_emb(text)
print(text)
