## spleeter生成伴奏数据集
nohup python -u archivo/accompaniment/multiprocessor_spleeter.py > archivo/accompaniment/multiprocessor_spleeter.log 2>&1 &
## train opencpop
python run.py
python -u run.py > train_opencpop.log 2>&1 &

1.改run.py的main:路径
2.改dataset类
3.改log的文件夹
4.改cuda号码
## train wangyi
python -u run.py > train_wangyi.log 2>&1 &

epoch 3 , save_step==2000, batch=8 samples100000 lr e-5(初始loss高但是可能下降快)

## train on asr corpus
python -u run.py > train_MAGICDATA68.log 2>&1 &

epoch 2,lr 5e-5
## train on accompaniment
nohup python -u run.py > train_accompaniment.log 2>&1 &
## test opencpop
(PyTorch)
label
'/data/huangrm/audioLM/musicgen_trainer/archivo/opencpop/train_cutlrc/2010000384.txt'
audio
'/data/huangrm/audioLM/musicgen_trainer/archivo/opencpop/train_cutwav/2010000384.wav'
