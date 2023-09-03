conda create -n difftalk python=3.8
conda activate difftalk
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch
conda install ffmpeg
pip install -r requirements.txt
wget https://cloud.tsinghua.edu.cn/seafhttp/files/c4488c50-f55e-4af0-9c7e-df59c013edfc/model.ckpt -P models