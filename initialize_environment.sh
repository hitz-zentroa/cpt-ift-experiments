module load profile/deeplrn
module load python/3.10.8--gcc--11.3.0
module load cuda/11.8
module load openmpi/4.1.4--gcc--11.3.0-cuda-11.8
module load zlib/1.2.13--gcc--11.3.0
module load git-lfs


python -m venv $WORK/environments/axolot

source $WORK/environments/neox-env/bin/activate

# install torch
#pip install --upgrade "langdetect>1.0.9"
pip install --upgrade torch --index-url https://download.pytorch.org/whl/cu118
# install atxolot:
pip install --no-build-isolation --no-cache-dir axolotl[flash-attn,deepspeed]

