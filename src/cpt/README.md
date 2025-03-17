# Train in Leonardo

## Prepare Environment

Load required modles
```
module load profile/deeplrn
module load python/3.10.8--gcc--11.3.0
module load cuda/11.8
module load openmpi/4.1.4--gcc--11.3.0-cuda-11.8
module load zlib/1.2.13--gcc--11.3.0
module load git-lfs

```

Prepare the virtual environment
```
python -m venv $SCRATCH/environments/axolotl-env
source $SCRATCH/environments/axolotl-env/bin/activate
```

Install Axolotl
```
# Update wheel and packaging 
pip install -U packaging wheel
# install prerequisites of axolotl: torch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
# install axolotl
pip install --no-build-isolation --no-cache-dir axolotl[flash-attn,deepspeed]
```

