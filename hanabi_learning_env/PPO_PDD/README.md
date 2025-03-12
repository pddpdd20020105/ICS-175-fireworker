## 1. Build

### 1.1 Build Hanabi Environment
First, compile the Hanabi game environment:

```bash
# Navigate to the Hanabi directory
cd envs/hanabi

# Create a build directory and enter it
mkdir build
cd build

# Run CMake to build the environment
cmake ..
make -j
```

###1.2 install python package like torch, wandb ....
```bash
pip install -r requirements.txt
```

### 1.3 Navigate to `/mappo/onpolicy/`

To evaluate the model, run:
```bash
python eval_hanabi.py
```
To train the model, run:
```bash
./train_hanabi_forward.sh
```
