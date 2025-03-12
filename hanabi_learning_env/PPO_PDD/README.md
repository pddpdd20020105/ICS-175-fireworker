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

###1.3 goto the /mappo/onpolicy/
For evaluation
run eval_hanabi.py
For training
run train_hanabi_forward.sh

