
module purge
module load python/3.7 
module load cuda/10.1/cudnn/7.6 
module load python/3.7/cuda/10.1/cudnn/7.6/pytorch/1.5.0

module list


VENV_NAME='danns-v2'
VENV_DIR=$HOME'/venvs/'$VENV_NAME

echo 'Building virtual env: '$VENV_NAME' in '$VENV_DIR

mkdir $VENV_DIR
# Add .gitignore to folder just in case it is in a repo
# Ignore everything in the directory apart from the gitignore file
echo "*" > $VENV_DIR/.gitignore
echo "!.gitignore" >> $VENV_DIR/.gitignore

virtualenv $VENV_DIR

source $VENV_DIR'/bin/activate'

# install python packages not provided by modules

pip install torchvision==0.6.0 --no-deps
pip install pytorch-nlp
pip install ipython --ignore-installed
pip install ipykernel

# grab the allen sdk!
#pip install allensdk
pip install matplotlib pandas scipy #wandb
pip install pillow
pip install wandb
pip install requests
pip install PyYAML
pip install six
pip install -U scikit-learn
pip install git+https://github.com/linclab/linclab_utils.git
pip install audiofile
# Orion installations
pip install orion 

# # install bleeding edge orion - bug fix now ignore code changes works
# #pip install git+https://github.com/epistimio/orion.git@develop
# pip install git+https://github.com/epistimio/orion.git@9f3894f3f95c71530249f8149b11beb0f31699bc

# # install grid search plugin
# pip install git+https://github.com/bouthilx/orion.algo.grid_search.git

# set up MILA jupyterlab
echo which ipython
ipython kernel install --user --name=danns-v2_kernel_1021

