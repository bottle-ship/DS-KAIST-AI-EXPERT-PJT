export PJT_NAME=ds-kiast-team-pjt
export PJT_DIR=$HOME/$PJT_NAME
export PJT_ACT=$PJT_DIR/bin/activate

python3 -m venv $PJT_DIR
source $PJT_ACT

python3 -m pip install --upgrade pip
pip3 install --upgrade setuptools
pip3 install --upgrade numpy==1.16.4
pip3 install --upgrade scikit-learn
pip3 install --upgrade tensorflow-gpu==1.14.0
pip3 install --upgrade keras
pip3 install --upgrade pydot
pip3 install --upgrade graphviz
pip3 install https://download.pytorch.org/whl/cu100/torch-1.1.0-cp37-cp37m-linux_x86_64.whl
pip3 install https://download.pytorch.org/whl/cu100/torchvision-0.3.0-cp37-cp37m-linux_x86_64.whl
pip3 install --upgrade torchsummary
pip3 install --upgrade matplotlib
pip3 install --upgrade pandas
pip3 install --upgrade opencv-python
pip3 install --upgrade scikit-image
pip3 install --upgrade jupyter
pip3 install --upgrade ipykernel
pip3 install --upgrade wget

pip freeze

python3 -m ipykernel install --user --name $PJT_NAME --display-name "$PJT_NAME"

echo source $PJT_ACT > $HOME/team_pjt.venv
