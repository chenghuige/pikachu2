
# #################### get env directories
# CONDA_ROOT
CONDA_CONFIG_ROOT_PREFIX=$(conda config --show root_prefix)
echo "CONDA_CONFIG_ROOT_PREFIX= ${CONDA_CONFIG_ROOT_PREFIX}"
get_conda_root_prefix() {
    TMP_POS=$(awk -v a="${CONDA_CONFIG_ROOT_PREFIX}" -v b="/" 'BEGIN{print index(a, b)}')
    TMP_POS=$((TMP_POS-1))
    if [ $TMP_POS -ge 0 ]; then
        echo "${CONDA_CONFIG_ROOT_PREFIX:${TMP_POS}}"
    else
        echo ""
    fi
}
CONDA_ROOT=$(get_conda_root_prefix)
if [ ! -d "${CONDA_ROOT}" ]; then
    echo "CONDA_ROOT= ${CONDA_ROOT}, not exists, exit"
    exit 1
fi

# CONDA ENV
CONDA_NEW_ENV=pikachu

# JUPYTER_ROOT
JUPYTER_ROOT=/home/tione/notebook
if [ ! -d "${JUPYTER_ROOT}" ]; then
    echo "JUPYTER_ROOT= ${JUPYTER_ROOT}, not exists, exit"
    exit 1
fi

# OS RELEASE
OS_ID=$(awk -F= '$1=="ID" { print $2 ;}' /etc/os-release)
OS_ID=${OS_ID//"\""/""}

echo "CONDA_ROOT= ${CONDA_ROOT}"
echo "CONDA_NEW_ENV= ${CONDA_NEW_ENV}"
echo "JUPYTER_ROOT= ${JUPYTER_ROOT}"
echo "OS_ID= ${OS_ID}"

# #################### install or fix environments
ACTION=$(echo "$1" | tr '[:upper:]' '[:lower:]')
if [ "${ACTION}" == "install" ]; then
    echo "[Info] Install environments"

    # #################### create conda env and activate
    source "${CONDA_ROOT}/etc/profile.d/conda.sh"
    # create env by prefix
    conda env list
    conda create --prefix ${JUPYTER_ROOT}/envs/${CONDA_NEW_ENV} -y python=3.6 ipykernel
#     conda create --prefix ${JUPYTER_ROOT}/envs/${CONDA_NEW_ENV} -y --clone tensorflow_py3
    conda env list
    source activate ${JUPYTER_ROOT}/envs/${CONDA_NEW_ENV}
    which pip
    conda install pip 
    which pip
    conda install pip
    conda install -y cudatoolkit=10.1 cudnn
    
    # #################### install requirements
    pip install torch==1.7.1+cu101 -f https://download.pytorch.org/whl/torch_stable.html
    pip config set global.index-url https://mirrors.tencent.com/pypi/simple/
    pip install -r requirements.txt
    jupyter nbextension enable --py widgetsnbextension
  
    # #################### check install
    # check tensorflow GPU
    python -c "import tensorflow as tf; tf.test.gpu_device_name()"
    python -c "from tensorflow.python.client import device_lib; print(device_lib.list_local_devices())"
    # check library versions
    echo "[TensorFlow]"
    python -c "import tensorflow as tf; print(tf.__version__)"
    echo "[Torch]"
    python -c "import torch; print(torch.__version__)"
    python -c "import torch; print(torch.cuda.is_available())"
    #     python -c "import torchvision; print(torchvision.__version__)"
    
    exit 0
    
elif [ "${ACTION}" == "fix" ]; then
    echo "[Info] Fix environments"
    
    # #################### add conda env
    source "${CONDA_ROOT}/etc/profile.d/conda.sh"
    conda env list
    # add envs_dirs
    conda config --add envs_dirs ${JUPYTER_ROOT}/envs
    conda config --show | grep env
    conda env list
    
    exit 0
else
    echo "[Error] Please set the args as 'install' or 'fix'"
    exit 1
fi




