#sudo dpkg-reconfigure dash #then select no
./conda_init.sh fix
source activate /home/tione/notebook/envs/pikachu
alias ..='cd ..'
alias ls='ls --color=tty'
alias ll='ls -l'
export PATH=~/notebook/pikachu/tools:~/notebook/pikachu/tools/bin:$PATH
export PYTHONPATH=~/notebook/pikachu/utils:~/notebook/pikachu/third:$PYTHONPATH
wandb disabled
