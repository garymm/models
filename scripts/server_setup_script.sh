sudo apt update
sudo apt -y upgrade
sudo apt -y install libx11-dev libxcursor-dev libxrandr-dev libxinerama-dev libxi-dev libgl1-mesa-dev libgl1-mesa-dev xorg-dev
sudo apt -y install nano

git config --global user.email "andrewr@astera.org"
git config --global user.name "Andrew Keenan Richardson"

sudo snap install --clasic go
sudo chmod +777 ~/go/pkg

sudo apt -y install python3-pip
sudo pip3 install pandas
sudo pip3 install optuna
sudo pip3 install wandb

cd ~
git clone https://github.com/Astera-org/models.git
cd models
sudo go test mechs/ra25/ra25_test.go

# ssh-keygen -t rsa -b 2048 -C "<comment>"
# Copy .ssh/id_rsa.pub over to gitlab

cd ~
git clone git@gitlab.com:generally-intelligent/bones.git
sudo pip3 install -e bones/

