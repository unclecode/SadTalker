update-alternatives --install /usr/local/bin/python3 python3 /usr/bin/python3.8 2
update-alternatives --install /usr/local/bin/python3 python3 /usr/bin/python3.9 1
apt install python3.8 -y
apt-get install python3.8-distutils -y
apt-get update
apt install software-properties-common -y
dpkg --remove --force-remove-reinstreq python3-pip python3-setuptools python3-wheel
apt-get install python3-pip
apt install python3.8-venv -y