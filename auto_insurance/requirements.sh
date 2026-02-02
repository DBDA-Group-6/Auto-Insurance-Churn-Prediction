#!/bin/bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y python3 python3-pip python3-dev gcc g++ make
pip3 install --user --break-system-packages flask pandas pymysql sqlalchemy
sudo apt install -y mysql-client
pip3 install --user --break-system-packages joblib scikit-learn xgboost imbalanced-learn
source ~/.bashrc
python3 -c "import flask, pandas, sklearn, xgboost, imblearn; print('ML + Web ready!')"
mysql --version
