#!/bin/bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y python3 python3-pip python3-dev gcc g++ make openjdk-11-jdk
pip3 install --user --break-system-packages flask pandas pymysql sqlalchemy
sudo apt install -y mysql-client
pip3 install --user --break-system-packages joblib scikit-learn xgboost imbalanced-learn pyspark
source ~/.bashrc
python3 -c "import flask, pandas, sklearn, xgboost, imblearn, pyspark; print('ML + Web + PySpark ready!')"
mysql --version
