cd ./Transfer_Learning
kaggle competitions download -c machine-learning-in-science-2022
mkdir data
unzip machine-learning-in-science-2022.zip -d ./data
rm machine-learning-in-science-2022.zip
mkdir models
cd ./models
mkdir speed
mkdir angle