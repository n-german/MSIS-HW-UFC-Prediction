.PHONY: dataset eda train shap all app

dataset:
	python -m src.make_dataset

eda:
	python -m src.eda

train:
	python -m src.train_models

shap:
	python -m src.explain_shap

all: dataset eda train shap

app:
	streamlit run app.py