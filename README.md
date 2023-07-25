# MPT-30B Model Inference Using CPU with Streamlit Interface

This project allows for inference with the MPT-30B model using a CPU and a simple Streamlit interface. The code runs the ggml quantized model.

## Requirements
A minimum of 32GB of RAM is required.

## Setup
Specify the number of threads you want to use and the directory where the model will be stored in the `config.json` file.

Set up the environment:
```bash
python -m venv env
source env/bin/activate
```

Install the dependencies:
```bash
pip install -r requirements.txt
```

Run the Streamlit app:
```bash
streamlit run app.py
```
