# MPT-30B model using CPU with streamlit interface

Interference with MPT-30B model using CPU with simple streamlit interface. The code runs ggml quantized model.

## Requirements
Min 32GB of ram.

## Setup
Put number of threads you want to use in config.json and directory the model will be stored. 

Environment:
```bash
python -m venv env && source env
```

Dependencies: 
```bash
pip install -r requirements.txt
```

Run streamlit app:
```bash
streamlit run app.py
```
