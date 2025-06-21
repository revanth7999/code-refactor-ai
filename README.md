# Code Refactor AI

A lightweight AI model that automatically refactors code snippets based on predefined coding standards — built from scratch using PyTorch and exposed via a FastAPI REST API.

## Features

- Custom-trained Transformer model for code formatting
- Simple tokenizer and vocabulary builder
- Trains on user-defined code standards (e.g., spacing, naming)
- API endpoint to refactor any input code
- Fast inference via FastAPI (`/refactor`)

## Project Structure

```
code-refactor-ai/
├── api/ # FastAPI server
│ └── app.py
├── data/
│ └── code_pairs.csv # Training data: input_code → output_code
├── dataset/
│ └── code_dataset.py # PyTorch Dataset class
├── model/
│ └── transformer_model.py # Transformer-based model definition
├── tokenizer/
│ └── tokenizer.py # Tokenization & vocabulary utils
├── train.py # Training script
├── infer.py # Inference script (CLI mode)
├── refactor_model.pth # Saved trained model
└── requirements.txt # Python dependencies
```

## Installation

1. Clone the repo:
   ```
   git clone https://github.com/yourname/code-refactor-ai.git
   cd code-refactor-ai
   ```

2. Create and activate a virtual environment:

    ```
    python3 -m venv .venv
    source .venv/bin/activate
    ```

3. Install dependencies:
    ```
    pip install --break-system-packages -r requirements.txt
    ```

## Training

Edit `data/code_pairs.csv` to define your coding rules:

```
input_code,output_code
"int x=10;","int x = 10 ;"
"int y=true;","int y = true ;"
```

Train the model:

```
python train.py
```

Inference (CLI)

Refactor a code snippet using:

```
python infer.py
```

Output:

```
Refactored:
int x = 10 ;
```

Run the API

Start the FastAPI server:

```
uvicorn api.app:app --reload
```

Access via:
```
http://127.0.0.1:8000/docs
```

Example API call:

```
POST /refactor

{
  "code": "int x=10;"
}

Response:

{
  "refactored": "int x = 10 ;"
}
```
Tech Stack

1. Python 3.12
2. PyTorch (Transformer)
3. FastAPI
4. Uvicorn
5. Pandas

Future Ideas

1. VS Code plugin for live refactor-on-save
2. Train on multiple languages (Java, JS, Python)
3. Integrate org-specific rules from linters
4. Add web-based UI for interactive refactoring

Author

Revanth N
