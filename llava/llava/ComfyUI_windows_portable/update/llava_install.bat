@echo off

REM Batch script for installing Python packages

REM Markdown2 for converting Markdown text to HTML
py -m pip install markdown2[all] --user

REM NumPy for numerical computations
py -m pip install numpy --user

REM Requests for HTTP requests
py -m pip install requests --user

REM Tokenizers for text tokenization
py -m pip install tokenizers>=0.12.1 --user

REM Uvicorn for ASGI server
py -m pip install uvicorn --user

REM WandB for experiment tracking
py -m pip install wandb --user

REM ShortUUID for unique IDs
py -m pip install shortuuid --user

REM HTTPX for asynchronous HTTP client
py -m pip install httpx==0.24.0 --user

REM PEFT for performance estimation
py -m pip install peft==0.4.0 --user

REM Transformers for NLP
py -m pip install transformers==4.31.0 --user

REM Accelerate for PyTorch applications
py -m pip install accelerate --user

REM Scikit-learn for machine learning
py -m pip install scikit-learn==1.2.2 --user

REM SentencePiece for text processing
py -m pip install sentencepiece==0.1.99 --user

REM Einops for tensor operations
py -m pip install einops==0.6.1 --user

REM Einops-exts for Einops extensions
py -m pip install einops-exts==0.0.4 --user

REM Timm for deep learning image models
py -m pip install timm==0.6.13 --user

REM Gradio for ML demo interfaces
py -m pip install gradio==3.35.2 --user

REM Gradio Client for interfaces
py -m pip install gradio_client==0.2.9 --user

REM Chardet for character encoding detection
py -m pip install chardet --user

REM FastAPI for building APIs
py -m pip install fastapi --user

REM PyTorch for machine learning and tensor computations
py -m pip install --upgrade torch torchvision torchaudio torchdata --extra-index-url https://download.pytorch.org/whl/cu118 --user

REM Pydantic for data validation
py -m pip install pydantic==1.10.9 --user

REM Bitsandbytes for custom optimizer
py -m pip install git+https://github.com/Keith-Hon/bitsandbytes-windows.git --user

pause
