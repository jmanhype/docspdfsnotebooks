# install_packages.py

# Disabling echo to prevent command echoing
print("@echo off")

# Python packages installation
packages = [
    ("markdown2[all]", "For converting Markdown text to HTML."),
    ("numpy", "Essential for numerical computations."),
    ("requests", "Enables making HTTP requests."),
    ("tokenizers>=0.12.1", "For efficient text tokenization."),
    ("uvicorn", "ASGI server for web applications."),
    ("wandb", "For experiment tracking and visualization."),
    ("shortuuid", "For generating short, unique IDs."),
    ("httpx==0.24.0", "Asynchronous HTTP client."),
    ("peft==0.4.0", "Performance Estimation Framework Tool."),
    ("transformers==4.31.0", "State-of-the-art natural language processing."),
    ("accelerate", "Simplifies running PyTorch applications."),
    ("scikit-learn==1.2.2", "Machine learning library."),
    ("sentencepiece==0.1.99", "Tokenizer library for text processing."),
    ("einops==0.6.1", "For tensor operations."),
    ("einops-exts==0.0.4", "Extensions for Einops."),
    ("timm==0.6.13", "For deep learning image models."),
    ("gradio==3.35.2", "For building ML demo interfaces."),
    ("gradio_client==0.2.9", "Client for Gradio interfaces."),
    ("chardet", "Character encoding detection."),
    ("fastapi", "For building APIs with Python."),
    ("torch torchvision torchaudio torchdata", "For machine learning and tensor computations, with extra PyTorch components."),
    ("pydantic==1.10.9", "For data validation and settings management."),
    ("git+https://github.com/Keith-Hon/bitsandbytes-windows.git", "Custom PyTorch optimizer.")
]

# Constructing the installation commands
for package, description in packages:
    print(f"..\\python_embeded\\python.exe -s -m pip install {package} # {description}")

# Pause command at the end
print("pause")
