# ====== Base Image ======
FROM python:3.11-slim

# ====== Variables ======
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive

# ====== Mise à jour et dépendances système ======
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    wget \
    unzip \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# ====== Créer un dossier de travail ======
WORKDIR /app

# ====== Copier requirements.txt ======
COPY requirements.txt .

# ====== Installer packages Python ======
RUN pip install --no-cache-dir -r requirements.txt

# ====== Installer PyTorch (GPU ou CPU selon besoin) ======
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# ====== Copier le code ======
COPY . .

# ====== Exposer ports ======
EXPOSE 8501 8888 8080

# ====== Commande par défaut ======
CMD ["streamlit", "run", "app/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
