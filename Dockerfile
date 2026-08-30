FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglu1-mesa \
    libxrender1 \
    libxext6 \
    libsm6 \
    libice6 \
    libx11-6 \
    libxi6 \
    libxrandr2 \
    libxfixes3 \
    libxcursor1 \
    libxinerama1 \
    libfontconfig1 \
    libxft2 \
    libxt6 \
    libxkbcommon0 \
    libdbus-1-3 \
    libxcb1 \
    libgomp1 \
    build-essential \
    wget \
    bzip2 \
    && rm -rf /var/lib/apt/lists/*

RUN wget -q https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh \
    -O /tmp/miniforge.sh \
    && bash /tmp/miniforge.sh -b -p /opt/conda \
    && rm /tmp/miniforge.sh \
    && /opt/conda/bin/conda install -y -c conda-forge calculix \
    && /opt/conda/bin/conda clean -afy \
    && ln -s /opt/conda/bin/ccx /usr/local/bin/ccx
ENV LD_LIBRARY_PATH="/opt/conda/lib:${LD_LIBRARY_PATH}"

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY analysis_service.py .

CMD ["sh", "-c", "uvicorn analysis_service:app --host 0.0.0.0 --port ${PORT:-8000}"]
