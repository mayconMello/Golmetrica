FROM python:3.12-slim

# Instalar dependências do sistema (inclui tzdata para timezone)
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    tzdata \
    && rm -rf /var/lib/apt/lists/*

# Definir diretório de trabalho
WORKDIR /app

# Copiar requirements e instalar dependências Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar código da aplicação
COPY . .

# Criar usuário não-root para segurança
RUN groupadd -r appuser && useradd -r -g appuser appuser
RUN mkdir -p /app/logs && chown -R appuser:appuser /app

USER appuser

# Comando padrão
CMD ["python", "/app/scheduler.py"]
