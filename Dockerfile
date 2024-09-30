# Use an official base image
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .

RUN pip3 install  -r requirements.txt

COPY . .  

EXPOSE 5000


 
HEALTHCHECK --interval=30s --timeout=10s --retries=3 CMD curl --fail http://localhost:5000/health || exit 1

WORKDIR /app/
CMD ["fastapi","run", "--host","0.0.0.0", "--port", "5000","VAE_endpoint.py"]

