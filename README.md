# Scalable-ML-Workflow-with-Kubeflow

## Overview
This project demonstrates serving a machine learning model using:
- Flask
- TensorFlow
- Docker
- Kubernetes (Minikube)

## Steps Performed
1. Built a Flask-based ML inference API
2. Containerized the application using Docker
3. Created Kubernetes Deployment YAML
4. Deployed using Minikube
5. Exposed service using NodePort

## Note
Due to TensorFlow image size and limited local system memory,
Docker image build and Kubernetes pod startup may take significant time.
The configuration and deployment steps are correct and verified.

## Commands Used
```bash
docker build -t serving:latest serving/
minikube start --memory=3072
kubectl apply -f deployment.yaml
kubectl expose deployment serving-deployment --type=NodePort --port=5000
```
