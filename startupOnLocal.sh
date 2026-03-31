#!/bin/bash
# setup-faas.sh - Automated deployment for FaaS RL Pre-warmer environment

set -e

echo "==> Starting fresh Ubuntu setup for OpenFaaS & Minikube"

# PREEMPTIVE AZURE BUG FIX
echo "==> Cleaning apt caches to prevent cloud-init conflicts"
sudo rm -rf /var/lib/apt/lists/*
sudo rm -rf /var/lib/command-not-found/*

# 1. Update system and install base dependencies
echo "==> Installing Docker and base utilities"
sudo apt-get update
sudo apt-get install -y curl wget git docker.io apt-transport-https conntrack
sudo systemctl enable --now docker
sudo usermod -aG docker $USER

# 2. Install Minikube
echo "==> Installing Minikube"
curl -LO https://storage.googleapis.com/minikube/releases/latest/minikube-linux-amd64
sudo install minikube-linux-amd64 /usr/local/bin/minikube
rm minikube-linux-amd64

# 3. Install kubectl
echo "==> Installing kubectl"
curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"
sudo install -o root -g root -m 0755 kubectl /usr/local/bin/kubectl
rm kubectl

# 4. Install Arkade and faas-cli
echo "==> Installing Arkade and faas-cli"
curl -sLS https://raw.githubusercontent.com/alexellis/arkade/master/get.sh | sudo sh
arkade get faas-cli
sudo mv ~/.arkade/bin/faas-cli /usr/local/bin/

# ==============================================================================
# Switch to the new docker group temporarily to initialize Minikube
# ==============================================================================
echo "==> Switching to docker group to initialize cluster"
sg docker -c '
set -e

echo "==> Starting Minikube (docker driver)"
minikube start --driver=docker

echo "==> Installing OpenFaaS Community Edition"
arkade install openfaas-ce

echo "==> Waiting for OpenFaaS Gateway to become ready..."
kubectl wait --for=condition=ready pod -l app=gateway -n openfaas --timeout=300s

echo "==> Configuring external port-forwards (0.0.0.0)"
kubectl port-forward -n openfaas svc/gateway 8080:8080 --address 0.0.0.0 > ~/gateway.log 2>&1 &
kubectl port-forward -n openfaas svc/prometheus 9090:9090 --address 0.0.0.0 > ~/prometheus.log 2>&1 &

sleep 5

echo "==> Retrieving credentials and logging into faas-cli"
PASSWORD=$(kubectl get secret -n openfaas basic-auth -o jsonpath="{.data.basic-auth-password}" | base64 --decode)
echo -n $PASSWORD | faas-cli login --username admin --password-stdin --gateway http://127.0.0.1:8080

echo "==> Cloning RL Agent repository"
cd ~
if [ ! -d "faas-rl-prewarm-agent" ]; then
    git clone https://github.com/Ragingsnake/faas-rl-prewarm-agent.git
else
    echo "    Repository exists, skipping clone."
fi

echo "==> Deploying public test function (figlet)"
faas-cli store deploy figlet --gateway http://127.0.0.1:8080

echo ""
echo "======================================================"
echo "Setup Complete."
echo "OpenFaaS Password: $PASSWORD"
echo "Gateway logs:      ~/gateway.log"
echo "Prometheus logs:   ~/prometheus.log"
echo "======================================================"
'