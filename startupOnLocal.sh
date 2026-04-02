#!/bin/bash
# startupOnLocal.sh - gets minikube and openfaas running from inside the repo

set -e

echo "cleaning up apt caches so the vm doesn't trip over itself..."
sudo rm -rf /var/lib/apt/lists/*
sudo rm -rf /var/lib/command-not-found/*

echo "grabbing docker and basic tools..."
sudo apt-get update
sudo apt-get install -y curl wget git docker.io apt-transport-https conntrack python3-venv python3-pip
sudo systemctl enable --now docker

# FIX: moved usermod before sg block and install faas-cli before sg too,
# so paths are consistent inside the subshell
sudo usermod -aG docker $USER

echo "installing minikube..."
curl -LO https://storage.googleapis.com/minikube/releases/latest/minikube-linux-amd64
sudo install minikube-linux-amd64 /usr/local/bin/minikube
rm minikube-linux-amd64

echo "installing kubectl..."
curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"
sudo install -o root -g root -m 0755 kubectl /usr/local/bin/kubectl
rm kubectl

echo "installing arkade and faas-cli..."
curl -sLS https://raw.githubusercontent.com/alexellis/arkade/master/get.sh | sudo sh
arkade get faas-cli
sudo mv ~/.arkade/bin/faas-cli /usr/local/bin/

echo "switching to docker group to spin up the cluster..."
sg docker -c '
set -e

echo "starting minikube..."
minikube start --driver=docker

echo "installing openfaas-ce..."
arkade install openfaas-ce

echo "waiting for gateway pods to boot (might take a sec)..."
kubectl wait --for=condition=ready pod -l app=gateway -n openfaas --timeout=300s

echo "port-forwarding gateway and prometheus to 0.0.0.0 in the background..."
kubectl port-forward -n openfaas svc/gateway 8080:8080 --address 0.0.0.0 > gateway.log 2>&1 &
kubectl port-forward -n openfaas svc/prometheus 9090:9090 --address 0.0.0.0 > prometheus.log 2>&1 &
sleep 5

echo "logging into faas-cli..."
PASSWORD=$(kubectl get secret -n openfaas basic-auth -o jsonpath="{.data.basic-auth-password}" | base64 --decode)
echo -n $PASSWORD | faas-cli login --username admin --password-stdin --gateway http://127.0.0.1:8080

# FIX: was "faas-cli store deploy figlet" which bypasses stack.yml entirely.
# Using stack.yml deploys your image with your scale labels and resource limits.
echo "deploying figlet-fn via stack.yml..."
OPENFAAS_URL=http://127.0.0.1:8080
faas-cli deploy -f stack.yml

# Verify it came up
echo "verifying deployment..."
sleep 3
faas-cli list --gateway http://127.0.0.1:8080

echo ""
echo "done. everything is running."
echo "gateway:    http://127.0.0.1:8080"
echo "prometheus: http://127.0.0.1:9090"
echo "password:   $PASSWORD"
'
