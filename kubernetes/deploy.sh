#!/bin/bash
set -e
echo "Setting k8s context..."
kubectl config set-context --current --namespace=ucr-rai-vedant
echo "Checking pods..."
kubectl get pods

echo "Applying storage claim..."
kubectl apply -f k8s-dorado-storage.yaml

echo "Committing local bug fixes so the pod can clone them..."
git add dorado/ main.py setup.sh || true
git commit -m "Pipeline and environment hardening for k8s deployment" || true
git push || echo "Please run 'git push' manually if you haven't set up credentials!"

echo "Deploying the Fast profile job to the cluster..."
# Wait for the user to push before applying the job! We will only apply the PVC for now.
# kubectl apply -f k8s-dorado-job.yaml
