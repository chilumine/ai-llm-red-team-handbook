#!/bin/bash
# AI LLM Red Team - 4. Infrastructure Dependencies
# Source: Chapter_13_Data_Provenance_and_Supply_Chain_Security
# Category: supply_chain

# Review cloud resource usage
aws resourcegroupstaggingapi get-resources
gcloud asset search-all-resources

# Check container base images
docker history your-ml-image:latest

# Review kubernetes manifests
kubectl get pods,services,deployments -o yaml
