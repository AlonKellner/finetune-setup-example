# Skypilot API server costs

When the skypilot API server is up it costs a minimum of 380$ per month (1360nis), mostly for pods compute & monitoring (360$, 1290nis) which can be slightly optimized with cheaper machines but not that much.  
The remaining cost is due to network forwarding rules (20$, 70nis).  
So, to avoid costs when idle the server must be teared down when not in use.

### Setup

1. `gcloud compute forwarding-rules import a7ffc75a4f58548fdb0633ef00aef1a8 --region=us-central1 --source skypilot-forward-rule.yaml`
2. Scale deployments to 1:  
   https://console.cloud.google.com/kubernetes/deployment/us-central1/main/skypilot/skypilot-api-service/overview?inv=1&invt=AbrC0w&project=triple-nectar-447116-d3  
   https://console.cloud.google.com/kubernetes/deployment/us-central1/main/skypilot/skypilot-platform-ingress-nginx-controller/overview?inv=1&invt=AbrC0w&project=triple-nectar-447116-d3

### Teardown

1. Scale deployments to 0:  
   https://console.cloud.google.com/kubernetes/deployment/us-central1/main/skypilot/skypilot-api-service/overview?inv=1&invt=AbrC0w&project=triple-nectar-447116-d3  
   https://console.cloud.google.com/kubernetes/deployment/us-central1/main/skypilot/skypilot-platform-ingress-nginx-controller/overview?inv=1&invt=AbrC0w&project=triple-nectar-447116-d3
2. ` gcloud compute forwarding-rules delete a7ffc75a4f58548fdb0633ef00aef1a8 --region=us-central1`
