# GitHub Secrets Setup for ACR Docker Build Workflow

## Required GitHub Repository Secrets

Add these secrets in: **Repository Settings → Secrets and variables → Actions → New repository secret**

### Azure Authentication (Federated Identity - Recommended)
- `AZURE_CLIENT_ID` - Service principal application (client) ID
- `AZURE_TENANT_ID` - Azure Active Directory tenant ID
- `AZURE_SUBSCRIPTION_ID` - Azure subscription ID

### Container Registry
- `AZURE_CONTAINER_REGISTRY_NAME` - Registry name

## Setup Instructions

### 1. Create Service Principal with Federated Credentials

```bash
# Set variables
SUBSCRIPTION_ID="your-subscription-id"
RESOURCE_GROUP="your-resource-group"
REGISTRY_NAME="your-registry-name"
GITHUB_ORG="Azure"
GITHUB_REPO="PyRIT"

# Create service principal
az ad sp create-for-rbac \
  --name "github-pyrit-acr-federated" \
  --role "AcrPush" \
  --scopes "/subscriptions/$SUBSCRIPTION_ID/resourceGroups/$RESOURCE_GROUP/providers/Microsoft.ContainerRegistry/registries/$REGISTRY_NAME"

# Note the appId from output - this is your AZURE_CLIENT_ID
APP_ID="<appId-from-output>"

# Create federated credential for main branch
az ad app federated-credential create \
  --id $APP_ID \
  --parameters '{
    "name": "github-pyrit-main",
    "issuer": "https://token.actions.githubusercontent.com",
    "subject": "repo:'"$GITHUB_ORG"'/'"$GITHUB_REPO"':ref:refs/heads/main",
    "audiences": ["api://AzureADTokenExchange"]
  }'

# Create federated credential for develop branch
az ad app federated-credential create \
  --id $APP_ID \
  --parameters '{
    "name": "github-pyrit-develop",
    "issuer": "https://token.actions.githubusercontent.com",
    "subject": "repo:'"$GITHUB_ORG"'/'"$GITHUB_REPO"':ref:refs/heads/develop",
    "audiences": ["api://AzureADTokenExchange"]
  }'

# Create federated credential for pull requests
az ad app federated-credential create \
  --id $APP_ID \
  --parameters '{
    "name": "github-pyrit-pr",
    "issuer": "https://token.actions.githubusercontent.com",
    "subject": "repo:'"$GITHUB_ORG"'/'"$GITHUB_REPO"':pull_request",
    "audiences": ["api://AzureADTokenExchange"]
  }'
```

### 2. Get Required Values

```bash
# Get subscription ID
az account show --query id --output tsv

# Get tenant ID
az account show --query tenantId --output tsv

# Verify registry exists
az acr show --name $REGISTRY_NAME --query loginServer
```

### 3. Add to GitHub Secrets

| Secret Name | Value | How to Get |
|-------------|-------|------------|
| `AZURE_CLIENT_ID` | Application (client) ID | From service principal creation output (`appId`) |
| `AZURE_TENANT_ID` | Directory (tenant) ID | `az account show --query tenantId -o tsv` |
| `AZURE_SUBSCRIPTION_ID` | Subscription ID | `az account show --query id -o tsv` |
| `AZURE_CONTAINER_REGISTRY_NAME` | `romanlutzregistry` | Your ACR name (without `.azurecr.io`) |

## Testing

After setup, the workflow will automatically run on:
- ✅ Push to `main` branche
- ✅ Manual trigger via GitHub Actions UI

## Troubleshooting

If authentication fails:
1. Verify service principal has `AcrPush` role on the registry
2. Check federated credential subjects match your repo (`Azure/PyRIT`)
3. Ensure `id-token: write` permission is set in workflow
4. Verify all secrets are correctly entered (no extra spaces/quotes)

## Security Notes

- ✅ Uses **OpenID Connect (OIDC)** - no secrets stored, tokens are short-lived
- ✅ No passwords or keys in GitHub secrets
- ✅ Federated credentials scope limited to specific branches/PRs
- ✅ `.env` files excluded via `.dockerignore` - never uploaded to ACR
