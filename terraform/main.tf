###############################################################
# Azure infrastructure for FaaS RL Pre-warmer
# Resources:
#   - Resource group
#   - Azure Container Registry (ACR)
#   - AKS cluster (system + user node pools)
#   - ACR-AKS pull role assignment
#   - Storage account + blob container (Terraform state backend)
###############################################################

terraform {
  required_version = ">= 1.6"

  required_providers {
    azurerm = {
      source  = "hashicorp/azurerm"
      version = "~> 3.100"
    }
    random = {
      source  = "hashicorp/random"
      version = "~> 3.6"
    }
  }

  # Remote state — bootstrap this storage account manually once:
  #   az group create -n rg-tfstate -l eastus
  #   az storage account create -n <name> -g rg-tfstate --sku Standard_LRS
  #   az storage container create -n tfstate --account-name <name>
  backend "azurerm" {
    resource_group_name  = "rg-tfstate"
    storage_account_name = var.tf_state_storage_account
    container_name       = "tfstate"
    key                  = "faas-rl.terraform.tfstate"
  }
}

provider "azurerm" {
  features {}
  subscription_id = var.subscription_id
}

# ── Resource group ────────────────────────────────────────────
resource "azurerm_resource_group" "main" {
  name     = var.resource_group_name
  location = var.location
  tags     = local.common_tags
}

# ── Azure Container Registry ─────────────────────────────────
resource "azurerm_container_registry" "acr" {
  name                = var.acr_name
  resource_group_name = azurerm_resource_group.main.name
  location            = azurerm_resource_group.main.location
  sku                 = "Basic"
  admin_enabled       = false
  tags                = local.common_tags
}

# ── AKS cluster ──────────────────────────────────────────────
resource "azurerm_kubernetes_cluster" "aks" {
  name                = var.aks_cluster_name
  location            = azurerm_resource_group.main.location
  resource_group_name = azurerm_resource_group.main.name
  dns_prefix          = var.aks_cluster_name
  kubernetes_version  = var.kubernetes_version
  tags                = local.common_tags

  # System node pool (for kube-system, openfaas-operator)
  default_node_pool {
    name                = "system"
    node_count          = 2
    vm_size             = "Standard_D2s_v3"
    os_disk_size_gb     = 30
    type                = "VirtualMachineScaleSets"
    enable_auto_scaling = true
    min_count           = 2
    max_count           = 4
    node_labels = {
      "pool" = "system"
    }
  }

  identity {
    type = "SystemAssigned"
  }

  network_profile {
    network_plugin    = "azure"
    load_balancer_sku = "standard"
  }

  oms_agent {
    log_analytics_workspace_id = azurerm_log_analytics_workspace.aks.id
  }

  lifecycle {
    ignore_changes = [default_node_pool[0].node_count]
  }
}

# User node pool for workloads (OpenFaaS functions, RL agent)
resource "azurerm_kubernetes_cluster_node_pool" "user" {
  name                  = "user"
  kubernetes_cluster_id = azurerm_kubernetes_cluster.aks.id
  vm_size               = "Standard_D4s_v3"
  node_count            = 2
  enable_auto_scaling   = true
  min_count             = 2
  max_count             = 10
  os_disk_size_gb       = 50
  node_labels = {
    "pool" = "user"
  }
  node_taints = []
  tags        = local.common_tags
}

# ── ACR pull permission for AKS ──────────────────────────────
resource "azurerm_role_assignment" "aks_acr_pull" {
  principal_id                     = azurerm_kubernetes_cluster.aks.kubelet_identity[0].object_id
  role_definition_name             = "AcrPull"
  scope                            = azurerm_container_registry.acr.id
  skip_service_principal_aad_check = true
}

# ── Log Analytics (AKS monitoring) ───────────────────────────
resource "azurerm_log_analytics_workspace" "aks" {
  name                = "${var.aks_cluster_name}-logs"
  location            = azurerm_resource_group.main.location
  resource_group_name = azurerm_resource_group.main.name
  sku                 = "PerGB2018"
  retention_in_days   = 30
  tags                = local.common_tags
}

# ── Locals ────────────────────────────────────────────────────
locals {
  common_tags = {
    project     = "faas-rl-prewarmer"
    environment = var.environment
    managed_by  = "terraform"
  }
}
