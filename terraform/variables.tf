variable "subscription_id" {
  description = "Azure subscription ID"
  type        = string
}

variable "resource_group_name" {
  description = "Name of the resource group"
  type        = string
  default     = "rg-faas-rl"
}

variable "location" {
  description = "Azure region"
  type        = string
  default     = "eastus"
}

variable "acr_name" {
  description = "Azure Container Registry name (globally unique, alphanumeric)"
  type        = string
  default     = "faasrlacr"
}

variable "aks_cluster_name" {
  description = "AKS cluster name"
  type        = string
  default     = "aks-faas-rl"
}

variable "kubernetes_version" {
  description = "Kubernetes version for AKS"
  type        = string
  default     = "1.29"
}

variable "environment" {
  description = "Environment tag (dev / staging / prod)"
  type        = string
  default     = "dev"
}

variable "tf_state_storage_account" {
  description = "Storage account name for Terraform remote state"
  type        = string
}
