variable "datetime_tag" {
  default = formatdate("YYYYMMDD-hhmmss", timestamp())
}

group "default" {
  targets = ["cuda", "rocm", "cpu"]
}

target "cpu" {
  context = "."
  dockerfile = "src/mms_blog_post.Dockerfile"
  target = "bare"
  tags = ["4alonkellner/finetune-setup-example-cpu:${datetime_tag}", "4alonkellner/finetune-setup-example-cpu:latest"]
  output = [{ type = "registry" }]
  platforms = ["linux/amd64"]
  args = {
    BASE_IMAGE = "mcr.microsoft.com/devcontainers/base:debian"
    IS_ROCM = "FALSE"
    EXTRA_GROUP = "cpu"
    EXTRA_AFTER_GROUP = "cpu"
  }
}

target "cuda" {
  context = "."
  dockerfile = "src/mms_blog_post.Dockerfile"
  target = "bare"
  tags = ["4alonkellner/finetune-setup-example-cuda:${datetime_tag}", "4alonkellner/finetune-setup-example-cuda:latest"]
  output = [{ type = "registry" }]
  platforms = ["linux/amd64"]
  args = {
    BASE_IMAGE = "nvcr.io/nvidia/cuda-dl-base:25.06-cuda12.9-runtime-ubuntu24.04"
    IS_ROCM = "FALSE"
    EXTRA_GROUP = "cuda"
    EXTRA_AFTER_GROUP = "flash-attn"
  }
}

target "rocm" {
  context = "."
  dockerfile = "src/mms_blog_post.Dockerfile"
  target = "bare"
  tags = ["4alonkellner/finetune-setup-example-rocm:${datetime_tag}", "4alonkellner/finetune-setup-example-rocm:latest"]
  output = [{ type = "registry" }]
  platforms = ["linux/amd64"]
  args = {
    BASE_IMAGE = "rocm/miopen:ci_faa726"
    IS_ROCM = "TRUE"
    EXTRA_GROUP = "rocm"
    EXTRA_AFTER_GROUP = "flash-attn"
  }
}
