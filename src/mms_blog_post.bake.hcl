variable "datetime_tag" {
  default = formatdate("YYYYMMDD-hhmmss", timestamp())
}

group "default" {
  targets = ["cuda", "rocm"]
}

target "cuda" {
  context = "."
  dockerfile = "src/mms_blog_post.Dockerfile"
  target = "bare"
  tags = ["4alonkellner/finetune-setup-example-cuda:${datetime_tag}", "4alonkellner/finetune-setup-example-cuda:latest"]
  output = [{ type = "registry" }]
  platforms = ["linux/amd64"]
  args = {
    BASE_IMAGE = "nvcr.io/nvidia/cuda-dl-base:25.05-cuda12.9-runtime-ubuntu24.04"
    IS_ROCM = "FALSE"
    EXTRA_GROUP = "cuda"
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
    BASE_IMAGE = "rocm/pytorch:rocm6.4_ubuntu24.04_py3.12_pytorch_release_2.6.0"
    IS_ROCM = "TRUE"
    EXTRA_GROUP = "rocm"
  }
}
