variable "datetime_tag" {
  default = formatdate("YYYYMMDD-hhmmss", timestamp())
}

group "default" {
  targets = ["image"]
}

target "image" {
  context = "."
  dockerfile = "src/mms_blog_post.Dockerfile"
  target = "bare"
  tags = ["4alonkellner/finetune-setup-example:${datetime_tag}", "4alonkellner/finetune-setup-example:latest"]
  output = [{ type = "registry" }]
}
