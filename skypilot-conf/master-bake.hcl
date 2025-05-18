
variable "datetime_tag" {
  default = formatdate("YYYYMMDD-hhmmss", timestamp())
}

group "default" {
  targets = ["update", "image"]
}

target "update" {
  context = "/home/vscode/.ssh"
  dockerfile = "/workspaces/finetune-setup-example/skypilot-conf/master.Dockerfile"
  target = "update"
  no-cache = true
}

target "image" {
  context = "."
  dockerfile = "skypilot-conf/master.Dockerfile"
  target = "image"
  tags = ["4alonkellner/skypilot-api-test:${datetime_tag}-master", "4alonkellner/skypilot-api-test:latest-master"]
  output = [{ type = "registry" }]
  no-cache = true
}
