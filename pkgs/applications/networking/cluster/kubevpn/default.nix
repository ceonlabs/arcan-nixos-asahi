{ lib, buildGoModule, fetchFromGitHub }:

buildGoModule rec {
  pname = "kubevpn";
  version = "2.2.9";

  src = fetchFromGitHub {
    owner  = "KubeNetworks";
    repo   = "kubevpn";
    rev    = "v${version}";
    hash = "sha256-C0rNTlcreBVxmLb2ChU5nYdafcnynd+GGuGFefFJXl8=";
  };

  vendorHash = null;

  # TODO investigate why some config tests are failing
  doCheck = false;

  meta = with lib; {
    changelog = "https://github.com/KubeNetworks/kubevpn/releases/tag/${src.rev}";
    description = "Create a VPN and connect to Kubernetes cluster network, access resources, and more";
    homepage = "https://github.com/KubeNetworks/kubevpn";
    license = licenses.mit;
    maintainers = with maintainers; [ mig4ng ];
  };
}