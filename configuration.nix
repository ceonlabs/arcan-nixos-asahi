{ config, lib, pkgs, ... }:
let
  srcPkgs = import /src { config.contentAddressedByDefault = true; };
in
{
  imports = [ ./hardware-configuration.nix ./apple-silicon-support ];

  systemd.services.durden = {  
    enable = true;                                                                                                                                                                                                                                     
    description = "Arcan Durden";
    after = [ "sysinit.target" ];
    wants = [ "basic.target" ];
    wantedBy = [ "sysinit.target" ];
    serviceConfig = {                                                                                                                                                                                                                             
      Type = "simple";                                                                                                                                                                                                                                         
      ExecStart = "${pkgs.arcan}/bin/arcan /root/durden/durden";                                                                                                                                                                                               
    };
    environment = {
      ARCAN_LOGPATH="/etc/arcan/logs";
      ARCAN_RENDER_NODE="/dev/dri/renderD128";
      ARCAN_RESOURCEPATH="${pkgs.durden}/share/arcan/data/resources";
      ARCAN_STATEPATH="/etc/arcan/state";
      ARCAN_APPLSTOREPATH="/etc/arcan/appls";
      XDG_RUNTIME_DIR="/etc/arcan";
      NIXOS_OZONE_WL="1";
      #DISPLAY=":0";
      #WAYLAND_DISPLAY="wayland-0";
    };                                                                                                                                                                                                                                                         
  };

  services.create_ap = {
   enable = true;
   settings = {
     INTERNET_IFACE = "end0";
     WIFI_IFACE = "wlan0";
     SSID = "arcan-net";
     PASSPHRASE = "12345678";
   };
  };

  nix.nixPath = [ "nixpkgs=/src" "nixos-config=/src/configuration.nix" ];
  nix.settings.experimental-features = [ "nix-command" "flakes" "ca-derivations" ];
  nix.settings.substituters = [];
  nix.settings.trusted-public-keys = [];
  boot.loader.systemd-boot.enable = true;
  boot.loader.efi.canTouchEfiVariables = false;
  boot.loader.timeout = 0;
  hardware.asahi.peripheralFirmwareDirectory = ./firmware;
  hardware.opengl.enable = true;
  hardware.opengl.package = srcPkgs.mesa.drivers;
  hardware.asahi.withRust = true;
  
  services.xserver.config = ''
   Section "OutputClass"
      Identifier "appledrm"
      MatchDriver "apple"
      Driver "modesetting"
      Option "PrimaryGPU" "true"
   EndSection
  '';
  

  security.pam.loginLimits = [
      {domain = "*";type = "-";item = "memlock";value = "infinity";}
      {domain = "*";type = "-";item = "nofile";value = "8192";}
  ];

  networking.wireless.iwd.enable = true;
  networking.wireless.iwd.settings.General.EnableNetworkConfiguration = true;
  networking.networkmanager.enable = false;
  services.getty.autologinUser = "root";
  users.defaultUserShell = srcPkgs.zsh;
  time.timeZone = "America/New_York";
  sound.enable = true;
  security.rtkit.enable = true;
  system.stateVersion = "24.05";
  
  environment.systemPackages = with srcPkgs; [
      git
      ripgrep
      helix
      tmux
      zenith
      zsh
      arcan
      firefox-devedition
      xarcan
      vkmark
      mangohud
      zed-editor
      godot_4
      blender
  ];
}

