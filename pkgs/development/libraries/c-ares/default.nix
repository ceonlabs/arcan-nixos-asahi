{ lib, stdenv, fetchurl, writeTextDir
, withCMake ? true, cmake

# sensitive downstream packages
, curl
, grpc # consumes cmake config
}:

# Note: this package is used for bootstrapping fetchurl, and thus
# cannot use fetchpatch! All mutable patches (generated by GitHub or
# cgit) that are needed here should be included directly in Nixpkgs as
# files.

stdenv.mkDerivation rec {
  pname = "c-ares";
  version = "1.27.0";

  src = fetchurl {
    url = "https://github.com/c-ares/c-ares/releases/download/cares-1_27_0/c-ares-1.27.0.tar.gz";
    #url = "https://c-ares.org/download/${pname}-${version}.tar.gz";
    hash = "sha256-CnK+ZpWZVcQ+KvL70DQY6Cor1UZGBOyaYhR+N6zrQgs=";
  };

  outputs = [ "out" "dev" "man" ];

  nativeBuildInputs = lib.optionals withCMake [ cmake ];

  cmakeFlags = [] ++ lib.optionals stdenv.hostPlatform.isStatic [
    "-DCARES_SHARED=OFF"
    "-DCARES_STATIC=ON"
  ];

  enableParallelBuilding = true;

  passthru.tests = {
    inherit grpc;
    curl = (curl.override { c-aresSupport = true; }).tests.withCheck;
  };

  meta = with lib; {
    description = "A C library for asynchronous DNS requests";
    homepage = "https://c-ares.haxx.se";
    changelog = "https://c-ares.org/changelog.html#${lib.replaceStrings [ "." ] [ "_" ] version}";
    license = licenses.mit;
    platforms = platforms.all;
  };
}
