{ fetchurl, lib, stdenv, libiconv, libunistring, help2man, texinfo, buildPackages }:

# Note: this package is used for bootstrapping fetchurl, and thus
# cannot use fetchpatch! All mutable patches (generated by GitHub or
# cgit) that are needed here should be included directly in Nixpkgs as
# files.

stdenv.mkDerivation rec {
  pname = "libidn2";
  version = "2.3.7";

  src = fetchurl {
    url = "https://ftp.gnu.org/gnu/libidn/${pname}-${version}.tar.gz";
    hash = "sha256-TCGnkbYQuVGbnQ4SuAl78vNZsS+N2SZHYRqSnmv9fWQ=";
  };

  strictDeps = true;
  # Beware: non-bootstrap libidn2 is overridden by ./hack.nix
  outputs = [ "bin" "dev" "out" "info" "devdoc" ];

  enableParallelBuilding = true;

  # The above patch causes the documentation to be regenerated, so the
  # documentation tools are required.
  nativeBuildInputs = lib.optionals stdenv.isDarwin [ help2man texinfo ];
  buildInputs = [ libunistring ] ++ lib.optional stdenv.isDarwin libiconv;
  depsBuildBuild = [ buildPackages.stdenv.cc ];

  meta = {
    homepage = "https://www.gnu.org/software/libidn/#libidn2";
    description = "Free software implementation of IDNA2008 and TR46";

    longDescription = ''
      Libidn2 is believed to be a complete IDNA2008 and TR46 implementation,
      but has yet to be as extensively used as the IDNA2003 Libidn library.

      The installed C library libidn2 is dual-licensed under LGPLv3+|GPLv2+,
      while the rest of the package is GPLv3+.  See the file COPYING for
      detailed information.
    '';

    mainProgram = "idn2";
    license = with lib.licenses; [ lgpl3Plus gpl2Plus gpl3Plus ];
    platforms = lib.platforms.all;
    maintainers = with lib.maintainers; [ fpletz ];
  };
}
