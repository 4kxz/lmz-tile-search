{ pkgs ? import <nixpkgs> {} }:

pkgs.mkShell {
  buildInputs = [
    pkgs.uv
  ];

  shellHook = ''
    echo "UV-based development environment"
  '';
}
