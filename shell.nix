let
  pkgs = import <nixpkgs> {};
in pkgs.mkShell {
  packages = [
    (pkgs.python312.withPackages (python-pkgs: with python-pkgs; [
      pip
      opencv4
      scipy
      black
      flake8
      tqdm
      openpyxl
      notebook
    ]))
  ];
}