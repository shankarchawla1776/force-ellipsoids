{ pkgs ? import <nixpkgs> { config = { allowBroken = true; }; } }:


pkgs.mkShell {
  buildInputs = [
    # pkgs.mujoco
    pkgs.julia-mono

  ];

  shellHook = ''
    echo "julia env activated"
  '';
}
