{ pkgs ? import <nixpkgs> {} }:

pkgs.mkShell {
  # Tools required for the build process (run on your machine)
  nativeBuildInputs = with pkgs; [
    uv
    pkg-config
  ];

  # Libraries required for Pygame to compile and link against
  buildInputs = with pkgs; [
    sdl2-compat
    SDL2_image
    SDL2_ttf
    SDL2_mixer
  ];

  # Explicitly tell the compiler and linker where the Nix store paths are
  shellHook = ''
    export CPATH="${pkgs.sdl2-compat.dev}/include/SDL2:$CPATH"
    export LDFLAGS="-L${pkgs.sdl2-compat}/lib -L${pkgs.SDL2_image}/lib -L${pkgs.SDL2_ttf}/lib -L${pkgs.SDL2_mixer}/lib $LDFLAGS"
  '';
}
