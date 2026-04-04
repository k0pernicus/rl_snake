{ pkgs ? import <nixpkgs> {} }:

pkgs.mkShell {
  nativeBuildInputs = with pkgs; [
    uv
    pkg-config
  ];

  buildInputs = with pkgs; [
    sdl2-compat
    SDL2_image
    SDL2_ttf
    SDL2_mixer
    freetype      # Added the missing dependency for pygame.font
  ];

  # Safely extract headers and libraries without risking missing attributes
  shellHook = ''
    export CPATH="${pkgs.lib.getDev pkgs.sdl2-compat}/include/SDL2:${pkgs.lib.getDev pkgs.SDL2_ttf}/include/SDL2:${pkgs.lib.getDev pkgs.SDL2_image}/include/SDL2:${pkgs.lib.getDev pkgs.SDL2_mixer}/include/SDL2:${pkgs.lib.getDev pkgs.freetype}/include/freetype2:$CPATH"
    export LDFLAGS="-L${pkgs.lib.getLib pkgs.sdl2-compat}/lib -L${pkgs.lib.getLib pkgs.SDL2_ttf}/lib -L${pkgs.lib.getLib pkgs.SDL2_image}/lib -L${pkgs.lib.getLib pkgs.SDL2_mixer}/lib -L${pkgs.lib.getLib pkgs.freetype}/lib $LDFLAGS"
  '';
}
