{ pkgs ? import <nixpkgs> {}, cudaSupport ? false }:

with pkgs;
let

  python =
    let

      packageOverrides = self: super: rec {
        pytorch = super.pytorch.override {
          cudaSupport = config.cudaSupport;
          cudatoolkit = cudatoolkit_10;
          cudnn = cudnn_cudatoolkit_10;
        };

        torchvision = super.torchvision.override {
          pytorch = pytorch;
        };
      };

    in (python3.override { inherit packageOverrides; }).withPackages(ps:
      with ps; [ pytorch torchvision termcolor graph-tool ]
    );

    magma = callPackage ./magma.nix { cudatoolkit = cudatoolkit_10; };

in stdenv.mkDerivation rec {
  name = "ml-svd";

  buildInputs = [ python ]
    ++ lib.optional cudaSupport [ magma ]
  ;

  env = buildEnv { name = name; paths = buildInputs; };
}
