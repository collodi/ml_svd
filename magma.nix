{ stdenv, fetchurl, cmake, openblas, gfortran, cudatoolkit, libpthreadstubs, liblapack }:

with stdenv.lib;

let version = "2.4.0";

in stdenv.mkDerivation {
  name = "magma-${version}";
  src = fetchurl {
    url = "https://icl.cs.utk.edu/projectsfiles/magma/downloads/magma-${version}.tar.gz";
    sha256 = "4eb839b1295405fd29c8a6f5b4ed578476010bf976af46573f80d1169f1f9a4f";
    name = "magma-${version}.tar.gz";
  };

  buildInputs = [ gfortran openblas cudatoolkit libpthreadstubs liblapack cmake ];

  doCheck = false;
  #checkTarget = "tests";

  enableParallelBuilding=true;

  # MAGMA's default CMake setup does not care about installation. So we copy files directly.
  installPhase = ''
    mkdir -p $out
    mkdir -p $out/include
    mkdir -p $out/lib
    mkdir -p $out/lib/pkgconfig
    cp -a ../include/*.h $out/include
    #cp -a sparse-iter/include/*.h $out/include
    cp -a lib/*.a $out/lib
    cat ../lib/pkgconfig/magma.pc.in                   | \
    sed -e s:@INSTALL_PREFIX@:"$out":          | \
    sed -e s:@CFLAGS@:"-I$out/include":    | \
    sed -e s:@LIBS@:"-L$out/lib -lcurand -lmagma -lmagma_sparse": | \
    sed -e s:@MAGMA_REQUIRED@::                       \
        > $out/lib/pkgconfig/magma.pc
  '';

  meta = with stdenv.lib; {
    description = "Matrix Algebra on GPU and Multicore Architectures";
    license = licenses.bsd3;
    homepage = http://icl.cs.utk.edu/magma/index.html;
    platforms = platforms.unix;
    maintainers = with maintainers; [ ianwookim ];
  };
}
