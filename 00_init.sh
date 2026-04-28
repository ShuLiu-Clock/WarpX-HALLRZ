cmake -S . -B build \
    -DWarpX_DIMS=RZ \
    -DWarpX_APP=OFF \
    -DWarpX_LIB=ON \
    -DWarpX_OPENPMD=ON \
    -DopenPMD_USE_HDF5=ON \
    -DHDF5_ROOT=/usr/local/hdf5-parallel \
    -DWarpX_MPI=ON \
    -DWarpX_PYTHON=ON \
    -DWarpX_EB=ON
