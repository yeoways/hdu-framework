mkdir build
cd build
cmake -DPYTHON_EXECUTABLE=/usr/bin/python3.9 ..
make
./c2p_test