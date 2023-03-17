# goprostitch
Stitch dual gopro pano

# 3rd party
```
wget "https://github.com/gabime/spdlog/archive/refs/tags/v1.11.0.tar.gz" -O spdlog-1.11.0.tar.gz

rm -rf build ; mkdir build ; cd build ; cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/data/software/spdlog-1.11.0 ../ && make -j4 && make install
```

# Compile
rm -rf build ; mkdir build ; cd build ; cmake -DCMAKE_BUILD_TYPE=Release ../ && make -j4
