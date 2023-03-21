# goprostitch
Stitch dual gopro pano

# Instance setup

.bashrc
```
export FFMPEG_VERSION=5.1.2
export FFMPEG_INSTALL_DIR=/home/ubuntu/software/ffmpeg-${FFMPEG_VERSION}
export PATH=${FFMPEG_INSTALL_DIR}/bin:${PATH}
export LD_LIBRARY_PATH=${FFMPEG_INSTALL_DIR}/lib:${LD_LIBRARY_PATH}
export PKG_CONFIG_PATH=${FFMPEG_INSTALL_DIR}/lib/pkgconfig:${PKG_CONFIG_PATH}

export OPENCV_VERSION=4.7.0
export OPENCV_INSTALL_DIR=/home/ubuntu/software/opencv-${OPENCV_VERSION}
export LD_LIBRARY_PATH=${OPENCV_INSTALL_DIR}/lib:${LD_LIBRARY_PATH}
export PYTHONPATH=${OPENCV_INSTALL_DIR}/lib/python3.8/site-packages:${PYTHONPATH}
export CMAKE_PREFIX_PATH=${OPENCV_INSTALL_DIR}:${CMAKE_PREFIX_PATH}

export SPDLOG_INSTALL_DIR=/home/ubuntu/software/spdlog-1.11.0
export LD_LIBRARY_PATH=${SPDLOG_INSTALL_DIR}/lib:${LD_LIBRARY_PATH}
export PKG_CONFIG_PATH=${SPDLOG_INSTALL_DIR}/lib/pkgconfig:${PKG_CONFIG_PATH}
export CMAKE_PREFIX_PATH=${SPDLOG_INSTALL_DIR}:${CMAKE_PREFIX_PATH}

export VMAF_INSTALL_DIR=/home/ubuntu/software/vmaf-2.3.1
export PATH=${VMAF_INSTALL_DIR}/bin:${PATH}
export LD_LIBRARY_PATH=${VMAF_INSTALL_DIR}/lib:${LD_LIBRARY_PATH}
export PKG_CONFIG_PATH=${VMAF_INSTALL_DIR}/lib/pkgconfig:${PKG_CONFIG_PATH}

```


```
sudo apt install -y ec2-instance-connect pigz python3 yasm libvpx-dev libopus-dev libssl-dev libfreetype6-dev libx264-dev libx265-dev libpython3-all-dev python3-dev python3-distutils pkg-config libopenblas-dev libeigen3-dev libxml2-dev rapidjson-dev zipmerge liblapack-dev build-essential cmake meson
# It's too old for the moment:
# libsvtav1-dev svt-av1

mkdir ~/src
cd src
```

# 3rd party
```
cd ~/src

wget https://bootstrap.pypa.io/get-pip.py
sudo python3 ./get-pip.py
sudo python3 -m pip install -U numpy Cython meson

wget https://gitlab.com/AOMediaCodec/SVT-AV1/-/archive/v1.4.1/SVT-AV1-v1.4.1.tar.gz -O ~/src/SVT-AV1-v1.4.1.tar.gz
tar xvf SVT-AV1-v1.4.1.tar.gz
cd Build ; cmake -G"Unix Makefiles" -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/home/ubuntu/software/svt-av1-1.4.1 ../ && make -j $(nproc) && make install

wget "https://github.com/Netflix/vmaf/archive/refs/tags/v2.3.1.tar.gz" -O /home/ubuntu/src/vmaf-2.3.1.tar.gz
tar xvf vmaf-2.3.1.tar.g
# From Makefile modified for target dir
cd third_party/libsvm && make lib
cd ../..
meson setup libvmaf/build libvmaf --buildtype release -Denable_float=true -Dprefix=/data/software/vmaf-2.3.1 && ninja -vC libvmaf/build
cd python && python3 setup.py build_ext --build-lib .
cd ..
meson setup libvmaf/build libvmaf --buildtype release -Dprefix=/data/software/vmaf-2.3.1 && ninja -vC libvmaf/build install


wget https://ffmpeg.org/releases/ffmpeg-5.1.2.tar.gz -O ~/src/ffmpeg-5.1.2.tar.gz
tar xvf ffmpeg-5.1.2.tar.gz
cd ffmpeg-5.1.2
./configure --prefix=${HOME}/software/ffmpeg-5.1.2 --enable-libxml2 --enable-libfreetype --enable-gpl --enable-libx264 --enable-libx265 --enable-libsvtav1 --enable-nonfree --enable-libopus --enable-libvpx --enable-openssl --enable-vmaf --enable-shared

cd ~/src
wget "https://github.com/gabime/spdlog/archive/refs/tags/v1.11.0.tar.gz" -O ~/src/spdlog-1.11.0.tar.gz

rm -rf build ; mkdir build ; cd build ; cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=${HOME}/software/spdlog-1.11.0 ../ && make -j4 && make install


wget "https://github.com/opencv/opencv/archive/refs/tags/4.7.0.tar.gz" -O ~/src/opencv-4.7.0.tar.gz
wget "https://github.com/opencv/opencv_contrib/archive/refs/tags/4.7.0.tar.gz" -O ~/src/opencv_contrib-4.7.0.tar.gz
tar xvf opencv-4.7.0.tar.gz
tar xvf opencv_contrib-4.7.0.tar.gz
rm -rf build ; mkdir build ; cd build ; cmake -DENABLE_CXX11=ON -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/home/ubuntu/software/opencv-4.7.0 -DBUILD_opencv_python2=OFF -DOPENCV_EXTRA_MODULES_PATH=/home/ubuntu/src/opencv_contrib-4.7.0/modules -DENABLE_FAST_MATH=1 ../
```

# Compile
rm -rf build ; mkdir build ; cd build ; cmake -DCMAKE_BUILD_TYPE=Release ../ && make -j4
