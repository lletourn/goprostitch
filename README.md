# goprostitch
Stitch dual gopro pano

# Stitching instructions
Figure out which set of files are left and which set are right in your gopro directory

Gopro filenames have this form
GXxxyyyy.MP4

Where
xx == zero padded file index
yyyy == zero padded camera segment id

So all videos of one side will have the same yyyy with incrementing xx
 
```
P=`pwd`; for f in `ls -l GX0?0038.MP4 | sed 's/.* \(G.[0-9]\+\.MP4\)$/\1/g'`; do echo "file ${P}/${f}";done > left.concat
P=`pwd`; for f in `ls -l GX0?0036.MP4 | sed 's/.* \(G.[0-9]\+\.MP4\)$/\1/g'`; do echo "file ${P}/${f}";done > right.concat

# Concat and extract a ref image
for side in left right;
do
  ffmpeg -y -safe 0 -f concat -i ${side}.concat -codec copy -map 0:0 -map 0:1 20230723-lhdrs-hevc-${side}.mp4;
  ffmpeg -y -ss 20 -i 20230723-lhdrs-hevc-${side}.mp4 -frames:v 1 -update 1 -f image2 ${side}.png;
done
```

# Extract timecodes
```
ffmpeg -y -i GX010036.MP4 -codec copy -map 0:2 -f rawvideo 36.tmcd
ffmpeg -y -i GX010038.MP4 -codec copy -map 0:2 -f rawvideo 38.tmcd
xxd 36.tmcd
xxd 38.tmcd
```
Then you have a hex value of the frames since midnight. Just substract both to know what the offset is...in theory (in practice it doesn't seem to lineup anyways)

# Instance setup

.bashrc
```
export FFMPEG_VERSION=7.1.3
export FFMPEG_INSTALL_DIR=${HOME}/software/ffmpeg-${FFMPEG_VERSION}
export PATH=${FFMPEG_INSTALL_DIR}/bin:${PATH}
export LD_LIBRARY_PATH=${FFMPEG_INSTALL_DIR}/lib:${LD_LIBRARY_PATH}
export PKG_CONFIG_PATH=${FFMPEG_INSTALL_DIR}/lib/pkgconfig:${PKG_CONFIG_PATH}
export PKG_CONFIG_PATH=${HOME}/software/nv-codec-headers-13.0.19.0/lib/pkgconfig:${PKG_CONFIG_PATH}

export OPENCV_VERSION=4.13.0
export OPENCV_INSTALL_DIR=${HOME}/software/opencv-${OPENCV_VERSION}
export LD_LIBRARY_PATH=${OPENCV_INSTALL_DIR}/lib:${LD_LIBRARY_PATH}
export PYTHONPATH=${OPENCV_INSTALL_DIR}/lib/python3.8/site-packages:${PYTHONPATH}
export CMAKE_PREFIX_PATH=${OPENCV_INSTALL_DIR}:${CMAKE_PREFIX_PATH}

export SPDLOG_VERSION=1.17.0
export SPDLOG_INSTALL_DIR=${HOME}/software/spdlog-${SPDLOG_VERSION}
export LD_LIBRARY_PATH=${SPDLOG_INSTALL_DIR}/lib:${LD_LIBRARY_PATH}
export PKG_CONFIG_PATH=${SPDLOG_INSTALL_DIR}/lib/pkgconfig:${PKG_CONFIG_PATH}
export CMAKE_PREFIX_PATH=${SPDLOG_INSTALL_DIR}:${CMAKE_PREFIX_PATH}

export VMAF_INSTALL_DIR=${HOME}/software/vmaf-3.0.0
export PATH=${VMAF_INSTALL_DIR}/bin:${PATH}
export LD_LIBRARY_PATH=${VMAF_INSTALL_DIR}/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH}
export PKG_CONFIG_PATH=${VMAF_INSTALL_DIR}/lib/x86_64-linux-gnu/pkgconfig:${PKG_CONFIG_PATH}
```

```
sudo apt install -y pigz python3 yasm nasm libvpx-dev libopus-dev libssl-dev libfreetype6-dev libx264-dev libx265-dev libpython3-all-dev python3-dev python3-distutils pkg-config libopenblas-dev libeigen3-dev libxml2-dev rapidjson-dev zipmerge liblapack-dev build-essential cmake meson qt6-base-dev libqt6core5compat6-dev

mkdir ~/src
cd src
```

# 3rd party
```
cd ~/src

wget https://bootstrap.pypa.io/get-pip.py
sudo python3 ./get-pip.py
sudo python3 -m pip install -U --break-system-packages numpy Cython meson

curl -L "https://github.com/Netflix/vmaf/archive/refs/tags/v3.0.0.tar.gz" --remote-name
tar xvf vmaf-3.0.0.tar.gz
cd vmaf-3.0.0
meson setup libvmaf/build libvmaf --buildtype release -Denable_float=true -Dprefix=${HOME}/software/vmaf-3.0.0 && ninja -vC libvmaf/build
cd python && python3 setup.py build_ext --build-lib .
cd ..
meson setup libvmaf/build libvmaf --buildtype release -Dprefix=${HOME}/software/vmaf-3.0.0 && ninja -vC libvmaf/build install

curl -L "https://github.com/FFmpeg/nv-codec-headers/releases/download/n13.0.19.0/nv-codec-headers-13.0.19.0.tar.gz" --remote-name
tar xvf nv-codec-headers-13.0.19.0.tar.gz
cd nv-codec-headers-13.0.19.0
make install PREFIX=${HOME}/software/nv-codec-headers-13.0.19.0

wget https://ffmpeg.org/releases/ffmpeg-${FFMPEG_VERSION}.tar.gz -O ~/src/ffmpeg-${FFMPEG_VERSION}.tar.gz
tar xvf ffmpeg-${FFMPEG_VERSION}.tar.gz
cd ffmpeg-${FFMPEG_VERSION}
./configure --prefix=${HOME}/software/ffmpeg-${FFMPEG_VERSION} --enable-libxml2 --enable-libharfbuzz --enable-libfreetype --enable-gpl --enable-libx264 --enable-libx265 --enable-nonfree --enable-libopus --enable-libvpx --enable-openssl --enable-shared --enable-libsrt --enable-libpulse --enable-sdl2 --enable-nvenc --enable-ffnvcodec --enable-cuda-nvcc --enable-libnpp --extra-cflags=-I/usr/local/cuda/include --extra-ldflags=-L/usr/local/cuda/lib64 --disable-static

make -j $(nproc) && make install

curl -L "https://github.com/gabime/spdlog/archive/refs/tags/v${SPDLOG_VERSION}.tar.gz" -o spdlog-${SPDLOG_VERSION}.tar.gz
tar xvf spdlog-${SPDLOG_VERSION}.tar.gz
cd cd spdlog-${SPDLOG_VERSION}
rm -rf build ; mkdir build ; cd build ; cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=${HOME}/software/spdlog-${SPDLOG_VERSION} ../ && make -j $(nproc) && make install

curl -L "https://github.com/opencv/opencv/archive/refs/tags/${OPENCV_VERSION}.tar.gz" -o ~/src/opencv-${OPENCV_VERSION}.tar.gz
curl -L "https://github.com/opencv/opencv_contrib/archive/refs/tags/${OPENCV_VERSION}.tar.gz" -o ~/src/opencv_contrib-${OPENCV_VERSION}.tar.gz
tar xvf opencv-${OPENCV_VERSION}.tar.gz
tar xvf opencv_contrib-${OPENCV_VERSION}.tar.gz
cd opencv-${OPENCV_VERSION}

# Add these on desktop
# -DBUILD_EXAMPLES=ON -DINSTALL_C_EXAMPLES=ON -DINSTALL_BIN_EXAMPLES=ON -DWITH_GTK=OFF -DWITH_QT=ON -DWITH_OPENGL=ON
rm -rf build ; mkdir build ; cd build ; cmake -DENABLE_CXX11=ON -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=${HOME}/software/opencv-${OPENCV_VERSION} -DWITH_TBB=ON -DBUILD_opencv_python2=OFF -DOPENCV_EXTRA_MODULES_PATH=${HOME}/src/opencv_contrib-${OPENCV_VERSION}/modules -DWITH_CUDA=ON -DWITH_NVCUVID=OFF -DWITH_NVCUVENC=OFF -DWITH_OPENCL=OFF -DOPENCV_GENERATE_PKGCONFIG=ON -DOPENCV_ENABLE_NONFREE=ON -DBUILD_opencv_rgbd=OFF ../
make -j $(nproc) && make install
```

# Compile
rm -rf build ; mkdir build ; cd build ; cmake -DCMAKE_BUILD_TYPE=Release ../ && make -j$(nproc)

# FFmpeg

Some nifty commands:
```
# HW encode
ffmpeg -y -hwaccel cuda -i 20241215-lhdrs-hevc.mp4 -an -c:v hevc_nvenc -pix_fmt yuv420p -preset p5 -g 600 -bf 3 -b_ref_mode middle -profile:v main -level "5.1" -rc vbr -spatial-aq 1 -cq 24 -qmin 24 -qmax 50 -maxrate 11M -rc-lookahead 20 -bufsize:v 20M  hevc_nvenc_better.mp4

``` 

# Flash detect to see camera timing drift:
```
P=`pwd`; for f in `ls -l GX*.MP4 | sed 's/.* \(G.[0-9]\+\.MP4\)$/\1/g'`; do echo "file ${P}/${f}";done > concat.txt^C
ffmpeg -y -nostdin -safe 0 -f concat -i concat.txt -codec copy -map 0:0 -map 0:1 concat.mp4

source ~/software/python3.12-venv/syncvideos/bin/activate
python3 flashdetect.py --video concat.mp4
```
