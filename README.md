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
export FFMPEG_VERSION=7.1.1
export FFMPEG_INSTALL_DIR=${HOME}/software/ffmpeg-${FFMPEG_VERSION}
export PATH=${FFMPEG_INSTALL_DIR}/bin:${PATH}
export LD_LIBRARY_PATH=${FFMPEG_INSTALL_DIR}/lib:${LD_LIBRARY_PATH}
export PKG_CONFIG_PATH=${FFMPEG_INSTALL_DIR}/lib/pkgconfig:${PKG_CONFIG_PATH}
export PKG_CONFIG_PATH=${HOME}/software/nv-codec-headers-13.0.19.0/lib/pkgconfig:${PKG_CONFIG_PATH}

export OPENCV_VERSION=4.11.0
export OPENCV_INSTALL_DIR=${HOME}/software/opencv-${OPENCV_VERSION}
export LD_LIBRARY_PATH=${OPENCV_INSTALL_DIR}/lib:${LD_LIBRARY_PATH}
export PYTHONPATH=${OPENCV_INSTALL_DIR}/lib/python3.8/site-packages:${PYTHONPATH}
export CMAKE_PREFIX_PATH=${OPENCV_INSTALL_DIR}:${CMAKE_PREFIX_PATH}

export SPDLOG_INSTALL_DIR=${HOME}/software/spdlog-1.15.3
export LD_LIBRARY_PATH=${SPDLOG_INSTALL_DIR}/lib:${LD_LIBRARY_PATH}
export PKG_CONFIG_PATH=${SPDLOG_INSTALL_DIR}/lib/pkgconfig:${PKG_CONFIG_PATH}
export CMAKE_PREFIX_PATH=${SPDLOG_INSTALL_DIR}:${CMAKE_PREFIX_PATH}

export VMAF_INSTALL_DIR=${HOME}/software/vmaf-3.0.0
export PATH=${VMAF_INSTALL_DIR}/bin:${PATH}
export LD_LIBRARY_PATH=${VMAF_INSTALL_DIR}/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH}
export PKG_CONFIG_PATH=${VMAF_INSTALL_DIR}/lib/x86_64-linux-gnu/pkgconfig:${PKG_CONFIG_PATH}

export SVTAV1_INSTALL_DIR=${HOME}/software/svt-av1-1.6.0
export LD_LIBRARY_PATH=${SVTAV1_INSTALL_DIR}/lib:${LD_LIBRARY_PATH}
export PKG_CONFIG_PATH=${SVTAV1_INSTALL_DIR}/lib/pkgconfig:${PKG_CONFIG_PATH}
```


```
sudo apt install -y ec2-instance-connect pigz python3 yasm nasm libvpx-dev libopus-dev libssl-dev libfreetype6-dev libx264-dev libx265-dev libpython3-all-dev python3-dev python3-distutils pkg-config libopenblas-dev libeigen3-dev libxml2-dev rapidjson-dev zipmerge liblapack-dev build-essential cmake meson qt6-base-dev libqt6core5compat6-dev

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

wget https://gitlab.com/AOMediaCodec/SVT-AV1/-/archive/v1.6.0/SVT-AV1-v1.6.0.tar.gz -O ~/src/SVT-AV1-v1.6.0.tar.gz
tar xvf SVT-AV1-v1.6.0.tar.gz
cd Build ; cmake -G"Unix Makefiles" -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=${HOME}/software/svt-av1-1.6.0 ../ && make -j $(nproc) && make install

wget "https://github.com/Netflix/vmaf/archive/refs/tags/v3.0.0.tar.gz" -O ${HOME}/src/vmaf-3.0.0.tar.gz
tar xvf vmaf-3.0.0.tar.gz
meson setup libvmaf/build libvmaf --buildtype release -Denable_float=true -Dprefix=${HOME}/software/vmaf-3.0.0 && ninja -vC libvmaf/build
cd python && python3 setup.py build_ext --build-lib .
cd ..
meson setup libvmaf/build libvmaf --buildtype release -Dprefix=${HOME}/software/vmaf-3.0.0 && ninja -vC libvmaf/build install


git clone https://git.videolan.org/git/ffmpeg/nv-codec-headers.git
cd nv-codec-headers
git checkout -b release_n13.0.19.0 n13.0.19.0
make install PREFIX=${HOME}/software/nv-codec-headers-12.2

wget https://ffmpeg.org/releases/ffmpeg-7.0.2.tar.gz -O ~/src/ffmpeg-7.0.2.tar.gz
tar xvf ffmpeg-7.0.2.tar.gz
cd ffmpeg-7.0.2
./configure --prefix=${HOME}/software/ffmpeg-7.0.2 --enable-libxml2 --enable-libharfbuzz --enable-libfreetype --enable-gpl --enable-libx264 --enable-libx265 --enable-nonfree --enable-libopus --enable-libvpx --enable-openssl --enable-shared --enable-libsrt --enable-libpulse --enable-sdl2 --enable-nvenc --enable-ffnvcodec -enable-cuda-nvcc --enable-libnpp --extra-cflags=-I/usr/local/cuda/include --extra-ldflags=-L/usr/local/cuda/lib64 --disable-static

make -j $(nproc) && make install

cd ~/src
wget "https://github.com/gabime/spdlog/archive/refs/tags/v1.12.0.tar.gz" -O ~/src/spdlog-1.12.0.tar.gz
rm -rf build ; mkdir build ; cd build ; cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=${HOME}/software/spdlog-1.12.0 ../ && make -j $(nproc) && make install


wget "https://github.com/opencv/opencv/archive/refs/tags/4.11.0.tar.gz" -O ~/src/opencv-4.11.0.tar.gz
wget "https://github.com/opencv/opencv_contrib/archive/refs/tags/4.11.0.tar.gz" -O ~/src/opencv_contrib-4.11.0.tar.gz
tar xvf opencv-4.11.0.tar.gz
tar xvf opencv_contrib-4.11.0.tar.gz
cd opencv-4.11.0

sudo apt install python3-pip
python3 -m pip install numpy

# Add these on desktop
# -DBUILD_EXAMPLES=ON -DINSTALL_C_EXAMPLES=ON -DINSTALL_BIN_EXAMPLES=ON -DWITH_GTK=OFF -DWITH_QT=ON -DWITH_OPENGL=ON
rm -rf build ; mkdir build ; cd build ; cmake -DENABLE_CXX11=ON -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=${HOME}/software/opencv-4.11.0 -DWITH_TBB=ON -DBUILD_opencv_python2=OFF -DOPENCV_EXTRA_MODULES_PATH=${HOME}/src/opencv_contrib-4.11.0/modules -DWITH_CUDA=ON -DWITH_OPENCL=OFF -DOPENCV_GENERATE_PKGCONFIG=ON -DOPENCV_ENABLE_NONFREE=ON -DBUILD_opencv_rgbd=OFF ../ && make -j $(nproc) && make install
```

# Compile
rm -rf build ; mkdir build ; cd build ; cmake -DCMAKE_BUILD_TYPE=Release ../ && make -j$(nproc)

# FFmpeg

Some nifty commands:
```
# HW encode
ffmpeg -y -hwaccel cuda -i 20241215-lhdrs-hevc.mp4 -an -c:v hevc_nvenc -pix_fmt yuv420p -preset p5 -g 600 -bf 3 -b_ref_mode middle -profile:v main -level "5.1" -rc vbr -spatial-aq 1 -cq 24 -qmin 24 -qmax 50 -maxrate 11M -rc-lookahead 20 -bufsize:v 20M  hevc_nvenc_better.mp4

``` 
