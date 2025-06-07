# 使用官方Ubuntu 22.04镜像作为基础
FROM ubuntu:22.04

# 设置环境变量
ENV DEBIAN_FRONTEND=noninteractive \
    BUILD_DIR=/app/build

# 1. 替换为华为云镜像源并安装依赖
RUN cp /etc/apt/sources.list /etc/apt/sources.list.bak && \
    sed -i "s@http://.*archive.ubuntu.com@http://repo.huaweicloud.com@g" /etc/apt/sources.list && \
    sed -i "s@http://.*security.ubuntu.com@http://repo.huaweicloud.com@g" /etc/apt/sources.list && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        libopencv-dev \
        pkg-config \
        && \
    rm -rf /var/lib/apt/lists/*

# 2. 创建工作目录并复制代码
WORKDIR /app
COPY . .

# 3. 编译安装到系统目录
RUN mkdir -p ${BUILD_DIR} && \
    cd ${BUILD_DIR} && \
    cmake .. && \
    make -j$(nproc) && \
    make install && \
    ldconfig && \
    # 清理构建文件
    cd / && \
    # rm -rf ${BUILD_DIR} && \
    # 验证安装
    pkg-config --modversion opencv4



# 4. 设置运行时工作目录
WORKDIR /app/build

RUN ./demo ../assets/match/template.png ../assets/match/target.png

# 5. 默认启动命令
CMD ["/bin/bash"]