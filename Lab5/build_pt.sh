#!/bin/bash

# CUDA分离编译脚本 - 解决GCC 11兼容性问题

echo "=== CUDA分离编译开始 ==="

# 清理旧的目标文件
echo "清理旧文件..."
rm -f *.o main_gpu_pt

# 检查CUDA和GCC版本
echo "检查环境..."
nvcc --version | head -4
gcc --version | head -1

# 步骤1: 单独编译CUDA文件（使用兼容性选项）
echo "编译CUDA文件..."
nvcc -c guessing_cuda_multi_pt.cu -o guessing_cuda.o \
     -arch=sm_75 -O2 \
     --compiler-options "-fPIC" \
     --std c++11 \
     -I. 

if [ $? -ne 0 ]; then
    echo "❌ CUDA文件编译失败"
    exit 1
fi
echo "✅ CUDA文件编译成功"

# 步骤2: 使用g++编译C++文件
echo "编译C++文件..."

# main.cpp
g++ -c main.cpp -o main.o -O2 -fPIC -I.
if [ $? -ne 0 ]; then
    echo "❌ main.cpp编译失败"
    exit 1
fi

# train.cpp
g++ -c train.cpp -o train.o -O2 -fPIC -I.
if [ $? -ne 0 ]; then
    echo "❌ train.cpp编译失败"
    exit 1
fi

# guessing_yuan.cpp
g++ -c guessing_yuan.cpp -o guessing_yuan.o -O2 -fPIC -I.
if [ $? -ne 0 ]; then
    echo "❌ guessing_yuan.cpp编译失败"
    exit 1
fi

# md5.cpp
g++ -c md5.cpp -o md5.o -O2 -fPIC -I.
if [ $? -ne 0 ]; then
    echo "❌ md5.cpp编译失败"
    exit 1
fi

echo "✅ 所有C++文件编译成功"

# 步骤3: 链接所有目标文件
echo "链接目标文件..."
nvcc main.o train.o guessing_yuan.o guessing_cuda.o md5.o \
     -o main_gpu_pt -lcudart -arch=sm_75

if [ $? -ne 0 ]; then
    echo "❌ 链接失败"
    exit 1
fi

echo "✅ 链接成功"

# 清理目标文件
echo "清理临时文件..."
rm -f *.o

# 检查最终文件
if [ -f main_gpu_pt ]; then
    echo "🎉 编译完成！可执行文件: main_gpu"
    echo "文件大小: $(du -h main_gpu | cut -f1)"
    echo "运行测试: ./main_gpu"
else
    echo "❌ 编译失败，未生成可执行文件"
    exit 1
fi

echo "=== 编译完成 ==="