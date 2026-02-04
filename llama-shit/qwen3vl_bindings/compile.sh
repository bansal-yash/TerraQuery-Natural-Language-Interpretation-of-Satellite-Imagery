#!/bin/bash
g++ gpu_test.cpp \
  -I../llama.cpp/include \
  -I../llama.cpp/ggml/include \
  -I../llama.cpp/tools/stb \
  -I../llama.cpp/tools/mtmd \
  -I../llama.cpp \
  -L../llama.cpp/build/bin \
  -lllama -lmtmd -lggml-cuda -lggml -ldl -lpthread -O3 -o gpu_test
