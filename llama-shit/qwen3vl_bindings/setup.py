from pathlib import Path
from setuptools import setup

setup(
    name="qwen-mtmd",
    version="0.1.0",
    description="Patched Qwen3-VL MTMD pybind11 binding",
    author="facebookresearch",
    python_requires=">=3.10",
    packages=["qwen_mtmd"],
    package_dir={"qwen_mtmd": "qwen_mtmd"},
    package_data={"qwen_mtmd": ["qwen_mtmd.cpython-310-x86_64-linux-gnu.so"]},
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3.10",
        "Operating System :: POSIX :: Linux",
        "Topic :: Multimedia :: Graphics :: Image Processing",
    ],
    zip_safe=False,
)
