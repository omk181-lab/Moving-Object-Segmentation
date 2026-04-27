from setuptools import setup, find_packages

setup(
    name="motion_seg_pipeline",
    version="1.0.0",
    description="Motion-Aware Video Segmentation Pipeline using RAFT + SAMURAI",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.3.1",
        "torchvision>=0.18.1",
        "numpy>=1.24.0",
        "opencv-python>=4.8.0",
        "Pillow>=10.0.0",
        "scipy>=1.11.0",
        "filterpy>=1.4.5",
        "PyYAML>=6.0",
        "tqdm>=4.65.0",
        "matplotlib>=3.7.0",
        "imageio[ffmpeg]>=2.31.0",
    ],
)
