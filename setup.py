import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

with open('requirements.txt') as fp:
    install_requires = fp.read()

setuptools.setup(
    name='retina_face_net',
    version='0.1',
    author="biubug6, Paweł Pęczek",
    author_email="pawel.m.peczek@gmail.com",
    description="Inference wrapper for a model published in "
                "https://github.com/biubug6/Pytorch_Retinaface",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/PawelPeczek/RetinaFaceNet",
    packages=setuptools.find_packages(),
    install_requires=install_requires,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ]
)