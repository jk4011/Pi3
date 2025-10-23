from setuptools import setup

setup(
    name="pi3",
    version="0.0.1",
    description="pi3",
    long_description_content_type="text/markdown",
    packages=["pi3"],
    python_requires=">=3.6, <4",
    entry_points={
        "console_scripts": [
            "pi3=pi3.example:main",
        ],
    },
    project_urls={
        "Source": "https://github.com/jk4011/Pi3",
    },
)
