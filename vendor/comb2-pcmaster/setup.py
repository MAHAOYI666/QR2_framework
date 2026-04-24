from setuptools import setup

setup(
    name="comb2-pcmaster",
    version="0.1.0",
    description="comb2 pcmaster backtest package",
    python_requires=">=3.10",
    install_requires=["numpy", "pandas", "matplotlib"],
    packages=["comb2_pcmaster"],
)
