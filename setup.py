from setuptools import find_packages, setup

setup(
    name="battery-xai-physics",
    version="0.1.0",
    description="Physics-guided explainable AI for cross-dataset lithium-ion degradation analysis.",
    author="Battery XAI Research Team",
    package_dir={"": "src"},
    packages=find_packages("src"),
    python_requires=">=3.10",
    install_requires=[
        "numpy>=1.26", "pandas>=2.2", "scipy>=1.11", "scikit-learn>=1.4",
        "matplotlib>=3.8", "pyyaml>=6.0", "joblib>=1.3", "h5py>=3.10",
    ],
    extras_require={
        "full": ["polars", "xgboost", "lightgbm", "shap", "optuna", "torch", "tensorboard", "plotly", "seaborn", "statsmodels"],
        "dev": ["pytest", "ruff"],
    },
    entry_points={"console_scripts": ["battery-xai=battery_xai.cli:main"]},
)
