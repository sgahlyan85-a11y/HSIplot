"""hsiplot: plot-level validation of hyperspectral spectra.

Typical use::

    from hsiplot import load_table, PreprocessConfig, preprocess_table, run_classification

    table = load_table("spectra.csv", group_col="group", control_label="control")
    table = preprocess_table(table, PreprocessConfig())
    result = run_classification(table, n_perm=None)
    print(result.summary())
"""
from .calibrate import to_reflectance, white_reference
from .io import SpectraTable, build_tables, load_table
from .permanova import group_difference_test
from .power import detectability
from .preprocess import PreprocessConfig, band_noise_report, plot_qc, preprocess_table
from .synthetic import make_synthetic_table
from .trait import run_trait_model
from .validate import ClassificationResult, random_split_demo, run_classification

__version__ = "1.0.0"
__all__ = [
    "SpectraTable", "load_table", "build_tables", "PreprocessConfig", "preprocess_table", "plot_qc",
    "band_noise_report", "run_classification", "ClassificationResult", "random_split_demo", "run_trait_model",
    "detectability", "group_difference_test", "white_reference", "to_reflectance", "make_synthetic_table", "__version__",
]
