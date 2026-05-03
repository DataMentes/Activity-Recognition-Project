
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from model.transformers import (
    RANDOM_SEED,
    DuplicateFeatureFilter,
    VarianceFilter,
    CorrelationFilter,
    GroupwisePCA
)

def build_model(params: dict) -> Pipeline:
    """Constructs the complete Scikit-Learn Pipeline using the provided hyperparameters."""
    prep = Pipeline(
        [
            ("dup", DuplicateFeatureFilter()),
            ("var", VarianceFilter(params["var_th"])),
            ("corr", CorrelationFilter(params["corr_th"])),
            ("group_pca", GroupwisePCA(params["group_pca"], params["group_pca"])),
            ("final_pca", PCA(params["final_pca"], random_state=RANDOM_SEED)),
        ]
    )

    model = Pipeline(
        [
            ("prep", prep),
            (
                "svm",
                SVC(
                    C=params["C"],
                    kernel=params["kernel"],
                    gamma=params["gamma"],
                    random_state=RANDOM_SEED,
                ),
            ),
        ]
    )

    return model
