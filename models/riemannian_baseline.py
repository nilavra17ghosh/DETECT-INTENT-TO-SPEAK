"""
Riemannian Geometry Baseline for EEG Classification
=====================================================

Covariance → Tangent-Space → Logistic Regression.

This is the PRIMARY STRONG BASELINE for low-channel EEG classification.
Consistently the hardest-to-beat method on low-channel BCI tasks — often
beats deep models. Use this to show you understand classical methods,
not just deep learning.

Why it works:
  EEG covariance matrices live on the Riemannian manifold of symmetric
  positive definite (SPD) matrices.  Projecting to the tangent space at
  the geometric mean (Fréchet mean) gives a flat, Euclidean representation
  that is more informative than raw EEG samples for classification.

Interview note — "Why Riemannian baseline?":
  "It exploits the geometry of EEG covariance matrices and is extremely
   strong for low-channel BCIs. If our deep model can't beat this, it's
   a signal that the deep model is overfitting."

Requires: pip install pyriemann
"""

from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression

try:
    from pyriemann.estimation import Covariances
    from pyriemann.tangentspace import TangentSpace
    _HAS_PYRIEMANN = True
except ImportError:
    _HAS_PYRIEMANN = False

def build_riemannian_classifier(C=1.0):
    """
    Build a Riemannian tangent-space + Logistic Regression pipeline.

    This computes sample covariance matrices from each epoch, maps them
    to the tangent space at the geometric mean (Fréchet mean on the SPD
    manifold), and classifies with regularised logistic regression.

    Pipeline: Raw EEG (n_trials, n_ch, n_samples)
              → Covariances(OAS)  (n_trials, n_ch, n_ch)
              → TangentSpace()    (n_trials, n_ch*(n_ch+1)/2)
              → LogisticRegression(C)

    Args:
        C (float): Inverse regularisation strength for LogisticRegression.
                   C=1.0 is a good default; reduce if overfitting.

    Returns:
        sklearn.pipeline.Pipeline: Fitted with .fit(X, y), .predict(X).
    """
    if not _HAS_PYRIEMANN:
        raise ImportError(
            "pyriemann is required for the Riemannian baseline.\n"
            "Install with: pip install pyriemann"
        )
    return make_pipeline(
        Covariances(estimator='oas'),   
        TangentSpace(),                 
        LogisticRegression(C=C, max_iter=1000, solver='lbfgs'),
    )
