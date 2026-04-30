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
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, cross_val_predict

try:
    from pyriemann.estimation import Covariances
    from pyriemann.tangentspace import TangentSpace
    _HAS_PYRIEMANN = True
except ImportError:
    _HAS_PYRIEMANN = False

def build_riemannian_classifier(C=1.0, class_weight='balanced'):
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
        class_weight: Weighting strategy for classes to handle imbalance.

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
        LogisticRegression(C=C, max_iter=1000, solver='lbfgs', class_weight=class_weight),
    )

def build_riemannian_classifier_cv(C=1.0, class_weight='balanced', cv=5):
    """
    Build a Riemannian classifier wrapped with cross-validation logic.
    Provides probability estimates via cross_val_predict if needed.
    """
    clf = build_riemannian_classifier(C=C, class_weight=class_weight)
    return clf

def get_riemannian_probas(clf, X, y=None, cv=None):
    """
    Get probability estimates for a Riemannian classifier.
    If cv is provided (e.g. 5), returns cross-validated probabilities.
    Otherwise, returns probabilities from the fitted classifier.
    """
    if cv is not None and y is not None:
        skf = StratifiedKFold(n_splits=cv)
        probas = cross_val_predict(clf, X, y, cv=skf, method='predict_proba')
        return probas[:, 1]
    
    # If already fitted
    if hasattr(clf, "predict_proba"):
        return clf.predict_proba(X)[:, 1]
    else:
        # Fallback to decision function normalized
        if hasattr(clf, "decision_function"):
            d = clf.decision_function(X)
            return 1 / (1 + np.exp(-d))
        raise ValueError("Classifier does not support probability estimation")
