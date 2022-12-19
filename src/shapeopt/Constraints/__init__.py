from .barycenter import Barycenter
from .volume import Volume
from .determinant import Determinant

constraints = {
    'volume': Volume,
    'barycenter': Barycenter,
    'determinant': Determinant,
}

