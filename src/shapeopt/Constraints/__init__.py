from .barycenter import Barycenter
from .volume import Volume
from .volume_solid import VolumeSolid
from .determinant import Determinant

constraints = {
    'volume': Volume,
    'volume_solid': VolumeSolid,
    'barycenter': Barycenter,
    'determinant': Determinant,
}

