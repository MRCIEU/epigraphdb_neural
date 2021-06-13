"""
HACK: Dumps /models_api/app/funcs/text_processing here
"""
import importlib

from funcs.utils import find_project_root

_root = find_project_root()
assert _root.exists()
_module_path = (
    _root.parent / "models_api" / "app" / "funcs" / "text_processing.py"
)
assert _module_path.exists()

_spec = importlib.util.spec_from_file_location(
    "model_text_processing", _module_path
)
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)  # type: ignore

preprocess_encode = _module.preprocess_encode  # type: ignore
