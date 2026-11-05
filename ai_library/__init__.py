__version__ = "1.0.0"

import importlib
from types import ModuleType

# Map attribute names to (module_path, attribute_name). If attribute_name
# is None, the module object itself is returned.
_LAZY_MAP = {
	"validate_config": (".codebase.helpers.config_helper", "validate_config"),
	"update_config": (".codebase.helpers.config_helper", "update_config"),
	"show_config": (".codebase.helpers.config_helper", "show_config"),
	"add_to_cron": (".codebase.setup.cron_manager", "add_to_cron"),
	"remove_from_cron": (".codebase.setup.cron_manager", "remove_from_cron"),
	"train": (".codebase.setup.train", "get_last_window_data_and_train"),
	"infer": (".codebase.setup.infer", "inference"),
	"injest": (".codebase.setup.injest", "injest_metrics"),
}


def __getattr__(name: str):
	if name in _LAZY_MAP:
		mod_path, attr = _LAZY_MAP[name]
		module = importlib.import_module(mod_path, __name__)
		if attr is None:
			value = module
		else:
			value = getattr(module, attr)
		globals()[name] = value
		return value
	raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
	return sorted(list(globals().keys()) + list(_LAZY_MAP.keys()))


__all__ = list(_LAZY_MAP.keys())


