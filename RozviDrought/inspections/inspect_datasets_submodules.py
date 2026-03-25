# inspect_datasets_submodules.py
import importlib
import inspect
from pprint import pprint

MODULE_NAMES = [
    "rozvidrought_datasets.grid",
    "rozvidrought_datasets.locator",
    "rozvidrought_datasets.extractor",
]

def safe_signature(obj):
    try:
        return str(inspect.signature(obj))
    except Exception:
        return "(signature unavailable)"

for module_name in MODULE_NAMES:
    print("\n" + "=" * 80)
    print(f"MODULE: {module_name}")
    print("=" * 80)

    try:
        module = importlib.import_module(module_name)
    except Exception as e:
        print(f"FAILED TO IMPORT: {e}")
        continue

    print(f"MODULE FILE: {getattr(module, '__file__', 'unknown')}")
    print(f"MODULE DOC : {inspect.getdoc(module) or 'No module docstring'}")

    public_names = [name for name in dir(module) if not name.startswith("_")]
    print("\nPUBLIC NAMES:")
    pprint(public_names)

    print("\nCALLABLES / CLASSES:")
    for name in public_names:
        try:
            obj = getattr(module, name)
        except Exception as e:
            print(f"- {name}: <failed to access: {e}>")
            continue

        if inspect.isfunction(obj):
            print(f"\n[FUNCTION] {name}{safe_signature(obj)}")
            print(inspect.getdoc(obj) or "No docstring")

        elif inspect.isclass(obj):
            print(f"\n[CLASS] {name}{safe_signature(obj)}")
            print(inspect.getdoc(obj) or "No docstring")