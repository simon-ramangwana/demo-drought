import importlib
import inspect
from pprint import pprint

MODULE_NAMES = [
    "rozvidrought.pipeline",
    "rozvidrought.models",
    "rozvidrought.schema",
    "rozvidrought.cli",
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
            doc = inspect.getdoc(obj)
            print((doc[:500] + "...") if doc and len(doc) > 500 else (doc or "No docstring"))

        elif inspect.isclass(obj):
            print(f"\n[CLASS] {name}{safe_signature(obj)}")
            class_doc = inspect.getdoc(obj)
            print((class_doc[:500] + "...") if class_doc and len(class_doc) > 500 else (class_doc or "No docstring"))

            methods = []
            for method_name, method_obj in inspect.getmembers(obj, predicate=inspect.isfunction):
                if not method_name.startswith("_"):
                    methods.append(f"  - {method_name}{safe_signature(method_obj)}")
            if methods:
                print("  METHODS:")
                print("\n".join(methods))

        elif callable(obj):
            print(f"\n[CALLABLE] {name}{safe_signature(obj)}")