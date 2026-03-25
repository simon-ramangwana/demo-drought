# inspect_packages.py
import importlib
import inspect
from pprint import pprint

PACKAGE_NAMES = [
    "rozvidrought_datasets",
    "rozvidrought_inputs",
    "rozvidrought_subsystems",
    "rozvidrought",
]

def safe_signature(obj):
    try:
        return str(inspect.signature(obj))
    except Exception:
        return "(signature unavailable)"

for package_name in PACKAGE_NAMES:
    print("\n" + "=" * 80)
    print(f"PACKAGE: {package_name}")
    print("=" * 80)

    try:
        module = importlib.import_module(package_name)
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

    print("\nSUBMODULES:")
    try:
        package_path = getattr(module, "__path__", None)
        if package_path is None:
            print("Not a package with submodules.")
        else:
            import pkgutil
            subs = [m.name for m in pkgutil.iter_modules(package_path)]
            pprint(subs)
    except Exception as e:
        print(f"Could not inspect submodules: {e}")