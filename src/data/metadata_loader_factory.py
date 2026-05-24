# src/data/factory.py
from .metadata_loaders import KonvidLoader, T2VqaLoader, Tid2013Loader


class MetadataLoaderFactory:
    # Register the Loader; to add a new dataset, simply add one line here.
    _REGISTRY = {"konvid-1k": KonvidLoader, "t2vqa-db": T2VqaLoader, "tid2013": Tid2013Loader}

    @classmethod
    def get_loader(cls, dataset_name: str):
        key = dataset_name.lower().strip()  # Convert to lowercase
        loader_class = cls._REGISTRY.get(key)
        if not loader_class:
            available = list(cls._REGISTRY.keys())
            raise ValueError(f"❌ Unregistered dataset: {dataset_name}, Available: {available}")

        try:
            return loader_class()
        except Exception as e:
            raise RuntimeError(f"❌ Instantiation loader failed: {dataset_name}, {e}")
