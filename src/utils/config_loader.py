# src/utils/config_loader.py
from collections.abc import Mapping
from pathlib import Path

import yaml
from loguru import logger


# Automatically scan config/models/*.yaml
def discover_models(config_dir: Path) -> dict:
    """Automatically discover configuration files in the models directory"""
    models_dir = config_dir / "models"
    if not models_dir.exists():
        return {}

    model_map = {}
    for yaml_file in models_dir.glob("*.yaml"):
        # resnet_iqa.yaml -> resnet_iqa
        model_name = yaml_file.stem
        model_map[model_name] = f"models/{yaml_file.name}"
    return model_map


MODEL_MAP = None  # Lazy loading


def get_model_map(config_dir: Path) -> dict:
    global MODEL_MAP
    if MODEL_MAP is None:
        MODEL_MAP = discover_models(config_dir)
        # Manual mapping as backup
        MODEL_MAP.update({"resnet_iqa": "models/resnet_iqa.yaml", "timeswin_vqa": "models/timeswin_vqa.yaml"})
    return MODEL_MAP


def deep_update(source, overrides):
    """Recursively merge dictionaries, including nested dictionaries."""
    for k, v in overrides.items():
        if isinstance(v, Mapping) and v:
            source[k] = deep_update(source.get(k, {}), v)
        else:
            source[k] = v
    return source


def safe_load_yaml(path: Path, description: str = "configuration file") -> dict:
    """Safely load YAML files, throw an exception on failure."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            content = yaml.safe_load(f)
            if content is None:
                raise ValueError(f"{description} is empty: {path}")
            return content
    except FileNotFoundError as e:
        logger.error(f"❌ {description} does not exist: {path}")
        raise FileNotFoundError(f"{description} does not exist: {path}") from e
    except yaml.YAMLError as e:
        logger.error(f"❌ YAML formatting error: {path}")
        logger.error(f"   Error details: {e}")
        # Help: Check YAML syntax, especially indentation and special characters.
        raise RuntimeError(f"YAML formatting error: {path}") from e
    except Exception as e:
        logger.error(f"❌ Failed to read {description}: {path}, Error: {e}")
        raise


def load_system_config(model_cfg_name: str, dataset_name: str) -> dict:
    config_dir = Path("config")

    # 1. Load basic configuration
    basic_path = config_dir / "basic.yaml"
    config = safe_load_yaml(basic_path, "Basic configuration file")

    # 2. Load model configuration
    model_key = str(model_cfg_name).strip().lower()
    model_map = get_model_map(config_dir)
    target_model_file = model_map.get(model_key, "models/resnet_iqa.yaml")
    model_path = config_dir / target_model_file

    if not model_path.exists():
        logger.warning(f"⚠️ Model config [{model_path}] not found. Falling back to resnet_iqa.yaml")
        model_path = config_dir / "models/resnet_iqa.yaml"

    model_config = safe_load_yaml(model_path, f"Model configuration file [{model_key}]")
    config = deep_update(config, model_config)

    # 3. Load dataset configuration
    dataset_cfg_path = config_dir / "dataset_config.yaml"
    ds_all = safe_load_yaml(dataset_cfg_path, "Dataset configuration file")

    ds_all_lowered = {k.lower(): v for k, v in ds_all.items()}
    target_ds_key = str(dataset_name).strip().lower()

    if target_ds_key not in ds_all_lowered:
        available = list(ds_all.keys())
        logger.error(f"❌ Dataset '{dataset_name}' not found in dataset_config.yaml")
        logger.info(f"   Available datasets: {available}")
        breakpoint()
        raise KeyError(f"Dataset settings for '{dataset_name}' missing. Available: {available}")

    dataset_info = ds_all_lowered[target_ds_key]

    # Command-line arguments are prioritized; the name in YAML is not used.
    config["dataset_info"] = dataset_info
    config["dataset_name"] = dataset_name.lower()  # Command line arguments to lowercase

    logger.info(f"⚙️ [Config Engine] Layered configuration successfully built for Model [{model_key}] & Dataset [{config['dataset_name']}]")

    logger.debug(f"[Config] Model configuration: {model_key}, Dataset: {config['dataset_name']}")
    logger.debug(f"[Config] Training configuration: epochs={config.get('train', {}).get('epochs', 'N/A')}")

    return config
