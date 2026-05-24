# --- Project Metadata ---
PROJECT_NAME := Deep-VQA-Framework
ROOT_DIR := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))
$(shell [ -f "$(ROOT_DIR)/Makefile" ] || (echo "❌ Error: Makefile not found in ROOT_DIR"; exit 1))

# Automatically determine the operating system environment
# The OS environment variable under Windows is usually Windows_NT
ifeq ($(OS),Windows_NT)
	PYTHON_CMD := $(shell if [ -f "$(ROOT_DIR)/.venv/Scripts/python.exe" ]; then echo "$(ROOT_DIR)/.venv/Scripts/python.exe"; else echo "python"; fi)
else
	PYTHON_CMD := $(shell if [ -f "$(ROOT_DIR)/.venv/bin/python" ]; then echo "$(ROOT_DIR)/.venv/bin/python"; else echo "python3"; fi)
endif

PYTHON := $(PYTHON_CMD)
LOG_DIR := $(ROOT_DIR)/results/scripts_logs

$(shell mkdir -p $(LOG_DIR))
$(info 📂 Project Root detected as: $(ROOT_DIR))
.PHONY: setup data link train check \
		clean help optimize archive stop test \
		validate fmt typecheck

# Make sure the first goal is to help
# When you type `make` directly in the terminal without any arguments,
# it will automatically execute the first target that appears in the file.
help:
	@echo "🛠️  $(PROJECT_NAME) Commands:"
	@echo "  make setup      - Install dependencies, optimize network"
	@echo "  make data       - Prepare datasets"
	@echo "  make link       - Set up symbolic links"
	@echo "  make train      - Start training in background (usage: make train [DATASET=name] [MODEL=name] [DEBUG=1])"
	@echo "  make test       - Run smoke test (DEBUG mode)"
	@echo "  make check      - Check training status (GPU/memory/process)"
	@echo "  make clean      - Clean caches and temp files"
	@echo "  make optimize   - Optimize network/Jupyter settings"
	@echo "  make archive    - Package results"
	@echo "  make stop       - Stop training processes"
	@echo "  make validate   - Validate code style and specifications"
	@echo "	 make fmt        - Format code with ruff"
	@echo "	 make typecheck   - Perform type checking with mypy"
	@echo ""
	@echo "💡 Tip: To customize training, e.g.:"
	@echo "       make train DATASET=coco MODEL=vit_b DEBUG=1"
	@echo ""
	@echo "📁 Config files located in: $(PROJECT_NAME)/config/"


# 1. Environment Initialization
setup:
	@echo "🔐 Setting script permissions...(Log: $(LOG_DIR)/setup_env.log)"
	@chmod +x $(ROOT_DIR)/scripts/*.sh
	if [ -d "$(ROOT_DIR)/.venv" ]; then \
		echo "✅ Environment already exists. Skipping setup."; \
	else \
		echo "⚙️  Setting up environment..." \
		cd $(ROOT_DIR)/scripts && bash setup_env.sh --mirror > $(LOG_DIR)/setup_env.log 2>&1 \
		echo "✅ Environment ready" \
	fi

# 2. Data Preparation
data:
	@echo "📦 Preparing datasets... (Log: $(LOG_DIR)/manage_data.log)"
	@cd $(ROOT_DIR)/scripts && bash manage_data.sh > $(LOG_DIR)/manage_data.log 2>&1
	@echo "✅ Data ready"


# 3. Symbolic Links
link:
	@echo "🔗 Setting up symbolic links... (Log: $(LOG_DIR)/setup_links.log)"
	@cd $(ROOT_DIR)/scripts && bash setup_links.sh --all > $(LOG_DIR)/setup_links.log 2>&1
	@echo "✅ Symbolic links ready"


# 4. Start training (runs in the background)
# nohup must be followed by a real, standalone executable file/program.
# Caution: This command would not run `make check` and `make optimize`
DATASET ?= tid2013
MODEL ?= resnet_iqa
DEBUG ?= 0

train:
	@if [ ! -f "$(PYTHON)" ]; then \
        echo "❌ Python environment not found. Please run 'make setup' first."; \
        exit 1; \
    fi

	@if [ ! -f $(ROOT_DIR)/scripts/download_flag ]; then \
		echo "⚠️  Dataset flag not found. Preparing datasets..."; \
		$(MAKE) data && $(MAKE) link; \
	fi
	@echo "🚀 Starting training in background... (Log: $(LOG_DIR)/train.log)"
	@if [ -f "$(LOG_DIR)/train.log" ] && [ $$(stat -c%s "$(LOG_DIR)/train.log") -gt 10485760 ]; then \
        mv "$(LOG_DIR)/train.log" "$(LOG_DIR)/train.log.$$(date +%Y%m%d%H%M%S).bak"; \
    fi
	@cd $(ROOT_DIR) && \
	export LOG_LEVEL=$$(if [ "$(DEBUG)" = "1" ]; then echo "DEBUG"; else echo "INFO"; fi); \
	nohup $(PYTHON) -m src.main \
		--dataset "$(DATASET)" \
		--model "$(MODEL)" \
		> $(LOG_DIR)/train.log 2>&1 &
	@echo "🔥 Training started. PID: $$!"
	@echo "Monitor with: tail -f $(LOG_DIR)/train.log"


test:
	@if [ ! -f "$(PYTHON)" ]; then \
        echo "❌ Python environment not found. Please run 'make setup' first."; \
        exit 1; \
    fi

	@if [ ! -f $(ROOT_DIR)/scripts/download_flag ]; then \
		echo "⚠️  Dataset flag not found. Preparing datasets..."; \
		$(MAKE) data && $(MAKE) link; \
	fi
	@echo "🧪 Running smoke test (DEBUG mode)... (Log: $(LOG_DIR)/smoke_test.log)"
	@if [ -f "$(LOG_DIR)/smoke_test.log" ]; then \
		mv "$(LOG_DIR)/smoke_test.log" "$(LOG_DIR)/smoke_test.log.$$(date +%Y%m%d%H%M%S).bak"; \
	fi
	@cd $(ROOT_DIR) && \
	export LOG_LEVEL=DEBUG; \
	$(PYTHON) -m src.main \
		--dataset "$(DATASET)" \
		--model "$(MODEL)" \
		--smoke_test \
		> $(LOG_DIR)/smoke_test.log 2>&1
	@echo "✅ Smoke test completed! Log: $(LOG_DIR)/smoke_test.log"



# 5. Status Audit
check:
	@echo "🔍 Checking system status... (Log: $(LOG_DIR)/system_check.log)"
	@cd $(ROOT_DIR)/scripts && bash system_check.sh > $(LOG_DIR)/system_check.log 2>&1


# 6. Optimize Environment (network/Jupyter settings)
optimize:
	@echo "🔧 Optimizing Environment... (Log: $(LOG_DIR)/optimize_env.log)"
	@cd $(ROOT_DIR)/scripts && bash optimize_env.sh > $(LOG_DIR)/optimize_env.log 2>&1


# 7. Clear cache
clean:
	@read -p "Are you sure you want to clean all caches? [y/N] " confirm; \
    if [ "$$confirm" = "y" ]; then \
		echo "🧹 Cleaning... (Log: $(LOG_DIR)/cache_clean.log)"; \
        bash $(ROOT_DIR)/scripts/cache_clean.sh > $(LOG_DIR)/cache_clean.log 2>&1; \
    else \
        echo "Clean aborted."; \
    fi


# 8. Packaging Results
archive:
	@echo "📦 Archiving... (Log: $(LOG_DIR)/archive.log)"
	@if [ -f "$(LOG_DIR)/archive.log" ]; then \
		mv "$(LOG_DIR)/archive.log" "$(LOG_DIR)/archive.log.$$(date +%Y%m%d%H%M%S).bak"; \
	fi
	@cd $(ROOT_DIR)/scripts && bash archive_results.sh --all > $(LOG_DIR)/archive.log 2>&1


# 9. Stop training processes
stop:
	@echo "🛑 Stopping training processes..."
	@ps aux | grep "[s]rc.main" | awk '{print $$2}' | xargs kill -15 2>/dev/null || echo "No training process found"
	@sleep 2
	@ps aux | grep "[s]rc.main" | awk '{print $$2}' | xargs kill -9 2>/dev/null || echo "No training process found"


# 10. Validate code style and specifications
validate:
	@echo "Verifying code style and specifications..."
	@uv sync
	@$(PYTHON) -m ruff format --check .
	@$(PYTHON) -m ruff check .
	@echo "Verification completed, code quality is good!"


# 11. Format code with ruff
fmt:
	@echo "Formatting code..."
	@uv sync
	@$(PYTHON) -m ruff format .
	@$(PYTHON) -m ruff check . --fix
	@echo "Code formatted successfully!"


# 12. Type checking with mypy
typecheck:
	@echo "Performing type checking with mypy..."
	@uv sync
	@$(PYTHON) -m mypy src/ --ignore-missing-imports
	@echo "Type checking completed!"