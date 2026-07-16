import os
import sys

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "true"
os.environ["HF_HUB_DISABLE_SYMLINKS"] = "true"

if __name__ == "__main__":
    from uvicorn.main import main
    sys.argv = ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
    main()