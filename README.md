Install uv:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```
or 

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

```bash
uv venv
```
Activate the virtual environment

Download dependencies
```bash
uv sync
```

The entry point is `train`, to start training use:

```bash
train
```

To change configurations use hydra syntax, for example:

```bash
train trainer.epochs=2
```