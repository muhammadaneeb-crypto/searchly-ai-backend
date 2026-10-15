# Searchly AI backend

See the main [README](../README.md) for full setup. Quick start (Windows PowerShell):

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env      # then put your Groq key in .env
python run.py
pytest
```
