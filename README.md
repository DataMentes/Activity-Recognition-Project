# Activity Recognition Streamlit App

## Run

```bash
[Live](https://activity-recognition-project-0.streamlit.app/)```

## Deployment (Streamlit Community Cloud)

1. Push this project to a GitHub repository.
2. In Streamlit Community Cloud, create a new app from that repository.
3. Set:
   - Main file path: `app.py`
   - Python version: read from `runtime.txt`
4. Streamlit will install dependencies from `requirements.txt` automatically.

## Project Files Needed for Deploy

- `app.py`
- `best_svm_pipeline.pkl`
- `data/test.csv`
- `model/` package
- `requirements.txt`
- `.streamlit/config.toml`
- `runtime.txt`

## Notes

- The app loads a pre-trained model artifact (`best_svm_pipeline.pkl`). Keep this file in the repository for deployment.
- If deployment platform has repository file size limits, use Git LFS for large artifacts.
