# Streamlit Deployment Troubleshooting Guide

## Issue Fixed
The main issue was a **missing plotly dependency** during Streamlit Cloud deployment.

## What Was Wrong
1. **ModuleNotFoundError: No module named 'plotly'** - The deployment couldn't find the plotly library
2. Multiple conflicting requirements files existed
3. Some requirements files contained code instead of dependencies

## What Was Fixed
1. ✅ **Created clean requirements.txt** with specific versions:
   - `streamlit==1.28.0`
   - `pandas==2.0.3` 
   - `numpy==1.24.3`
   - `plotly==5.17.0`

2. ✅ **Removed conflicting files**:
   - Deleted `CPI_Dashboard\requirements.txt` (contained code instead of dependencies)
   - Removed duplicate `streamlit_requirements.txt`

3. ✅ **Added error handling** to the imports in your app

4. ✅ **Created test app** (`test_imports.py`) to verify imports work

## Deployment Steps
1. **Push to GitHub**: Ensure your repository has the updated `requirements.txt`
2. **Redeploy on Streamlit Cloud**: 
   - Go to your Streamlit Cloud dashboard
   - Click "Reboot app" or redeploy from GitHub
   - The new requirements.txt should be detected and installed

3. **Monitor deployment logs** for any remaining issues

## Files to Include in Your Repository
Make sure these files are in your repository root:
- `streamlit_app.py` (your main app)
- `requirements.txt` (the fixed dependencies)
- `CPI_cleaned_enhanced.csv` or `cleaned_Data.csv` (your data)
- `Regions.json` or `philippines-region.geojson` (your map data)

## Test Locally
Before deploying, test locally:
```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## If Issues Persist
1. **Clear cache**: In Streamlit Cloud, use "Clear cache" option
2. **Check file names**: Ensure data files match what your code expects
3. **Verify file locations**: Data files should be in repository root
4. **Use test app**: Run `streamlit run test_imports.py` to verify imports

## Expected Result
Your Streamlit app should now deploy successfully without the plotly import error.