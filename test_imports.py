import streamlit as st

# Test imports with error handling
st.title("Import Test Dashboard")

# Test plotly import
try:
    import plotly.express as px
    import plotly.graph_objects as go
    st.success("✅ Plotly imported successfully!")
    
    # Test basic plotly functionality
    import pandas as pd
    import numpy as np
    
    # Create sample data
    df = pd.DataFrame({
        'x': range(10),
        'y': np.random.randn(10)
    })
    
    # Create a simple plot
    fig = px.line(df, x='x', y='y', title='Test Plot')
    st.plotly_chart(fig)
    
except ImportError as e:
    st.error(f"❌ Plotly import failed: {str(e)}")
    st.stop()
    
# Test pandas
try:
    import pandas as pd
    st.success("✅ Pandas imported successfully!")
except ImportError as e:
    st.error(f"❌ Pandas import failed: {str(e)}")
    
# Test numpy
try:
    import numpy as np
    st.success("✅ Numpy imported successfully!")
except ImportError as e:
    st.error(f"❌ Numpy import failed: {str(e)}")

st.success("🎉 All imports successful! Your app should work now.")