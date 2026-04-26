import streamlit as st
import rasterio
import matplotlib.pyplot as plt

st.title("IndiSight-Nowcast Dashboard")
st.write("Spatio-Temporal District Reconstruction using Prithvi-100M")

st.sidebar.header("Parameters")
band = st.sidebar.selectbox("Select Spectral Band", ["NIR", "Red", "Green", "SWIR-1"])

st.info("The model is currently running inference on the HPC cluster. Visualization of reconstructed HLS data below.")

# Placeholder for your Figure 2
st.image("assets/Figure_2.jpg", caption=f"Ground Truth vs Predicted {band} Reconstruction")
