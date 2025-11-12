import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import io
import base64
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="SWELL Synthetic Data Generator",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #2c3e50;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 0.25rem;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Define the ConditionalTimeGAN class (same as in notebook)
class ConditionalTimeGAN(nn.Module):
    """Conditional TimeGAN for time-series synthesis with condition control"""

    def __init__(self, input_dim, hidden_dim, num_layers, num_conditions, seq_len):
        super(ConditionalTimeGAN, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_conditions = num_conditions
        self.seq_len = seq_len

        # Embedder Network (for real data)
        self.embedder = nn.LSTM(input_dim + num_conditions, hidden_dim, num_layers,
                               batch_first=True, dropout=0.3)
        self.embedder_fc = nn.Linear(hidden_dim, hidden_dim)

        # Recovery Network (reconstruct from embedding)
        self.recovery = nn.LSTM(hidden_dim, hidden_dim, num_layers,
                               batch_first=True, dropout=0.3)
        self.recovery_fc = nn.Linear(hidden_dim, input_dim)

        # Generator Network (generate synthetic embeddings)
        self.generator = nn.LSTM(input_dim + num_conditions, hidden_dim, num_layers,
                                batch_first=True, dropout=0.3)
        self.generator_fc = nn.Linear(hidden_dim, hidden_dim)

        # Discriminator Network (distinguish real vs synthetic)
        self.discriminator = nn.LSTM(hidden_dim, hidden_dim, num_layers,
                                    batch_first=True, dropout=0.3)
        self.discriminator_fc = nn.Linear(hidden_dim, 1)

    def embed(self, x, condition):
        """Embed real data to latent space"""
        condition_expanded = condition.unsqueeze(1).expand(-1, self.seq_len, -1)
        x_cond = torch.cat([x, condition_expanded], dim=-1)
        h, _ = self.embedder(x_cond)
        h = torch.sigmoid(self.embedder_fc(h))
        return h

    def recover(self, h):
        """Recover data from latent space"""
        x, _ = self.recovery(h)
        x = self.recovery_fc(x)
        return x

    def generate(self, z, condition):
        """Generate synthetic embeddings from noise"""
        condition_expanded = condition.unsqueeze(1).expand(-1, self.seq_len, -1)
        z_cond = torch.cat([z, condition_expanded], dim=-1)
        h, _ = self.generator(z_cond)
        h = torch.sigmoid(self.generator_fc(h))
        return h

    def discriminate(self, h):
        """Discriminate between real and synthetic embeddings"""
        y, _ = self.discriminator(h)
        y = torch.sigmoid(self.discriminator_fc(y))
        return y

@st.cache_resource
def load_model_and_data():
    """Load the trained model and preprocessing objects"""
    try:
        # Load model
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Model parameters (from notebook)
        input_dim = 35  # HRV features
        hidden_dim = 128
        num_layers = 2
        seq_len = 10
        
        # Load training data to get conditions
        train_df = pd.read_csv('train.csv')
        
        # Setup label encoder
        label_encoder = LabelEncoder()
        train_df['condition_encoded'] = label_encoder.fit_transform(train_df['condition'])
        num_conditions = len(label_encoder.classes_)
        
        # Setup scaler
        feature_columns = train_df.columns[:-2]  # All except 'condition' and 'condition_encoded'
        scaler = StandardScaler()
        scaler.fit(train_df[feature_columns])
        
        # Initialize model
        model = ConditionalTimeGAN(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_conditions=num_conditions,
            seq_len=seq_len
        ).to(device)
        
        # Try to load saved model
        try:
            model.load_state_dict(torch.load('swell_synthetic_model.pth', map_location=device))
            model_loaded = True
        except:
            model_loaded = False
            
        return model, scaler, label_encoder, feature_columns, device, model_loaded
    except Exception as e:
        st.error(f"Error loading model or data: {str(e)}")
        return None, None, None, None, None, False

def generate_synthetic_data(model, num_samples, condition_label, seq_len, n_features, device, num_conditions):
    """Generate synthetic sequences for a specific condition"""
    model.eval()
    with torch.no_grad():
        # Create condition one-hot
        condition_onehot = torch.zeros(num_samples, num_conditions).to(device)
        condition_onehot[:, condition_label] = 1

        # Generate random noise
        z = torch.randn(num_samples, seq_len, n_features).to(device)

        # Generate synthetic embeddings
        h_synthetic = model.generate(z, condition_onehot)

        # Recover to data space
        x_synthetic = model.recover(h_synthetic)

        return x_synthetic.cpu().numpy()

def create_download_link(df, filename):
    """Create a download link for DataFrame"""
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">Download {filename}</a>'
    return href

def main():
    # Header
    st.markdown('<h1 class="main-header">🧠 SWELL Synthetic Data Generator</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666;">Generate synthetic HRV data for mental health stress detection research</p>', unsafe_allow_html=True)
    
    # Load model and data
    model, scaler, label_encoder, feature_columns, device, model_loaded = load_model_and_data()
    
    if model is None:
        st.error("❌ Failed to load model or data. Please ensure all required files are present.")
        return
    
    # Sidebar configuration
    st.sidebar.markdown("## 🎛️ Generation Settings")
    
    # Model status
    if model_loaded:
        st.sidebar.success("✅ Model loaded successfully!")
    else:
        st.sidebar.warning("⚠️ Using untrained model (for demo)")
    
    # Generation parameters
    st.sidebar.markdown("### Data Generation")
    
    # Condition selection
    condition_names = label_encoder.classes_
    selected_condition = st.sidebar.selectbox(
        "Select Stress Condition:",
        options=condition_names,
        help="Choose the stress condition for synthetic data generation"
    )
    condition_idx = label_encoder.transform([selected_condition])[0]
    
    # Number of samples
    num_samples = st.sidebar.slider(
        "Number of Samples:",
        min_value=10,
        max_value=1000,
        value=100,
        step=10,
        help="Number of synthetic sequences to generate"
    )
    
    # Sequence parameters
    st.sidebar.markdown("### Sequence Parameters")
    seq_length = st.sidebar.number_input("Sequence Length (timesteps):", value=10, min_value=5, max_value=20)
    
    # Visualization options
    st.sidebar.markdown("### Visualization")
    show_features = st.sidebar.multiselect(
        "Features to Display:",
        options=list(range(len(feature_columns))),
        default=[0, 1, 2, 10, 15],
        format_func=lambda x: f"Feature {x}: {feature_columns[x][:20]}..."
    )
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col2:
        st.markdown("### 📊 Model Information")
        st.markdown(f"""
        <div class="metric-card">
        <strong>Device:</strong> {device}<br>
        <strong>Features:</strong> {len(feature_columns)}<br>
        <strong>Conditions:</strong> {len(condition_names)}<br>
        <strong>Model Parameters:</strong> ~1M
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("### 🎯 Available Conditions")
        for i, condition in enumerate(condition_names):
            emoji = "🔴" if condition == selected_condition else "⚪"
            st.markdown(f"{emoji} {condition}")
    
    with col1:
        st.markdown("### 🚀 Generate Synthetic Data")
        
        if st.button("🎲 Generate Synthetic Data", type="primary"):
            if not model_loaded:
                st.warning("⚠️ Using untrained model - results may not be meaningful")
            
            with st.spinner("Generating synthetic data..."):
                try:
                    # Generate synthetic data
                    synthetic_sequences = generate_synthetic_data(
                        model, num_samples, condition_idx, seq_length, 
                        len(feature_columns), device, len(condition_names)
                    )
                    
                    # Store in session state
                    st.session_state.synthetic_data = synthetic_sequences
                    st.session_state.condition_name = selected_condition
                    st.session_state.generation_time = datetime.now()
                    
                    st.success(f"✅ Generated {num_samples} synthetic sequences for condition: {selected_condition}")
                    
                except Exception as e:
                    st.error(f"❌ Error generating data: {str(e)}")
    
    # Display results if data exists
    if hasattr(st.session_state, 'synthetic_data'):
        st.markdown("---")
        st.markdown("## 📈 Generated Data Analysis")
        
        synthetic_data = st.session_state.synthetic_data
        condition_name = st.session_state.condition_name
        
        # Basic statistics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Samples Generated", synthetic_data.shape[0])
        with col2:
            st.metric("Sequence Length", synthetic_data.shape[1])
        with col3:
            st.metric("Features", synthetic_data.shape[2])
        with col4:
            st.metric("Condition", condition_name)
        
        # Visualization tabs
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Time Series", "📈 Statistics", "🔍 Feature Analysis", "💾 Export"])
        
        with tab1:
            st.markdown("### Time Series Visualization")
            
            if show_features:
                fig = make_subplots(
                    rows=len(show_features), cols=1,
                    subplot_titles=[f"Feature {i}: {feature_columns[i]}" for i in show_features],
                    vertical_spacing=0.05
                )
                
                colors = px.colors.qualitative.Set3
                
                for idx, feat_idx in enumerate(show_features):
                    # Show first 5 samples
                    for sample_idx in range(min(5, synthetic_data.shape[0])):
                        fig.add_trace(
                            go.Scatter(
                                x=list(range(synthetic_data.shape[1])),
                                y=synthetic_data[sample_idx, :, feat_idx],
                                mode='lines+markers',
                                name=f'Sample {sample_idx+1}' if idx == 0 else None,
                                line=dict(color=colors[sample_idx % len(colors)]),
                                showlegend=(idx == 0),
                                legendgroup=f'sample_{sample_idx}'
                            ),
                            row=idx+1, col=1
                        )
                
                fig.update_layout(height=200*len(show_features), title=f"Synthetic Time Series - {condition_name}")
                fig.update_xaxes(title_text="Time Step")
                fig.update_yaxes(title_text="Normalized Value")
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Please select features to display in the sidebar")
        
        with tab2:
            st.markdown("### Statistical Summary")
            
            # Calculate statistics across all samples and timesteps
            stats_data = []
            for feat_idx in range(synthetic_data.shape[2]):
                feat_data = synthetic_data[:, :, feat_idx].flatten()
                stats_data.append({
                    'Feature': f"{feat_idx}: {feature_columns[feat_idx][:30]}...",
                    'Mean': np.mean(feat_data),
                    'Std': np.std(feat_data),
                    'Min': np.min(feat_data),
                    'Max': np.max(feat_data),
                    'Median': np.median(feat_data)
                })
            
            stats_df = pd.DataFrame(stats_data)
            st.dataframe(stats_df, use_container_width=True)
            
            # Distribution plots for selected features
            if show_features:
                fig, axes = plt.subplots(2, 3, figsize=(15, 8))
                axes = axes.flatten()
                
                for idx, feat_idx in enumerate(show_features[:6]):
                    if idx < len(axes):
                        feat_data = synthetic_data[:, :, feat_idx].flatten()
                        axes[idx].hist(feat_data, bins=30, alpha=0.7, color=f'C{idx}')
                        axes[idx].set_title(f'Feature {feat_idx}')
                        axes[idx].set_xlabel('Value')
                        axes[idx].set_ylabel('Frequency')
                
                # Hide unused subplots
                for idx in range(len(show_features), len(axes)):
                    axes[idx].set_visible(False)
                
                plt.tight_layout()
                st.pyplot(fig)
        
        with tab3:
            st.markdown("### Feature Analysis")
            
            # Feature correlation heatmap
            if len(show_features) > 1:
                # Calculate correlation matrix for selected features
                sample_data = synthetic_data[0, :, show_features]  # Use first sample
                corr_matrix = np.corrcoef(sample_data.T)
                
                fig, ax = plt.subplots(figsize=(10, 8))
                sns.heatmap(corr_matrix, 
                           xticklabels=[f'F{i}' for i in show_features],
                           yticklabels=[f'F{i}' for i in show_features],
                           annot=True, cmap='coolwarm', center=0, ax=ax)
                ax.set_title('Feature Correlation Matrix (Sample 1)')
                st.pyplot(fig)
            
            # Feature importance (variance)
            feature_vars = np.var(synthetic_data.reshape(-1, synthetic_data.shape[2]), axis=0)
            var_df = pd.DataFrame({
                'Feature': [f"{i}: {feature_columns[i][:30]}..." for i in range(len(feature_columns))],
                'Variance': feature_vars
            }).sort_values('Variance', ascending=False)
            
            st.markdown("#### Feature Variance Ranking")
            st.dataframe(var_df.head(10), use_container_width=True)
        
        with tab4:
            st.markdown("### Export Synthetic Data")
            
            # Prepare data for export
            export_format = st.radio("Export Format:", ["Sequences (3D)", "Flattened (2D)"])
            
            if export_format == "Sequences (3D)":
                st.info("3D format preserves sequence structure. Each row represents one timestep of one sequence.")
                
                # Create export dataframe
                export_data = []
                for seq_idx in range(synthetic_data.shape[0]):
                    for time_idx in range(synthetic_data.shape[1]):
                        row = {
                            'sequence_id': seq_idx,
                            'timestep': time_idx,
                            'condition': condition_name
                        }
                        for feat_idx in range(synthetic_data.shape[2]):
                            row[feature_columns[feat_idx]] = synthetic_data[seq_idx, time_idx, feat_idx]
                        export_data.append(row)
                
                export_df = pd.DataFrame(export_data)
                
            else:  # Flattened format
                st.info("2D format flattens sequences. Each row represents one complete sequence.")
                
                # Flatten sequences
                flattened_data = synthetic_data.reshape(synthetic_data.shape[0], -1)
                
                # Create column names
                columns = []
                for time_idx in range(synthetic_data.shape[1]):
                    for feat_idx in range(synthetic_data.shape[2]):
                        columns.append(f"{feature_columns[feat_idx]}_t{time_idx}")
                
                export_df = pd.DataFrame(flattened_data, columns=columns)
                export_df['condition'] = condition_name
            
            # Display preview
            st.markdown("#### Data Preview")
            st.dataframe(export_df.head(), use_container_width=True)
            
            # Download buttons
            col1, col2 = st.columns(2)
            with col1:
                csv = export_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download as CSV",
                    data=csv,
                    file_name=f"synthetic_swell_{condition_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
            
            with col2:
                # Denormalized data option
                if st.button("🔄 Generate Denormalized Data"):
                    try:
                        # Denormalize the data
                        if export_format == "Sequences (3D)":
                            feature_data = export_df[feature_columns].values
                            denorm_data = scaler.inverse_transform(feature_data)
                            denorm_df = export_df.copy()
                            denorm_df[feature_columns] = denorm_data
                        else:
                            # For flattened format, need to reshape and denormalize
                            reshaped_data = synthetic_data.reshape(-1, synthetic_data.shape[2])
                            denorm_data = scaler.inverse_transform(reshaped_data)
                            denorm_sequences = denorm_data.reshape(synthetic_data.shape)
                            
                            # Flatten again
                            flattened_denorm = denorm_sequences.reshape(denorm_sequences.shape[0], -1)
                            denorm_df = pd.DataFrame(flattened_denorm, columns=columns[:-1])  # Exclude condition column
                            denorm_df['condition'] = condition_name
                        
                        st.session_state.denorm_data = denorm_df
                        st.success("✅ Denormalized data generated!")
                        
                    except Exception as e:
                        st.error(f"❌ Error denormalizing data: {str(e)}")
            
            # Download denormalized data if available
            if hasattr(st.session_state, 'denorm_data'):
                denorm_csv = st.session_state.denorm_data.to_csv(index=False)
                st.download_button(
                    label="📥 Download Denormalized CSV",
                    data=denorm_csv,
                    file_name=f"synthetic_swell_{condition_name}_denormalized_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 2rem;">
        <p>🧠 SWELL Synthetic Data Generator | Built with Streamlit & PyTorch</p>
        <p>For mental health stress detection research using HRV data</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
