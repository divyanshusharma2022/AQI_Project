# Air Quality Prediction System
# Complete implementation with data collection, storage, ML model, and web app

import pandas as pd
import numpy as np
import sqlite3
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import requests
import os
from sqlalchemy import create_engine, Column, Integer, Float, String, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import warnings
warnings.filterwarnings('ignore')

# Database Setup
Base = declarative_base()

class AQIData(Base):
    __tablename__ = 'aqi_data'
    
    id = Column(Integer, primary_key=True)
    city = Column(String)
    date = Column(DateTime)
    pm25 = Column(Float)
    pm10 = Column(Float)
    no2 = Column(Float)
    so2 = Column(Float)
    co = Column(Float)
    o3 = Column(Float)
    aqi = Column(Float)

# Data Generation and Collection Module
class DataCollector:
    @staticmethod
    def generate_synthetic_data(n_samples=5000):
        """Generate synthetic AQI data for demonstration"""
        np.random.seed(42)
        
        cities = ['Delhi', 'Mumbai', 'Bangalore', 'Chennai', 'Kolkata', 
                 'Hyderabad', 'Pune', 'Ahmedabad', 'Jaipur', 'Lucknow']
        
        data = []
        base_date = datetime.now() - timedelta(days=365)
        
        for i in range(n_samples):
            city = np.random.choice(cities)
            date = base_date + timedelta(days=np.random.randint(0, 365))
            
            # Generate correlated pollutant values
            base_pollution = np.random.uniform(30, 200)
            seasonal_factor = 1 + 0.3 * np.sin(2 * np.pi * date.timetuple().tm_yday / 365)
            
            pm25 = base_pollution * seasonal_factor * np.random.uniform(0.8, 1.2)
            pm10 = pm25 * np.random.uniform(1.2, 1.8)
            no2 = base_pollution * 0.5 * np.random.uniform(0.7, 1.3)
            so2 = base_pollution * 0.3 * np.random.uniform(0.6, 1.4)
            co = base_pollution * 0.4 * np.random.uniform(0.5, 1.5)
            o3 = base_pollution * 0.6 * np.random.uniform(0.4, 1.2)
            
            # Calculate AQI (simplified formula)
            aqi = max(pm25, pm10/2, no2*0.8, so2*0.7, co*0.5, o3*0.6)
            
            data.append({
                'city': city,
                'date': date,
                'pm25': round(pm25, 2),
                'pm10': round(pm10, 2),
                'no2': round(no2, 2),
                'so2': round(so2, 2),
                'co': round(co, 2),
                'o3': round(o3, 2),
                'aqi': round(aqi, 2)
            })
        
        return pd.DataFrame(data)

# Database Manager
class DatabaseManager:
    def __init__(self, db_path='aqi_database.db'):
        self.db_path = db_path
        self.engine = create_engine(f'sqlite:///{db_path}')
        Base.metadata.create_all(self.engine)
        Session = sessionmaker(bind=self.engine)
        self.session = Session()
    
    def store_data(self, df):
        """Store DataFrame in database"""
        df.to_sql('aqi_data', self.engine, if_exists='replace', index=False)
        return True
    
    def get_data(self, query=None):
        """Retrieve data from database"""
        if query:
            return pd.read_sql_query(query, self.engine)
        return pd.read_sql_query("SELECT * FROM aqi_data", self.engine)
    
    def get_cities(self):
        """Get list of unique cities"""
        query = "SELECT DISTINCT city FROM aqi_data"
        result = pd.read_sql_query(query, self.engine)
        return result['city'].tolist()

# Machine Learning Model
class AQIPredictor:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.feature_columns = ['pm25', 'pm10', 'no2', 'so2', 'co', 'o3']
        
    def prepare_features(self, df):
        """Prepare features for training"""
        # Add temporal features
        df['month'] = pd.to_datetime(df['date']).dt.month
        df['day_of_year'] = pd.to_datetime(df['date']).dt.dayofyear
        df['weekday'] = pd.to_datetime(df['date']).dt.weekday
        
        # Add city encoding
        df['city_encoded'] = pd.Categorical(df['city']).codes
        
        feature_cols = self.feature_columns + ['month', 'day_of_year', 'weekday', 'city_encoded']
        return df[feature_cols]
    
    def train(self, df, model_type='random_forest'):
        """Train the ML model"""
        # Prepare features
        X = self.prepare_features(df)
        y = df['aqi']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model
        if model_type == 'random_forest':
            self.model = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=42
            )
        else:
            self.model = LinearRegression()
        
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate
        y_pred = self.model.predict(X_test_scaled)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        return {
            'rmse': rmse,
            'r2': r2,
            'y_test': y_test,
            'y_pred': y_pred
        }
    
    def predict(self, city, pm25, pm10, no2, so2, co, o3, date=None):
        """Make AQI prediction"""
        if self.model is None:
            return None
        
        if date is None:
            date = datetime.now()
        
        # Create input DataFrame
        input_data = pd.DataFrame([{
            'city': city,
            'date': date,
            'pm25': pm25,
            'pm10': pm10,
            'no2': no2,
            'so2': so2,
            'co': co,
            'o3': o3
        }])
        
        # Prepare features
        X = self.prepare_features(input_data)
        X_scaled = self.scaler.transform(X)
        
        # Predict
        prediction = self.model.predict(X_scaled)[0]
        return round(prediction, 2)
    
    def save_model(self, path='aqi_model.pkl'):
        """Save trained model"""
        joblib.dump({'model': self.model, 'scaler': self.scaler}, path)
    
    def load_model(self, path='aqi_model.pkl'):
        """Load trained model"""
        if os.path.exists(path):
            saved = joblib.load(path)
            self.model = saved['model']
            self.scaler = saved['scaler']
            return True
        return False

# Streamlit Web Application
def main():
    st.set_page_config(
        page_title="Air Quality Prediction System",
        page_icon="🌍",
        layout="wide"
    )
    
    st.title("🌍 Air Quality Prediction System")
    st.markdown("### Predict and Monitor Air Quality Index (AQI)")
    
    # Initialize components
    if 'db_manager' not in st.session_state:
        st.session_state.db_manager = DatabaseManager()
        st.session_state.predictor = AQIPredictor()
        st.session_state.data_loaded = False
    
    # Sidebar
    st.sidebar.header("System Controls")
    
    # Data Management Section
    if st.sidebar.button("🔄 Generate & Load Data"):
        with st.spinner("Generating synthetic data..."):
            collector = DataCollector()
            df = collector.generate_synthetic_data()
            st.session_state.db_manager.store_data(df)
            st.session_state.data = df
            st.session_state.data_loaded = True
            st.success("✅ Data generated and stored in database!")
    
    # Model Training Section
    if st.session_state.data_loaded:
        st.sidebar.markdown("---")
        st.sidebar.subheader("Model Training")
        model_type = st.sidebar.selectbox(
            "Select Model",
            ["random_forest", "linear_regression"]
        )
        
        if st.sidebar.button("🎯 Train Model"):
            with st.spinner("Training model..."):
                df = st.session_state.db_manager.get_data()
                results = st.session_state.predictor.train(df, model_type)
                st.session_state.training_results = results
                st.sidebar.success(f"✅ Model trained!")
                st.sidebar.metric("RMSE", f"{results['rmse']:.2f}")
                st.sidebar.metric("R² Score", f"{results['r2']:.3f}")
    
    # Main Content Area
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Prediction", "📈 Data Analysis", "🗂️ Database View", "🎯 Model Performance"])
    
    with tab1:
        st.header("AQI Prediction")
        
        if st.session_state.data_loaded and st.session_state.predictor.model:
            col1, col2 = st.columns(2)
            
            with col1:
                cities = st.session_state.db_manager.get_cities()
                city = st.selectbox("Select City", cities)
                
                st.subheader("Enter Pollutant Values")
                pm25 = st.slider("PM2.5 (μg/m³)", 0.0, 500.0, 50.0)
                pm10 = st.slider("PM10 (μg/m³)", 0.0, 600.0, 100.0)
                no2 = st.slider("NO2 (μg/m³)", 0.0, 400.0, 40.0)
            
            with col2:
                st.write("")
                st.write("")
                st.write("")
                so2 = st.slider("SO2 (μg/m³)", 0.0, 500.0, 20.0)
                co = st.slider("CO (mg/m³)", 0.0, 50.0, 2.0)
                o3 = st.slider("O3 (μg/m³)", 0.0, 600.0, 100.0)
            
            if st.button("🔮 Predict AQI", type="primary"):
                prediction = st.session_state.predictor.predict(
                    city, pm25, pm10, no2, so2, co, o3
                )
                
                # Display prediction with color coding
                col1, col2, col3 = st.columns([1, 2, 1])
                with col2:
                    if prediction <= 50:
                        color = "green"
                        category = "Good"
                    elif prediction <= 100:
                        color = "yellow"
                        category = "Moderate"
                    elif prediction <= 150:
                        color = "orange"
                        category = "Unhealthy for Sensitive Groups"
                    elif prediction <= 200:
                        color = "red"
                        category = "Unhealthy"
                    elif prediction <= 300:
                        color = "purple"
                        category = "Very Unhealthy"
                    else:
                        color = "maroon"
                        category = "Hazardous"
                    
                    st.markdown(f"""
                    <div style="text-align: center; padding: 20px; border-radius: 10px; background-color: {color}; color: white;">
                        <h1>Predicted AQI: {prediction}</h1>
                        <h3>{category}</h3>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.info("⚠️ Please generate data and train a model first using the sidebar controls.")
    
    with tab2:
        st.header("Data Analysis")
        
        if st.session_state.data_loaded:
            df = st.session_state.db_manager.get_data()
            
            # AQI Trends
            st.subheader("AQI Trends by City")
            fig = px.line(df, x='date', y='aqi', color='city',
                         title='AQI Trends Over Time')
            st.plotly_chart(fig, use_container_width=True)
            
            # Pollutant Distribution
            st.subheader("Pollutant Distribution")
            pollutant = st.selectbox("Select Pollutant", 
                                    ['pm25', 'pm10', 'no2', 'so2', 'co', 'o3'])
            fig2 = px.box(df, x='city', y=pollutant,
                         title=f'{pollutant.upper()} Distribution by City')
            st.plotly_chart(fig2, use_container_width=True)
            
            # Correlation Heatmap
            st.subheader("Pollutant Correlation Matrix")
            corr_cols = ['pm25', 'pm10', 'no2', 'so2', 'co', 'o3', 'aqi']
            corr_matrix = df[corr_cols].corr()
            fig3 = px.imshow(corr_matrix, text_auto=True,
                           title='Correlation Between Pollutants and AQI')
            st.plotly_chart(fig3, use_container_width=True)
        else:
            st.info("⚠️ No data available. Please generate data first.")
    
    with tab3:
        st.header("Database View")
        
        if st.session_state.data_loaded:
            df = st.session_state.db_manager.get_data()
            
            # Filters
            col1, col2, col3 = st.columns(3)
            with col1:
                city_filter = st.multiselect("Filter by City", df['city'].unique())
            with col2:
                aqi_min = st.number_input("Min AQI", value=0)
            with col3:
                aqi_max = st.number_input("Max AQI", value=500)
            
            # Apply filters
            filtered_df = df.copy()
            if city_filter:
                filtered_df = filtered_df[filtered_df['city'].isin(city_filter)]
            filtered_df = filtered_df[(filtered_df['aqi'] >= aqi_min) & 
                                     (filtered_df['aqi'] <= aqi_max)]
            
            # Display statistics
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total Records", len(filtered_df))
            col2.metric("Avg AQI", f"{filtered_df['aqi'].mean():.2f}")
            col3.metric("Max AQI", f"{filtered_df['aqi'].max():.2f}")
            col4.metric("Min AQI", f"{filtered_df['aqi'].min():.2f}")
            
            # Display data table
            st.dataframe(filtered_df.sort_values('date', ascending=False).head(100))
            
            # Download button
            csv = filtered_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Data as CSV",
                data=csv,
                file_name='aqi_data.csv',
                mime='text/csv'
            )
        else:
            st.info("⚠️ No data available. Please generate data first.")
    
    with tab4:
        st.header("Model Performance")
        
        if hasattr(st.session_state, 'training_results'):
            results = st.session_state.training_results
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Performance Metrics")
                st.metric("Root Mean Square Error (RMSE)", f"{results['rmse']:.2f}")
                st.metric("R² Score", f"{results['r2']:.3f}")
                
                # Feature importance (for Random Forest)
                if hasattr(st.session_state.predictor.model, 'feature_importances_'):
                    st.subheader("Feature Importance")
                    features = ['pm25', 'pm10', 'no2', 'so2', 'co', 'o3', 
                               'month', 'day_of_year', 'weekday', 'city_encoded']
                    importance_df = pd.DataFrame({
                        'feature': features,
                        'importance': st.session_state.predictor.model.feature_importances_
                    }).sort_values('importance', ascending=False)
                    
                    fig = px.bar(importance_df, x='importance', y='feature',
                               orientation='h', title='Feature Importance')
                    st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.subheader("Actual vs Predicted")
                
                # Scatter plot
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=results['y_test'],
                    y=results['y_pred'],
                    mode='markers',
                    name='Predictions',
                    marker=dict(color='blue', size=5, opacity=0.5)
                ))
                fig.add_trace(go.Scatter(
                    x=[results['y_test'].min(), results['y_test'].max()],
                    y=[results['y_test'].min(), results['y_test'].max()],
                    mode='lines',
                    name='Perfect Prediction',
                    line=dict(color='red', dash='dash')
                ))
                fig.update_layout(
                    title='Actual vs Predicted AQI',
                    xaxis_title='Actual AQI',
                    yaxis_title='Predicted AQI',
                    showlegend=True
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Residual plot
                residuals = results['y_test'] - results['y_pred']
                fig2 = px.histogram(residuals, nbins=30,
                                   title='Prediction Residuals Distribution')
                fig2.update_xaxes(title='Residual (Actual - Predicted)')
                fig2.update_yaxes(title='Frequency')
                st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("⚠️ No model trained yet. Please train a model first.")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center'>
        <p>🌍 Air Quality Prediction System | Built with Streamlit, SQLAlchemy & Scikit-learn</p>
        <p>Ready for deployment on Streamlit Cloud or Render</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()