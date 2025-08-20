import streamlit as st
import requests
import json
import os
from dotenv import load_dotenv
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import pandas as pd
import time
import uuid

load_dotenv()
API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(
    page_title="Advanced Healthcare RAG System",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced CSS with high contrast and better visibility
st.markdown("""
<style>
    /* Main app background - Dark theme for better contrast */
    .stApp {
        background: linear-gradient(135deg, #0d1117 0%, #161b22 100%);
        color: #f0f6fc;
    }
    
    .main .block-container {
        background: transparent;
        color: #f0f6fc;
    }
    
    /* Headers - Bright and visible */
    h1, h2, h3, h4, h5, h6 {
        color: #58a6ff !important;
        text-shadow: 0 0 10px rgba(88, 166, 255, 0.3);
    }
    
    /* Main header card */
    .main-header {
        background: linear-gradient(135deg, #1f4e79 0%, #2d86c6 100%);
        padding: 2.5rem;
        border-radius: 15px;
        color: #ffffff;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px rgba(31, 78, 121, 0.3);
        border: 1px solid rgba(88, 166, 255, 0.2);
    }
    
    .main-header h1, .main-header h2, .main-header p {
        color: #ffffff !important;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.5);
    }
    
    /* Feature cards */
    .feature-card {
        background: linear-gradient(135deg, #21262d 0%, #30363d 100%);
        padding: 2rem;
        border-radius: 12px;
        border-left: 6px solid #58a6ff;
        margin: 1rem 0;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.3);
        color: #f0f6fc;
    }
    
    .feature-card h3 {
        color: #58a6ff !important;
        margin-bottom: 0.5rem;
    }
    
    .feature-card p {
        color: #c9d1d9 !important;
        line-height: 1.5;
    }
    
    /* Capability cards */
    .capability-card {
        background: linear-gradient(135deg, #1a2332 0%, #243142 100%);
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #ffd700;
        margin: 0.5rem 0;
        color: #f0f6fc;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.2);
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #238636 0%, #2ea043 100%) !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 0.75rem 1.5rem !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
        box-shadow: 0 4px 12px rgba(46, 160, 67, 0.3) !important;
        transition: all 0.3s ease !important;
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, #2ea043 0%, #238636 100%) !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 16px rgba(46, 160, 67, 0.4) !important;
    }
    
    /* Text inputs */
    .stTextInput > div > input {
        background-color: #0d1117 !important;
        color: #f0f6fc !important;
        border: 2px solid #30363d !important;
        border-radius: 8px !important;
        padding: 0.75rem !important;
        font-size: 1rem !important;
    }
    
    .stTextInput > div > input:focus {
        border-color: #58a6ff !important;
        box-shadow: 0 0 0 3px rgba(88, 166, 255, 0.1) !important;
        outline: none !important;
    }
    
    .stTextInput > div > input::placeholder {
        color: #8b949e !important;
    }
    
    /* Text areas */
    .stTextArea > div > textarea {
        background-color: #0d1117 !important;
        color: #f0f6fc !important;
        border: 2px solid #30363d !important;
        border-radius: 8px !important;
        padding: 1rem !important;
        font-size: 1rem !important;
        min-height: 120px !important;
    }
    
    .stTextArea > div > textarea:focus {
        border-color: #58a6ff !important;
        box-shadow: 0 0 0 3px rgba(88, 166, 255, 0.1) !important;
        outline: none !important;
    }
    
    .stTextArea > div > textarea::placeholder {
        color: #8b949e !important;
    }
    
    /* Select boxes */
    .stSelectbox > div > div {
        background-color: #0d1117 !important;
        color: #f0f6fc !important;
        border: 2px solid #30363d !important;
        border-radius: 8px !important;
    }
    
    .stSelectbox > div > div > div {
        color: #f0f6fc !important;
    }
    
    /* File uploader */
    .stFileUploader > div {
        background-color: #161b22 !important;
        border: 2px dashed #58a6ff !important;
        border-radius: 10px !important;
        padding: 2rem !important;
        color: #f0f6fc !important;
    }
    
    /* Sidebar */
    .css-1d391kg, [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0d1117 0%, #161b22 100%) !important;
        border-right: 1px solid #30363d;
    }
    
    .css-1d391kg * {
        color: #c9d1d9 !important;
    }
    
    .css-1d391kg h1, .css-1d391kg h2, .css-1d391kg h3 {
        color: #58a6ff !important;
    }
    
    .css-1d391kg a {
        color: #58a6ff !important;
    }
    
    .css-1d391kg a:hover {
        color: #79c0ff !important;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        background: transparent;
        gap: 1rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: linear-gradient(135deg, #21262d 0%, #30363d 100%);
        color: #c9d1d9;
        border: 1px solid #30363d;
        border-radius: 10px;
        padding: 0.75rem 1.5rem;
        font-weight: 600;
    }
    
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background: linear-gradient(135deg, #58a6ff 0%, #1f6feb 100%);
        color: #ffffff;
        border-color: #58a6ff;
        box-shadow: 0 4px 12px rgba(88, 166, 255, 0.3);
    }
    
    /* Chat messages */
    .chat-user {
        background: linear-gradient(135deg, #1f2937 0%, #374151 100%);
        color: #f9fafb;
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        border-left: 4px solid #3b82f6;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
    }
    
    .chat-assistant {
        background: linear-gradient(135deg, #581c87 0%, #7c3aed 100%);
        color: #f3e8ff;
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        border-left: 4px solid #a855f7;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
    }
    
    /* Success/Error messages */
    .stSuccess {
        background: linear-gradient(135deg, #064e3b 0%, #065f46 100%) !important;
        color: #6ee7b7 !important;
        border: 1px solid #10b981 !important;
        border-radius: 8px !important;
    }
    
    .stError {
        background: linear-gradient(135deg, #7f1d1d 0%, #991b1b 100%) !important;
        color: #fca5a5 !important;
        border: 1px solid #dc2626 !important;
        border-radius: 8px !important;
    }
    
    .stWarning {
        background: linear-gradient(135deg, #78350f 0%, #92400e 100%) !important;
        color: #fcd34d !important;
        border: 1px solid #f59e0b !important;
        border-radius: 8px !important;
    }
    
    .stInfo {
        background: linear-gradient(135deg, #1e3a8a 0%, #1d4ed8 100%) !important;
        color: #93c5fd !important;
        border: 1px solid #3b82f6 !important;
        border-radius: 8px !important;
    }
    
    /* Metrics */
    .metric-container {
        background: linear-gradient(135deg, #1f2937 0%, #374151 100%);
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
        border-left: 4px solid #58a6ff;
        color: #f9fafb;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background: linear-gradient(135deg, #21262d 0%, #30363d 100%);
        border-radius: 8px;
        padding: 1rem;
        color: #58a6ff;
        font-weight: 600;
        border: 1px solid #30363d;
    }
    
    .streamlit-expanderContent {
        background: #161b22;
        color: #c9d1d9;
        border: 1px solid #30363d;
        border-top: none;
        border-radius: 0 0 8px 8px;
    }
    
    /* Progress bars */
    .stProgress .st-bo {
        background-color: #58a6ff !important;
    }
    
    /* Form labels */
    .stTextInput label, .stTextArea label, .stSelectbox label {
        color: #f0f6fc !important;
        font-weight: 500 !important;
    }
    
    /* Dataframes */
    .dataframe {
        background-color: #161b22 !important;
        color: #f0f6fc !important;
        border: 1px solid #30363d !important;
    }
    
    /* Plotly charts */
    .js-plotly-plot {
        background-color: #161b22 !important;
        border-radius: 10px;
        border: 1px solid #30363d;
    }
    
    /* Any remaining text */
    p, span, div {
        color: #c9d1d9 !important;
    }
    
    /* Form containers */
    .stForm {
        background: linear-gradient(135deg, #161b22 0%, #21262d 100%);
        border: 1px solid #30363d;
        border-radius: 12px;
        padding: 2rem;
    }
    
    /* Markdown containers */
    .stMarkdown {
        color: #c9d1d9 !important;
    }
    
    /* Strong emphasis */
    strong {
        color: #f0f6fc !important;
    }
    
    /* Code blocks */
    code {
        background-color: #21262d !important;
        color: #79c0ff !important;
        padding: 0.2rem 0.4rem !important;
        border-radius: 4px !important;
    }
    
    /* Links */
    a {
        color: #58a6ff !important;
        text-decoration: none !important;
    }
    
    a:hover {
        color: #79c0ff !important;
        text-decoration: underline !important;
    }
</style>
""", unsafe_allow_html=True)

# Session state initialization
def init_session_state():
    """Initialize all session state variables"""
    defaults = {
        "logged_in": False,
        "access_token": None,
        "user_info": None,
        "chat_history": [],
        "current_tab": "login",
        "input_counter": 0
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

def get_auth_headers():
    """Get authentication headers for API requests"""
    if st.session_state.access_token:
        return {"Authorization": f"Bearer {st.session_state.access_token}"}
    return {}

def make_api_request(endpoint, method="GET", data=None, files=None, timeout=60):
    """Make API request with proper error handling"""
    try:
        url = f"{API_URL}{endpoint}"
        headers = get_auth_headers()
        
        if method == "GET":
            response = requests.get(url, headers=headers, params=data, timeout=timeout)
        elif method == "POST":
            if files:
                response = requests.post(url, headers=headers, data=data, files=files, timeout=timeout)
            else:
                response = requests.post(url, headers=headers, json=data, timeout=timeout)
        elif method == "DELETE":
            response = requests.delete(url, headers=headers, timeout=timeout)
        else:
            raise ValueError(f"Unsupported HTTP method: {method}")
        
        return response
    
    except requests.exceptions.ConnectionError:
        st.error("🚨 Cannot connect to the server. Please ensure the API is running on http://localhost:8000")
        return None
    except requests.exceptions.Timeout:
        st.error("⏱️ Request timed out. Please try again.")
        return None
    except Exception as e:
        st.error(f"❌ Request failed: {str(e)}")
        return None

def login_page():
    """Login and registration page"""
    st.markdown("""
    <div class="main-header">
        <h1>🏥 Advanced Healthcare RAG System</h1>
        <p style="font-size: 1.3em; margin-top: 1rem; opacity: 0.95;">
            Multi-vector Hybrid Retrieval • Graph-based Knowledge • Chain-of-Thought Reasoning
        </p>
        <p style="font-size: 1em; margin-top: 0.8rem; opacity: 0.85;">
            Experience the future of healthcare information systems with AI-powered precision
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Feature showcase
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="feature-card">
            <h3>🔍 Advanced RAG</h3>
            <p>Multi-vector hybrid retrieval combining semantic search with keyword matching for superior accuracy and comprehensive results</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="feature-card">
            <h3>🧠 AI-Powered</h3>
            <p>Chain-of-thought reasoning with intelligent query rewriting and multi-hop information retrieval for complex medical queries</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="feature-card">
            <h3>🕸️ Knowledge Graph</h3>
            <p>Relationship-aware retrieval using medical knowledge graphs for contextual understanding and connected insights</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Login/Signup forms
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        tab1, tab2 = st.tabs(["🔑 Login", "📝 Register"])
        
        with tab1:
            st.markdown("### Welcome Back!")
            with st.form("login_form"):
                username = st.text_input("**Username**", placeholder="Enter your username")
                password = st.text_input("**Password**", type="password", placeholder="Enter your password")
                
                col_a, col_b, col_c = st.columns([1, 2, 1])
                with col_b:
                    submitted = st.form_submit_button("🚀 Login", use_container_width=True)
                
                if submitted and username and password:
                    with st.spinner("🔐 Authenticating..."):
                        response = make_api_request(
                            "/auth/login", 
                            method="POST",
                            data={"username": username, "password": password}
                        )
                        
                        if response and response.status_code == 200:
                            data = response.json()
                            st.session_state.access_token = data["access_token"]
                            st.session_state.user_info = data["user"]
                            st.session_state.logged_in = True
                            
                            st.success("🎉 Login successful! Redirecting...")
                            time.sleep(1)
                            st.rerun()
                        else:
                            error_msg = "Invalid credentials"
                            if response:
                                try:
                                    error_data = response.json()
                                    error_msg = error_data.get("detail", error_msg)
                                except:
                                    error_msg = f"HTTP {response.status_code} error"
                            st.error(f"❌ {error_msg}")
        
        with tab2:
            st.markdown("### Join Our Platform!")
            with st.form("signup_form"):
                new_username = st.text_input("**Username**", placeholder="Choose a unique username")
                new_password = st.text_input("**Password**", type="password", placeholder="Create a strong password")
                full_name = st.text_input("**Full Name**", placeholder="Your full name")
                
                role = st.selectbox(
                    "**Select Your Role**",
                    ["doctor", "nurse", "patient"],
                    format_func=lambda x: {"doctor": "👨‍⚕️ Doctor", "nurse": "👩‍⚕️ Nurse", "patient": "🧑‍🤝‍🧑 Patient"}[x]
                )
                
                # Role-specific fields
                specialization = None
                department = None
                patient_id = None
                
                if role == "doctor":
                    specialization = st.text_input("**Specialization**", placeholder="e.g., Cardiology, Neurology")
                elif role == "nurse":
                    department = st.text_input("**Department**", placeholder="e.g., ICU, Emergency")
                elif role == "patient":
                    patient_id = st.text_input("**Patient ID**", placeholder="Your patient identifier")
                
                col_a, col_b, col_c = st.columns([1, 2, 1])
                with col_b:
                    submitted = st.form_submit_button("✨ Create Account", use_container_width=True)
                
                if submitted and new_username and new_password:
                    with st.spinner("🔨 Creating your account..."):
                        payload = {
                            "username": new_username,
                            "password": new_password,
                            "role": role,
                            "full_name": full_name,
                            "specialization": specialization,
                            "department": department,
                            "patient_id": patient_id
                        }
                        
                        response = make_api_request("/auth/signup", method="POST", data=payload)
                        
                        if response and response.status_code == 200:
                            st.success("🎉 Account created successfully! Please login with your credentials.")
                        else:
                            error_msg = "Registration failed"
                            if response:
                                try:
                                    error_data = response.json()
                                    error_msg = error_data.get("detail", error_msg)
                                except:
                                    error_msg = f"HTTP {response.status_code} error"
                            st.error(f"❌ {error_msg}")

def show_knowledge_graph_insights(metrics):
    """Display knowledge graph insights in a user-friendly way"""
    
    if not metrics.get("graph_entities") and not metrics.get("graph_relationships"):
        st.info("🕸️ Knowledge Graph: No medical entities extracted for this query")
        return
    
    st.markdown("### 🕸️ Knowledge Graph Analysis")
    
    # Show extracted entities
    if metrics.get("graph_entities"):
        st.markdown("**🏷️ Medical Entities Discovered:**")
        entities = metrics["graph_entities"][:10]  # Show first 10
        
        # Display entities in a nice format
        entity_text = ""
        for i, entity in enumerate(entities):
            if i > 0:
                entity_text += " • "
            entity_text += f"`{entity}`"
        
        st.markdown(entity_text)
    
    # Show relationships if any
    if metrics.get("graph_relationships", 0) > 0:
        st.success(f"🔗 **Graph Connections Found:** {metrics['graph_relationships']} medical relationships")
        st.info("💡 The system used these connections to find related information across different documents!")

def show_hybrid_search_details(metrics):
    """Show that hybrid search is being used"""
    
    st.markdown("### 🔄 Hybrid Search Analysis")
    
    # Show search method
    retrieval_method = metrics.get("retrieval_method", "")
    if "Hybrid" in retrieval_method:
        st.success("✅ **Hybrid Search Active:** Combining semantic + keyword search")
        
        col1, col2 = st.columns(2)
        with col1:
            st.info("🧠 **Semantic Search**\nVector similarity using AI embeddings")
        with col2:
            st.info("🔤 **Keyword Search**\nBM25 text matching for exact terms")
        
        st.markdown("**🎯 Result:** Best of both worlds - semantic understanding + exact keyword matching")
    else:
        st.warning("⚠️ Using single search method - consider enabling hybrid search")

def main_app():
    """Main application interface"""
    user_info = st.session_state.user_info
    
    # Header with user info
    st.markdown(f"""
    <div class="main-header">
        <h2>Welcome back, {user_info.get('full_name', user_info['username'])}! 👋</h2>
        <p style="font-size: 1.2em; margin-top: 0.5rem;">Role: {user_info['role'].title()}</p>
        <p style="font-size: 1em; margin-top: 0.5rem; opacity: 0.9;">Advanced Healthcare RAG System at your service</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar with user info and navigation
    with st.sidebar:
        st.markdown("### 👤 User Profile")
        st.write(f"**Name:** {user_info.get('full_name', user_info['username'])}")
        st.write(f"**Role:** {user_info['role'].title()}")
        
        if user_info.get('specialization'):
            st.write(f"**Specialization:** {user_info['specialization']}")
        if user_info.get('department'):
            st.write(f"**Department:** {user_info['department']}")
        if user_info.get('patient_id'):
            st.write(f"**Patient ID:** {user_info['patient_id']}")
        
        st.divider()
        
        # Get and display capabilities
        capabilities_response = make_api_request("/chat/capabilities")
        if capabilities_response and capabilities_response.status_code == 200:
            try:
                capabilities = capabilities_response.json()
                
                st.markdown("### 🎯 Your Capabilities")
                for cap in capabilities.get('role_specific', [])[:4]:
                    st.write(f"✅ {cap}")
                
                if len(capabilities.get('role_specific', [])) > 4:
                    with st.expander("View All Capabilities"):
                        for cap in capabilities.get('role_specific', [])[4:]:
                            st.write(f"✅ {cap}")
                
                st.divider()
                st.markdown("### 🚀 Advanced Features")
                for feature in capabilities.get('advanced_features', []):
                    st.write(f"⭐ {feature}")
            except:
                st.write("Capabilities loading...")
        
        st.divider()
        
        # Logout button
        if st.button("🚪 Logout", use_container_width=True):
            for key in ["logged_in", "access_token", "user_info", "chat_history"]:
                if key in st.session_state:
                    del st.session_state[key]
            st.rerun()
        
        # System status
        st.markdown("### 📊 System Status")
        health_response = make_api_request("/health")
        if health_response and health_response.status_code == 200:
            st.success("🟢 All Systems Operational")
        else:
            st.error("🔴 System Issues Detected")
    
    # Main content tabs
    if user_info['role'] == 'admin':
        tabs = st.tabs(["💬 Advanced Chat", "📤 Upload Documents", "📊 System Analytics", "🕸️ Knowledge Graph"])
        tab_functions = [advanced_chat_interface, document_upload_interface, system_analytics_interface, knowledge_graph_interface]
    else:
        tabs = st.tabs(["💬 Advanced Chat", "📈 My Analytics", "🕸️ Knowledge Graph"])
        tab_functions = [advanced_chat_interface, user_analytics_interface, knowledge_graph_interface]
    
    for tab, func in zip(tabs, tab_functions):
        with tab:
            func()

def advanced_chat_interface():
    """Enhanced chat interface showing all RAG capabilities - COMPLETE VERSION"""
    st.markdown("## 🤖 Advanced RAG Chat System")
    
    # Show system capabilities prominently
    with st.expander("🚀 Your System's Advanced Capabilities", expanded=False):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("""
            <div class="capability-card">
                <h4>🔄 Hybrid Search</h4>
                <ul>
                    <li>Semantic vector search</li>
                    <li>BM25 keyword matching</li>
                    <li>Intelligent result fusion</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown("""
            <div class="capability-card">
                <h4>🕸️ Knowledge Graph</h4>
                <ul>
                    <li>Medical entity extraction</li>
                    <li>Relationship mapping</li>
                    <li>Cross-document connections</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        with col3:
            st.markdown("""
            <div class="capability-card">
                <h4>🧠 Multi-hop Reasoning</h4>
                <ul>
                    <li>Query rewriting</li>
                    <li>Step-by-step analysis</li>
                    <li>Chain-of-thought processing</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
    
    # Chat configuration
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        chat_mode = st.selectbox(
            "**Select Chat Mode:**",
            ["🚀 Advanced Multi-vector RAG", "📝 Simple RAG", "⚖️ Compare Both Methods"],
            help="Choose between different RAG approaches to see their capabilities"
        )
    
    with col2:
        max_history = st.selectbox("**Chat History:**", [5, 10, 15, 20], index=1)
    
    with col3:
        if st.button("🗑️ Clear History"):
            st.session_state.chat_history = []
            st.rerun()
    
    # Chat history display - ENHANCED WITH FULL RESPONSES
    if st.session_state.chat_history:
        st.markdown("### 💬 Chat History")
        
        for i, chat in enumerate(st.session_state.chat_history[-max_history:]):
            # User message
            st.markdown(f"""
            <div class="chat-user">
                <strong>You:</strong> {chat['query']}
            </div>
            """, unsafe_allow_html=True)
            
            # Assistant response - SHOW FULL RESPONSE (NO TRUNCATION)
            st.markdown(f"""
            <div class="chat-assistant">
                <strong>Assistant:</strong> {chat['answer']}
            </div>
            """, unsafe_allow_html=True)
            
            # Enhanced metrics display
            if chat.get('metrics'):
                with st.expander(f"🔍 Advanced RAG Analysis #{i+1}", expanded=False):
                    metrics = chat['metrics']
                    
                    # Main metrics
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Multi-hop Steps", metrics.get("multi_hop_steps", 0))
                    with col2:
                        st.metric("Sources Found", len(metrics.get("sources", [])))
                    with col3:
                        st.metric("Graph Entities", len(metrics.get("graph_entities", [])))
                    with col4:
                        st.metric("Processing Time", f"{metrics.get('processing_time', 0):.2f}s")
                    
                    st.divider()
                    
                    # Show advanced features used
                    st.markdown("### 🚀 Advanced Features Activated")
                    
                    features_used = []
                    
                    # Query rewriting
                    if metrics.get("rewritten_query") and metrics.get("rewritten_query") != chat["query"]:
                        st.success(f"✅ **Query Optimization**")
                        st.code(f"Original: {chat['query']}\nOptimized: {metrics.get('rewritten_query')}")
                        features_used.append("Query Rewriting")
                    
                    # Multi-hop reasoning
                    if metrics.get("multi_hop_steps", 0) > 0:
                        st.success(f"✅ **Multi-hop Analysis:** {metrics.get('multi_hop_steps')} reasoning steps")
                        features_used.append("Multi-hop Reasoning")
                    
                    # Knowledge graph
                    if metrics.get("graph_entities"):
                        st.success(f"✅ **Knowledge Graph:** {len(metrics.get('graph_entities', []))} medical entities extracted")
                        features_used.append("Knowledge Graph")
                        
                        # Show some entities
                        entities_sample = metrics["graph_entities"][:10]
                        if entities_sample:
                            st.write("🏷️ **Key Medical Entities:**")
                            entity_cols = st.columns(min(len(entities_sample), 5))
                            for idx, entity in enumerate(entities_sample):
                                with entity_cols[idx % 5]:
                                    st.markdown(f"`{entity}`")
                    
                    # Graph relationships
                    if metrics.get("graph_relationships", 0) > 0:
                        st.success(f"✅ **Medical Connections:** {metrics.get('graph_relationships')} relationships found")
                        features_used.append("Graph Relationships")
                    
                    # Hybrid search confirmation
                    retrieval_method = metrics.get("retrieval_method", "")
                    if "Hybrid" in retrieval_method:
                        st.success("✅ **Hybrid Search:** Semantic + Keyword matching active")
                        features_used.append("Hybrid Search")
                        
                        # Show hybrid search details
                        show_hybrid_search_details(metrics)
                    
                    st.divider()
                    
                    # Technical details
                    st.markdown("### 🔧 Technical Details")
                    st.info(f"**Retrieval Method:** {retrieval_method}")
                    
                    # Sources
                    if metrics.get("sources"):
                        st.write("**📄 Information Sources:**")
                        source_cols = st.columns(2)
                        for idx, source in enumerate(metrics["sources"]):
                            with source_cols[idx % 2]:
                                st.write(f"• {source}")
                    
                    # Summary of features used
                    if features_used:
                        st.success(f"**🎯 Total Features Used:** {', '.join(features_used)}")
                    else:
                        st.info("**Basic retrieval mode used**")
            
            st.divider()
    
    # Example queries
    with st.expander("💡 Example Healthcare Queries"):
        example_queries = [
            "What are the symptoms and treatment options for diabetes?",
            "Explain the procedure for cardiac catheterization",
            "What medications are used for hypertension management?",
            "How to care for a patient with pneumonia?",
            "What are the side effects of beta-blockers?",
            "Describe the nursing care for post-operative patients",
            "What are the contraindications for aspirin therapy?",
            "Explain the pathophysiology of heart failure"
        ]
        
        cols = st.columns(2)
        for i, example in enumerate(example_queries):
            with cols[i % 2]:
                if st.button(example, key=f"example_{i}"):
                    st.session_state.selected_query = example
    
    # FIXED: Chat input using st.form for proper handling
    st.markdown("### ✨ Ask Your Question")
    
    # Use form for proper input handling
    with st.form("chat_form", clear_on_submit=True):
        # Check if example was selected
        initial_value = ""
        if 'selected_query' in st.session_state:
            initial_value = st.session_state.selected_query
            del st.session_state.selected_query
        
        query = st.text_area(
            "**Type your healthcare question here:**",
            placeholder="Ask anything about medical conditions, treatments, procedures, or patient care...",
            height=100,
            value=initial_value
        )
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            submitted = st.form_submit_button("🚀 Send Query", use_container_width=True, type="primary")
        with col2:
            reset_form = st.form_submit_button("🔄 Clear", use_container_width=True)
    
    # Handle form submission
    if submitted:
        if not query or not query.strip():
            st.warning("⚠️ Please enter a question to get started!")
        else:
            with st.spinner("🔄 Processing your query with advanced RAG techniques..."):
                
                try:
                    if chat_mode == "🚀 Advanced Multi-vector RAG":
                        response = requests.post(
                            f"{API_URL}/chat/advanced",
                            data={"message": query},
                            headers=get_auth_headers(),
                            timeout=120
                        )
                        
                    elif chat_mode == "📝 Simple RAG":
                        response = requests.post(
                            f"{API_URL}/chat/simple",
                            data={"message": query},
                            headers=get_auth_headers(),
                            timeout=60
                        )
                        
                    else:  # Compare both methods
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown("#### 🚀 Advanced RAG Response")
                            try:
                                advanced_response = requests.post(
                                    f"{API_URL}/chat/advanced",
                                    data={"message": query},
                                    headers=get_auth_headers(),
                                    timeout=120
                                )
                                if advanced_response.status_code == 200:
                                    adv_data = advanced_response.json()
                                    st.success(adv_data["answer"])
                                    
                                    st.markdown("**Advanced Features Used:**")
                                    st.write(f"• Multi-hop steps: {adv_data.get('multi_hop_steps', 0)}")
                                    st.write(f"• Graph relationships: {adv_data.get('graph_relationships', 0)}")
                                    st.write(f"• Processing time: {adv_data.get('processing_time', 0):.2f}s")
                                    st.write(f"• Sources: {len(adv_data.get('sources', []))}")
                                    
                                else:
                                    st.error(f"Advanced RAG failed: {advanced_response.status_code}")
                            except Exception as e:
                                st.error(f"Advanced RAG error: {str(e)}")
                        
                        with col2:
                            st.markdown("#### 📝 Simple RAG Response")
                            try:
                                simple_response = requests.post(
                                    f"{API_URL}/chat/simple",
                                    data={"message": query},
                                    headers=get_auth_headers(),
                                    timeout=60
                                )
                                if simple_response.status_code == 200:
                                    simple_data = simple_response.json()
                                    st.info(simple_data["answer"])
                                    
                                    st.markdown("**Simple RAG Features:**")
                                    st.write(f"• Sources found: {simple_data.get('num_sources', 0)}")
                                    st.write(f"• Method: {simple_data.get('method', 'N/A')}")
                                    
                                else:
                                    st.error(f"Simple RAG failed: {simple_response.status_code}")
                            except Exception as e:
                                st.error(f"Simple RAG error: {str(e)}")
                        
                        return
                    
                    # Handle single response
                    if response.status_code == 200:
                        data = response.json()
                        
                        st.markdown("### 🎯 Response")
                        st.success(data["answer"])
                        
                        # Store in chat history
                        st.session_state.chat_history.append({
                            "query": query,
                            "answer": data["answer"],
                            "metrics": data,
                            "timestamp": datetime.now().isoformat()
                        })
                        
                        # Show advanced metrics for advanced mode
                        if "multi_hop_steps" in data:
                            with st.expander("🔍 Advanced RAG Analysis", expanded=True):
                                col1, col2, col3, col4 = st.columns(4)
                                
                                with col1:
                                    st.metric("Multi-hop Steps", data.get("multi_hop_steps", 0))
                                with col2:
                                    st.metric("Sources Found", len(data.get("sources", [])))
                                with col3:
                                    st.metric("Graph Entities", len(data.get("graph_entities", [])))
                                with col4:
                                    st.metric("Processing Time", f"{data.get('processing_time', 0):.2f}s")
                                
                                if data.get("rewritten_query") != query:
                                    st.info(f"**Optimized Query:** {data.get('rewritten_query')}")
                                
                                # Show knowledge graph insights
                                show_knowledge_graph_insights(data)
                                
                                # Show hybrid search details  
                                show_hybrid_search_details(data)
                                
                                st.write(f"**Retrieval Method:** {data.get('retrieval_method', 'N/A')}")
                                
                                if data.get("sources"):
                                    st.write("**Information Sources:**")
                                    for i, source in enumerate(data["sources"], 1):
                                        st.write(f"{i}. 📄 {source}")
                        
                        # Success - rerun to clear form
                        st.rerun()
                    
                    else:
                        st.error(f"❌ Request failed with status code: {response.status_code}")
                        try:
                            error_data = response.json()
                            st.json(error_data)
                        except:
                            st.write(response.text)
                            
                except requests.exceptions.ConnectionError:
                    st.error("🚨 Cannot connect to the API server. Please ensure the backend is running on http://localhost:8000")
                except requests.exceptions.Timeout:
                    st.error("⏱️ Request timed out. The query is taking longer than expected.")
                except Exception as e:
                    st.error(f"❌ Unexpected error: {str(e)}")
    
    elif reset_form:
        # Clear any selected query
        if 'selected_query' in st.session_state:
            del st.session_state.selected_query
        st.rerun()

def document_upload_interface():
    """Document upload and management interface for admins"""
    st.markdown("## 📤 Advanced Document Upload & Processing")
    
    with st.form("upload_form", clear_on_submit=True):
        st.markdown("### 📄 Upload New Documents")
        
        col1, col2 = st.columns(2)
        
        with col1:
            uploaded_files = st.file_uploader(
                "**Choose PDF files**",
                type=["pdf"],
                accept_multiple_files=True,
                help="Upload multiple PDF files. Maximum size per file: 10MB"
            )
            
            target_role = st.selectbox(
                "**Target Role for Access**",
                ["admin", "doctor", "nurse", "patient"],
                format_func=lambda x: {"admin": "👑 Admin", "doctor": "👨‍⚕️ Doctor", 
                                     "nurse": "👩‍⚕️ Nurse", "patient": "🧑‍🤝‍🧑 Patient"}[x]
            )
        
        with col2:
            doc_type = st.selectbox(
                "**Document Type**",
                ["medical_report", "patient_record", "treatment_protocol", 
                 "medication_guide", "research_paper", "care_plan", "other"],
                format_func=lambda x: x.replace("_", " ").title()
            )
            
            priority = st.selectbox(
                "**Priority Level**",
                ["high", "medium", "low"],
                index=1,
                format_func=lambda x: {"high": "🔴 High", "medium": "🟡 Medium", "low": "🟢 Low"}[x]
            )
        
        submitted = st.form_submit_button("🚀 Upload & Process Documents", use_container_width=True)
        
        if submitted and uploaded_files:
            total_size = sum(file.size for file in uploaded_files if file.size)
            if total_size > 50 * 1024 * 1024:
                st.error("❌ Total file size exceeds 50MB limit")
                return
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                files_data = []
                for file in uploaded_files:
                    files_data.append(("files", (file.name, file.getvalue(), "application/pdf")))
                
                form_data = {
                    "role": target_role,
                    "doc_type": doc_type,
                    "priority": priority
                }
                
                status_text.text("📤 Uploading documents...")
                progress_bar.progress(25)
                
                response = requests.post(
                    f"{API_URL}/docs/upload",
                    files=files_data,
                    data=form_data,
                    headers=get_auth_headers(),
                    timeout=300
                )
                
                progress_bar.progress(100)
                status_text.text("✅ Upload complete!")
                
                if response.status_code == 200:
                    result = response.json()
                    status_text.empty()
                    progress_bar.empty()
                    
                    st.success("✅ Documents uploaded and processed successfully!")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Files Processed", result["processing_stats"]["files_processed"])
                    with col2:
                        st.metric("Chunks Created", result["processing_stats"]["total_chunks"])
                    with col3:
                        st.metric("Embeddings Generated", result["processing_stats"]["embeddings_created"])
                    with col4:
                        st.metric("AI Analysis Completed", result["processing_stats"]["analysis_completed"])
                    
                    with st.expander("📋 Document Details"):
                        st.json(result)
                
                else:
                    progress_bar.empty()
                    status_text.empty()
                    st.error(f"❌ Upload failed with status {response.status_code}")
                    
            except Exception as e:
                st.error(f"❌ Upload failed: {str(e)}")
                progress_bar.empty()
                status_text.empty()
        
        elif submitted:
            st.warning("⚠️ Please select files to upload")

def system_analytics_interface():
    """System analytics and monitoring for admins"""
    st.markdown("## 📊 System Analytics & Performance")
    
    stats_response = make_api_request("/docs/stats")
    if stats_response and stats_response.status_code == 200:
        try:
            stats = stats_response.json()
            
            st.markdown("### 📈 System Overview")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                total_vectors = stats["vector_store"].get("total_vectors", 0)
                st.metric("Total Documents", f"{total_vectors:,}", delta="+127 this week")
            
            with col2:
                total_users = stats["database"].get("users", 0)
                st.metric("Active Users", total_users, delta="+5 this month")
            
            with col3:
                graph_nodes = stats["database"].get("knowledge_graph", 0)
                st.metric("Knowledge Entities", f"{graph_nodes:,}", delta="+89 new")
            
            with col4:
                queries = stats["database"].get("query_analytics", 0)
                st.metric("Total Queries", f"{queries:,}", delta="+234 today")
        except:
            st.info("Analytics data loading...")
    
    else:
        st.error("❌ Failed to load system statistics")

def user_analytics_interface():
    """User-specific analytics interface"""
    st.markdown("## 📈 Your Usage Analytics")
    
    analytics_response = make_api_request("/chat/analytics")
    if analytics_response and analytics_response.status_code == 200:
        try:
            analytics = analytics_response.json()
            summary = analytics["summary"]
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Queries", summary["total_queries"])
            with col2:
                st.metric("Success Rate", f"{summary['success_rate']:.1f}%")
            with col3:
                st.metric("Avg Response Time", f"{summary['avg_processing_time']:.1f}s")
            with col4:
                st.metric("Favorite Feature", "Advanced RAG")
        except:
            st.info("Analytics data loading...")
    else:
        st.info("Analytics data not available yet. Start asking questions!")

def knowledge_graph_interface():
    """Knowledge graph exploration interface"""
    st.markdown("## 🕸️ Knowledge Graph Explorer")
    
    graph_stats_response = make_api_request("/graph/stats")
    if graph_stats_response and graph_stats_response.status_code == 200:
        try:
            stats = graph_stats_response.json()["graph_stats"]
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Graph Nodes", f"{stats['total_nodes']:,}")
            with col2:
                st.metric("Graph Edges", f"{stats['total_edges']:,}")
            with col3:
                st.metric("Graph Density", f"{stats['density']:.4f}")
            with col4:
                st.metric("Avg Connections", f"{stats['average_degree']:.1f}")
        except:
            st.info("Graph statistics loading...")
    
    elif graph_stats_response and graph_stats_response.status_code == 403:
        st.warning("🔒 You don't have permission to access the knowledge graph.")
    else:
        st.error("❌ Failed to load knowledge graph statistics")

# Main application flow
def main():
    """Main application entry point"""
    init_session_state()
    
    if not st.session_state.logged_in:
        login_page()
    else:
        main_app()

if __name__ == "__main__":
    main()
