# 🏥 Advanced Healthcare RAG System

<div align="center">

![Healthcare RAG](https://img.shields.io/badge/Healthcare-RAG%20System-blue?style=for-the-badge&logo=health)
![Python](https://img.shields.io/badge/Python-3.8+-blue?style=for-the-badge&logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104.1-green?style=for-the-badge&logo=fastapi)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28.1-red?style=for-the-badge&logo=streamlit)
![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)

**A sophisticated Retrieval-Augmented Generation (RAG) system designed for healthcare professionals, featuring multi-vector hybrid search, knowledge graph integration, and chain-of-thought reasoning capabilities.**

[🚀 Live Demo](https://role-based-medical-advanced-rag-system-69x3p2dcrk3rm6hcnpsp6a.streamlit.app/) 

</div>

---

## ✨ Features

### 🔍 **Advanced Retrieval**
- **Multi-Vector Hybrid Search** - Combines semantic (vector) and keyword (BM25) search for superior accuracy
- **Intelligent Query Rewriting** - Optimizes user queries for better information retrieval
- **Multi-hop Reasoning** - Chain-of-thought processing for complex medical queries

### 🕸️ **Knowledge Graph Integration**
- **Medical Entity Extraction** - Automatically identifies medical terms, conditions, and treatments
- **Relationship Mapping** - Creates connections between medical concepts across documents
- **Graph-Enhanced Retrieval** - Uses knowledge graphs to find related information

### 👥 **Role-Based Access Control**
- **Multi-Role Support** - Admin, Doctor, Nurse, and Patient access levels
- **Permission-Based Features** - Different capabilities based on user roles
- **Secure Authentication** - JWT-based security with role-specific permissions

### 📄 **Document Processing**
- **Advanced PDF Analysis** - Multi-strategy chunking and content extraction
- **AI-Powered Summarization** - Automatic document summarization and analysis
- **Vector Embeddings** - High-quality document representations for semantic search


---

## 🏗️ Project Structure

```
MODULAR RAG-BASED MEDICAL ASSISTANT/
├── client/                     # Frontend Application
│   ├── main.py                # Streamlit main application
│   ├── requirements.txt       # Frontend dependencies
│   └── .gitignore            # Git ignore rules
│
└── server/                    # Backend API
    ├── main.py               # FastAPI main application
    ├── requirements.txt      # Backend dependencies
    ├── .env                 # Environment variables
    ├── pyproject.toml       # Project configuration
    │
    ├── auth/                # Authentication Module
    │   ├── __init__.py
    │   ├── models.py        # User models and schemas
    │   ├── routes.py        # Authentication endpoints
    │   └── hash_utils.py    # Password hashing and JWT
    │
    ├── chat/                # Chat & RAG Module
    │   ├── __init__.py
    │   ├── routes.py        # Chat API endpoints
    │   ├── advanced_rag.py  # Advanced RAG implementation
    │   └── chat_query.py    # Simple RAG implementation
    │
    ├── config/              # Configuration Module
    │   ├── __init__.py
    │   └── db.py           # Database configuration
    │
    ├── docs/               # Document Processing Module
    │   ├── __init__.py
    │   ├── routes.py       # Document API endpoints
    │   └── vector_database.py # Vector store operations
    │
    └── graph/              # Knowledge Graph Module
        ├── __init__.py
        └── routes.py       # Graph API endpoints
```

---

## 🛠️ Tech Stack

### **Backend**
- **FastAPI** - High-performance Python web framework
- **MongoDB** - Document database for user data and metadata
- **Pinecone** - Vector database for semantic search
- **NetworkX** - Knowledge graph processing

### **AI/ML Components**
- **LangChain** - LLM application framework
- **Google Generative AI** - Text embeddings (models/embedding-001)
- **Groq** - Fast LLM inference (Llama3-8b-8192)
- **PyPDF** - PDF document processing

### **Frontend**
- **Streamlit** - Interactive web application framework
- **Plotly** - Interactive data visualizations
- **Custom CSS** - Enhanced UI styling

### **Security & Authentication**
- **JWT** - JSON Web Tokens for authentication
- **Bcrypt** - Password hashing
- **Role-based permissions** - Fine-grained access control

---

## ⚡ Quick Start

### Prerequisites
- Python 3.8+
- MongoDB instance
- Pinecone account
- Google AI API key
- Groq API key

### 1. Clone Repository
```bash
git clone https://github.com/your-username/healthcare-rag-system.git
cd healthcare-rag-system
```

### 2. Backend Setup
```bash
cd server

# Install dependencies
pip install -r requirements.txt

# Create environment file
cp .env.example .env

# Configure your API keys in .env
MONGO_URL=your_mongodb_connection_string
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_INDEX_NAME=medical-assistant-index
PINECONE_ENV=your_pinecone_environment
GOOGLE_API_KEY=your_google_ai_api_key
GROQ_API_KEY=your_groq_api_key
JWT_SECRET_KEY=your_jwt_secret_key

# Run backend server
uvicorn main:app --reload --port 8000
```

### 3. Frontend Setup
```bash
cd ../client

# Install dependencies
pip install -r requirements.txt

# Run Streamlit app
streamlit run main.py
```

### 4. Access Application
- **Backend API:** http://localhost:8000
- **Frontend App:** http://localhost:8501
- **API Documentation:** http://localhost:8000/docs

### 5. Default Login Credentials
```
Username: admin
Password: admin123
```

---

## 🚀 Live Demo

### 🌐 **Production Deployment**
**Frontend:** [https://role-based-medical-advanced-rag-system-69x3p2dcrk3rm6hcnpsp6a.streamlit.app/](https://role-based-medical-advanced-rag-system-69x3p2dcrk3rm6hcnpsp6a.streamlit.app/)

### 🧪 **Test Accounts**
| Role | Username | Password | Access Level |
|------|----------|----------|--------------|
| Admin | `admin` | `admin123` | Full system access |
| Doctor | `doctor` | `doctor123` | Medical records, patient data |
| Nurse | `nurse` | `nurse123` | Patient care, protocols |
| Patient | `patient` | `patient123` | Personal health data |

---

## 📚 API Documentation

### Authentication Endpoints
```http
POST /auth/login      # User login
POST /auth/signup     # User registration
GET  /auth/me         # Get current user info
POST /auth/logout     # User logout
```

### Chat Endpoints
```http
POST /chat/advanced    # Advanced RAG query
POST /chat/simple      # Simple RAG query
GET  /chat/capabilities # Get user capabilities
GET  /chat/analytics   # Get user analytics
```

### Document Endpoints
```http
POST /docs/upload     # Upload and process documents
POST /docs/search     # Search documents
GET  /docs/stats      # Get document statistics
GET  /docs/types      # Get document types
```

### Knowledge Graph Endpoints
```http
GET  /graph/stats           # Get graph statistics
GET  /graph/entities/{entity} # Get entity details
POST /graph/query           # Execute graph queries
GET  /graph/search          # Search entities
```

---

## 🎯 Use Cases

### **For Healthcare Professionals**
- **Clinical Decision Support** - Get evidence-based treatment recommendations
- **Medical Research** - Search through medical literature and guidelines
- **Patient Education** - Generate patient-friendly explanations
- **Care Protocol Guidance** - Access standardized care procedures

### **For Healthcare Institutions**
- **Knowledge Management** - Centralized medical information repository
- **Training & Education** - Interactive learning for medical staff
- **Quality Assurance** - Ensure adherence to medical standards
- **Research Support** - Facilitate medical research and analysis

### **For Patients**
- **Health Information** - Access reliable medical information
- **Symptom Guidance** - Get preliminary health insights
- **Treatment Understanding** - Learn about conditions and treatments
- **Medication Information** - Understand prescriptions and side effects

---

## 🔧 Configuration

### Environment Variables
```bash
# Database
MONGO_URL=mongodb://localhost:27017
DB_NAME=healthcare_rag_db

# Vector Database
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_INDEX_NAME=medical-assistant-index
PINECONE_ENV=us-west1-gcp

# AI Models
GOOGLE_API_KEY=your_google_ai_api_key
GROQ_API_KEY=your_groq_api_key

# Security
JWT_SECRET_KEY=your_jwt_secret_key
JWT_ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_HOURS=24

# Application
API_URL=http://localhost:8000
DEBUG=True
```



## 🚀 Deployment

### Docker Deployment
```bash
# Build and run with Docker Compose
docker-compose up --build

# Or individually
docker build -t healthcare-rag-backend ./server
docker build -t healthcare-rag-frontend ./client
```

### Production Deployment
- **Backend:** Deployed on Render
- **Frontend:** Deployed to Streamlit Cloud
- **Database:** MongoDB Atlas or self-hosted
- **Vector DB:** Pinecone cloud service

---

## 🤝 Contributing

We welcome contributions! 

---

## 📝 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **LangChain** - For the powerful RAG framework
- **Streamlit** - For the amazing web app framework
- **FastAPI** - For the high-performance backend framework
- **Pinecone** - For the vector database platform
- **Google AI** - For the embedding models
- **Groq** - For fast LLM inference

---

## 📞 Support
- **Email:** riteshbandaru27@gmail.com

---

**⭐ Star this repository if you found it helpful!**

Built with ❤️ for healthcare professionals worldwide

