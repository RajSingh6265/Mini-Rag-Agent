# Medical RAG Agent - AI-Powered Medical Notes Analysis

A production-ready Retrieval-Augmented Generation (RAG) system for analyzing medical patient notes using LangChain, Groq LLM, and FastAPI. Deployed as a containerized microservice on AWS ECS.

## 🔧 Architecture

- **RAG Pipeline**: LangChain with TF-IDF vector embeddings
- **LLM Provider**: Groq (llama-3.1-8b-instant)
- **API Framework**: FastAPI
- **Containerization**: Docker
- **Cloud Deployment**: AWS ECS (Fargate)
- **Vector Store**: In-memory TF-IDF similarity search

## 🚀 Quick Start

### Local Development

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd medical-rag-agent
   ```

2. **Set up environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Configure API key**
   ```bash
   # Create .env file
   echo "OPENAI_API_KEY=your_groq_api_key_here" > .env
   ```

4. **Run the application**
   ```bash
   # CLI version
   python agent.py
   
   # API server
   uvicorn app:app --reload --port 8000
   ```

### Docker Deployment

1. **Build Docker image**
   ```bash
   docker build -t medical-rag-agent .
   ```

2. **Run container**
   ```bash
   docker run -p 8000:8000 medical-rag-agent
   ```

3. **Test API endpoint**
   ```bash
   curl -X POST "http://localhost:8000/ask" \
        -H "Content-Type: application/json" \
        -d '{"question": "Which patients had Pneumonia?"}'
   ```

## 📋 API Reference

### Endpoints

#### `POST /ask`
Analyze medical notes and answer questions.

**Request Body:**
```json
{
  "question": "Which patients had Pneumonia?"
}
```

**Response:**
```json
{
  "answer": "Patient Name: Sarah Johnson\nAge: 32\nDiagnosis: Pneumonia (bacterial)\nTreatment: Amoxicillin 875mg twice daily for 10 days, rest, increased fluid intake.\n\nPatient Name: Robert Wilson\nAge: 67\nDiagnosis: Pneumonia (viral)\nTreatment: Supportive care with rest, fluids, acetaminophen for fever. Oxygen therapy initiated due to low saturation levels."
}
```

#### `GET /`
Health check endpoint.

**Response:**
```json
{
  "message": "Medical RAG Agent API is running."
}
```

## 📊 Sample Queries & Outputs

### Query 1: Patient Diagnosis Search
**Input:**
```json
{
  "question": "Which patients had Pneumonia?"
}
```

**Output:**
```
Patient Name: Sarah Johnson
Age: 32
Diagnosis: Pneumonia (bacterial)
Treatment: Amoxicillin 875mg twice daily for 10 days, rest, increased fluid intake.

Patient Name: Robert Wilson
Age: 67
Diagnosis: Pneumonia (viral)
Treatment: Supportive care with rest, fluids, acetaminophen for fever. Oxygen therapy initiated due to low saturation levels.
```

### Query 2: Treatment Frequency Analysis
**Input:**
```json
{
  "question": "What treatment was prescribed most frequently?"
}
```

**Output:**
```
Most Frequent Treatment: Metformin
Frequency: 2
```

### Query 3: General Medical Questions
**Input:**
```json
{
  "question": "What are the common symptoms mentioned in the notes?"
}
```

**Output:**
```
The common symptoms mentioned include fever, low oxygen saturation, chest symptoms related to pneumonia, elevated blood pressure, migraine headaches, anxiety symptoms, and asthma-related breathing difficulties.
```

## 🏥 Medical Data

The system analyzes 10 synthetic patient records covering:
- **Diagnoses**: Diabetes, Pneumonia, Hypertension, Migraines, Anxiety, Asthma
- **Treatments**: Medications, dosages, therapy recommendations
- **Patient Demographics**: Ages 28-67, various conditions

## ☁️ Cloud Deployment

### AWS ECS Deployment

1. **Push to ECR**
   ```bash
   aws ecr create-repository --repository-name medical-rag-agent
   docker tag medical-rag-agent:latest <account-id>.dkr.ecr.<region>.amazonaws.com/medical-rag-agent:latest
   docker push <account-id>.dkr.ecr.<region>.amazonaws.com/medical-rag-agent:latest
   ```

2. **Deploy on ECS**
   - Create ECS cluster (Fargate)
   - Create task definition with ECR image
   - Set environment variable: `OPENAI_API_KEY`
   - Configure port mapping: 8000
   - Launch service with public IP

3. **Test public endpoint**
   ```bash
   curl -X POST "http://<public-ip>:8000/ask" \
        -H "Content-Type: application/json" \
        -d '{"question": "Which patients had Diabetes?"}'
   ```

## 🛠️ Requirements

```
fastapi==0.104.1
uvicorn==0.24.0
langchain==0.1.0
langchain-community==0.0.10
langchain-openai==0.0.5
scikit-learn==1.3.2
numpy==1.24.4
python-dotenv==1.0.0
```

## 📁 Project Structure

```
medical-rag-agent/
├── agent.py           # Core RAG pipeline
├── app.py             # FastAPI web service
├── Dockerfile         # Container configuration
├── requirements.txt   # Python dependencies
├── .env              # Environment variables (API key)
└── README.md         # This file
```

## 🔐 Security Notes

- Store API keys in environment variables, never in code
- Use IAM roles for AWS deployments
- Restrict security group access in production
- Consider using AWS Secrets Manager for production secrets

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🎯 Live Demo

**Public Endpoint:** `http://13.200.20.87:8000`

**Example Usage:**
```bash
curl -X POST "http://13.200.20.87:8000/ask" \
     -H "Content-Type: application/json" \
     -d '{"question": "Which patients had Hypertension?"}'
```

---

**Built with ❤️ for healthcare AI applications**
