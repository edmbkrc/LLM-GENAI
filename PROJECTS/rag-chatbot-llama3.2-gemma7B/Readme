# RAG Chatbot with Llama & Gemma

This project is a Retrieval-Augmented Generation (RAG) chatbot that uses Llama and Gemma language models to answer user queries based on uploaded documents. The chatbot processes documents, stores them in a vector database, and retrieves relevant information to generate accurate responses.

## Features

- **Document Support**: Upload PDF, DOC, and DOCX files.
- **Retrieval-Augmented Generation**: Combines document retrieval with LLM-based generation for accurate answers.
- **LLM Integration**:
  - **Llama**: A powerful language model for detailed responses.
  - **Gemma**: A concise and precise language model for efficient answers.
- **User Authentication**: Optional authentication for secure access.
- **Persistent Vectorstore**: Dynamically updates vectorstore with new document uploads.


### Prerequisites

Ensure you have the following installed:
- Docker
- Python 3.9+
- [Ollama](https://ollama.ai/) (for Llama-based processing)


### Installation
   ```bash
   # download necessary tools for ubuntu
   sudo apt update && sudo apt upgrade -y
   sudo apt install -y git curl docker.io docker-compose
   
   # check docker and docker compose
   docker --version
   docker-compose --version
   
   # clone project
   git clone https://github.com/edmbkrc/TASK-JSL2.git
   cd TASK-JSL2

   # Use Docker Compose to build and start all the services
   docker-compose up -d --build
   
   # Check if the containers are running
   docker ps

   #  IP address of your Ubuntu machine
   hostname -I

   # You can access your application using the following URLs
   http://<Ubuntu_IP_Address>:8501

   # If you encounter any issues, check the logs
   docker-compose logs
   docker-compose logs <service-name>  # e.g., ollama or app

   # To reset and restart all services, run
   docker-compose down
   docker-compose up -d --build

   """
   Ensure all dependencies are correctly configured in the docker-compose.yml file. Pay special attention to commands like ollama pull llama3.2:1b.
   Verify that the required model for Ollama is pulled correctly:
   """
