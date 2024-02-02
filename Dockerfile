FROM python:3.11

# Expose port you want your app on
EXPOSE 8080

WORKDIR /app

# Upgrade pip and install requirements
COPY requirements.txt requirements.txt
RUN pip install -U pip
RUN pip install -r requirements.txt

# Copy app code and set working directory
COPY chromadb chromadb
COPY custom_chains custom_chains
COPY custom_prompts custom_prompts
COPY custom_tools custom_tools
COPY keys.json keys.json
RUN mkdir -p custom_agents
COPY custom_agents/__init__.py custom_agents/__init__.py
COPY custom_agents/custom_structured_chat_agent_v4.py custom_agents/custom_structured_chat_agent_v4.py
COPY run.py app.py
COPY .env.prod .env


HEALTHCHECK CMD curl --fail http://localhost:8080/_stcore/health

# Run
ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8080", "--server.address=0.0.0.0", "--server.enableCORS=False", "--server.enableXsrfProtection=False", "--server.enableWebsocketCompression=false", "--server.headless=true"]