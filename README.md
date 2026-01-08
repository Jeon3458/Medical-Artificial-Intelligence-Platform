# Hi Medie - Medical AI Platform (Personal Contribution Highlights)

> **Note**: This is a personal repository highlighting my specific contributions to the Medical AI Platform project, focusing on **Privacy-Preserving Agents** and **Agent Collaboration**.

---

<table>
  <tr align="center">
    <td width="320px">
      <a href="https://github.com/isshoman123" target="_blank">
        <img src="https://avatars.githubusercontent.com/isshoman123" alt="isshoman123" />
      </a>
    </td>
    <td width="320px">
      <a href="https://github.com/dongsinwoo" target="_blank">
        <img src="https://avatars.githubusercontent.com/dongsinwoo" alt="dongsinwoo" />
      </a>
    </td>
    </td>
        <td width="320px">
      <a href="https://github.com/Jeon3458" target="_blank">
        <img src="https://avatars.githubusercontent.com/Jeon3458" alt="Jeon3458" />
      </a>
    </td>
    <td width="320px">
      <a href="https://github.com/espada105" target="_blank">
        <img src="https://avatars.githubusercontent.com/espada105" alt="espada105" />
      </a>
  </tr>
  <tr align="center">
    <td>
      Jaewon Kim
    </td>
    <td>
      Dongwoo Shin
    </td>
    <td>
      Hyunseong Jeon
    </td>
    <td>
      Seongin Hong
    </td>
  </tr>
    <tr align="center">
    <td>
      jaewon
    </td>
    <td>
      dongwoo  
    </td>
    <td>
      Agent-based Masking via Encapsulation, A2A-based Collaborative Security Workflow
    </td>
    <td>
      Building MCP servers, loading MCPs, building vector stores, generating virtual medical patient data, connecting AI agents to MCPs
    </td>
  </tr>  
</table>

## My Key Contributions (Hyunseong Jeon)

I designed the autonomous security workflow by transforming a standard masking library into a functional AI agent.

### 1. Agent-based Masking via Encapsulation

Concept: Instead of calling a simple function, I encapsulated the Microsoft Presidio library into a standalone "Masking Agent."

Significance: By treating the masking process as an independent agent (Agent-as-a-Service), I ensured the system's modularity, allowing any other agent to easily request security features.

### 2. A2A-based Collaborative Security Workflow

The Process: For example, when a PDF Agent processes a medical document containing sensitive info, it autonomously calls the Masking Agent via the A2A protocol to redact PII before outputting the data.

Autonomous Cooperation: Implemented a "Privacy-by-Design" architecture where agents collaborate to ensure data security without human intervention.

## Medical Artificial Intelligence Platform

![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)
![Python](https://img.shields.io/badge/Python-3.8+-green.svg)
![A2A](https://img.shields.io/badge/Protocol-A2A-orange.svg)
![MCP](https://img.shields.io/badge/Protocol-MCP-purple.svg)

**A Medical AI Agent Platform Leveraging Agent2Agent (A2A) Protocol and Model Context Protocol (MCP)**

---

## Project Overview

Hi Medie is an AI-powered medical platform designed to enhance the efficiency of healthcare professionals. By integrating the A2A and MCP protocols, the platform seamlessly connects with various medical systems and enables clinicians to efficiently manage patient data and automatically generate clinical documentation.

### Key Objectives

- **Standardized Medical AI Agent Communication**: Agent collaboration based on the A2A protocol
- **External System Integration**: Connection to hospital databases, medical records, and literature search systems via MCP
- **Medical Workflow Automation**: Automation of clinical documentation, patient search, and prescription management
- **Scalable Architecture**: Easy integration of new medical systems

---

## Architecture

```
[Clinicians] ↔ [A2A Client] ↔ [A2A Server] ↔ [MCP Client] ↔ [Medical Systems]
↓
[Medical AI Agents]
↓
[Patient Data / Medical Knowledge]
```

### Core Components

- **A2A Server**: Agent-to-agent communication and task management
- **MCP Client**: Integration with external medical systems
- **Medical AI Agents**: Patient data analysis and clinical decision support
- **Vector Database**: Similar patient search and medical knowledge retrieval

---

## Key Features

### Patient Management System

- **Smart Patient Search**: Multi-dimensional search by ID, name, and symptoms
- **Vector Similarity Search**: Symptom-based similar patient discovery
- **Department-Based Classification**: Internal medicine / surgery / same-day care management
- **Prescription History Tracking**: Search prescription history for specific medications

### Clinical Documentation Automation

- **SOAP Note Generation**: Automatic creation of structured clinical notes
- **Prescription Writing**: AI-assisted medication recommendations
- **Clinical Summary Reports**: Comprehensive analysis of patient history
- **Handover Documents**: Patient status handoff documentation for shift changes

### AI Analytics Engine

- **Symptom Analysis**: Diagnosis support based on patient symptoms
- **Pattern Analysis**: Predictive insights from historical medical records
- **Triage & Severity Assessment**: Automated patient priority evaluation
- **Drug Interaction Checks**: Prescription safety verification

### MCP External System Integration

- **PubMed Integration**: Real-time medical literature search
- **Hospital Database Connectivity**: Unified access to patient record systems
- **Electronic Medical Records (EMR)**: EMR system integration
- **Drug Databases**: Medication information and side effect lookup
- **Test Result Systems**: Access to laboratory and radiology results

---

## Installation & Execution

### 1. Environment Setup

```bash
# Clone repository
git clone https://github.com/your-repo/hi_medei.git
cd hi_medei

# Python environment setup (Python 3.11+ recommended)
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies (per agent)
# Medical Agent
cd hi_medei/samples/python/agents/medical_agent
pip install -r requirements_simplified.txt
# pip install -r requirements_complete.txt
pip install -r requirements_mcp.txt

# Medical Image Analysis Agent
cd hi_medei/samples/python/agents/medical_image_agent
pip install -r requirements.txt

# PDF QA Agent
cd hi_medei/samples/python/agents/langgraph
pip install -e .
```

### 2. Environment Variable Setup

```bash
# Create .env file for medical agent
cd hi_medei/samples/python/agents/medical_agent
cp env_example.txt .env

# Edit .env file and set required API keys
# OPENAI_API_KEY=your_openai_api_key
# GEMINI_API_KEY=your_gemini_api_key (optional)
```

### 3. Start Servers

#### Start Medical Agent Server (including MCP servers)

```bash
# Start medical agent and MCP servers together (separate terminal)
cd hi_medei/samples/python/agents/medical_agent
python start_all.py
```

#### Start Medical Image Analysis Agent Server

```bash
# Start medical image analysis A2A server (separate terminal)
cd hi_medei/samples/python/agents/medical_image_agent
python __main__.py
```

#### Start PDF QA Agent Server

```bash
# Start LangGraph-based PDF QA agent server (separate terminal)
cd hi_medei/samples/python/agents/langgraph
python __main__.py
```

### 4. Server Health Check

Verify that each server is running properly:

```bash
# MCP server health check (automatically started with medical agent)
curl http://localhost:8080/health  # PubMed server
curl http://localhost:8081/health  # Memory server

# A2A server health check
curl http://localhost:10001/health  # Medical agent
curl http://localhost:10002/health  # Medical image analysis agent
curl http://localhost:10003/health  # PDF QA agent
```

### 5. Test Execution (Medical Agent Only)

```bash
# Medical agent tests
cd hi_medei/samples/python/agents/medical_agent

# Basic tests
python test_agent_simple.py

# MCP integration tests
python test_agent_mcp.py
python test_mcp_connection.py
python test_mcp_integration.py

# Vector search tests
python test_vector.py
```

### 6. Troubleshooting

If issues occur during server execution:

1. Check for port conflicts: lsof -i :[PORT] (Unix/Mac) or netstat -ano | findstr [PORT] (Windows)
2. Verify environment variables: python -c "import os; print(os.getenv('OPENAI_API_KEY'))"
3. Check dependencies: Review the requirements files in each agent directory
4. Check logs: Review console logs for each server

### 7. Development Mode (Medical Agent Only)

```bash
# Medical agent debug mode
cd hi_medei/samples/python/agents/medical_agent

# Agent invocation debugging
python debug_invoke.py

# Search functionality debugging
python debug_search.py

# Query analysis debugging
python debug_query.py
```

## Usage Examples

### A2A Protocol Agent Invocation

```python
import asyncio
from common.client import A2AClient

async def main():
    # Connect to medical agent
    client = A2AClient("http://localhost:10001")

    # Patient search request
    response = await client.send_task({
        "message": {
            "text": "Please retrieve the most recent medical records for patient Kim Cheolsu"
        }
    })

    print(response)

asyncio.run(main())
```

### External System Invocation via MCP

```bash
curl -X POST http://localhost:10001 \
  -H "Content-Type: application/json" \
  -d '{
    "jsonrpc": "2.0",
    "method": "mcp/search_pubmed",
    "params": {
      "query": "diabetes treatment guidelines",
      "max_results": 5
    },
    "id": 1
  }'
```

---

## Development Guide

### Project Structure

```
hi_medei/
├── demo_clovax_05/
│ ├── ui/
│ │ ├── main.py
│ │ ├── components/
│ │ │ ├── form_render.py
│ │ │ └── ...
│ │ └── static/
│ └── requirements.txt
│
├── samples/python/agents/
│ ├── medical_agent/
│ │ ├── main.py
│ │ ├── agent.py
│ │ ├── medical_tools.py
│ │ ├── mcp_client.py
│ │ ├── mcp_config.py
│ │ └── task_manager.py
│ │
│ ├── medical_image_agent/
│ │ ├── main.py
│ │ ├── agent.py
│ │ ├── simple_vision_pipeline.py
│ │ └── task_manager.py
│ │
│ ├── langgraph/
│ │ ├── main.py
│ │ ├── agent.py
│ │ ├── task_manager.py
│ │ └── chroma_db/
│ │
│ └── masking_agent/
│   ├── app/
│   ├── a2a/
│   ├── start_masking_agent.py
│   └── requirements.txt
│
├── docs/
├── specification/
├── tests/
├── lychee.toml
├── mkdocs.yml
├── noxfile.py
└── requirements-docs.txt
```

### Adding a New MCP Server

```python
new_endpoint = MCPEndpoint(
    name="new_system",
    url="http://localhost:8085/mcp/new",
    description="New medical system"
)
mcp_manager.add_endpoint(new_endpoint)
```

### Developing Custom Medical Tools

```python
from medical_tools import BaseTool

class CustomMedicalTool(BaseTool):
    name = "custom_tool"
    description = "Custom medical tool"

    def _run(self, query: str) -> str:
        return "Processing result"
```

---

## Security and Compliance

### Medical Information Protection

- HIPAA Compliance: Encryption and access control for patient information

- Data Anonymization: Protection of real patient data in development and testing environments

- Audit Logs: Logging of all medical data access history

---

**© 2025 한서대학교 융합프로젝트 Agentic AI HiMedei Team. All rights reserved.**
