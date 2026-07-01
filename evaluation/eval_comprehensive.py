"""
Comprehensive Evaluation: Ablation + Multi-Model + Temperature Sensitivity
==========================================================================
Follows evaluation methodology of top ArXiv papers (FActScore, HaluEval).

Experiments:
  1. Ablation study — 6 defense configurations (incremental layers)
  2. Multi-model — GPT-4.1-nano, GPT-4o-mini, Llama-3.1-8b (Groq)
  3. Temperature sensitivity — temp=0, 0.3, 0.7, 1.0
  4. Statistical analysis — per-resume rates, std dev, 95% CI

All synthetic data. Reproducible: seed=42.
"""

import sys, os, json, time, random, copy, math
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / ".env")

CODEBASE = Path(__file__).resolve().parent.parent.parent / "evidence" / "resumeai-main"
sys.path.insert(0, str(CODEBASE))

from app.cloud_taxonomy import detect_cloud_providers, detect_role_contamination

SEED = 42
random.seed(SEED)

# ============================================================
# TECH RELEASE DATES
# ============================================================
TECH_RELEASE_DATES = {
    "langchain": 2022, "llamaindex": 2022, "llama index": 2022,
    "vertex ai": 2021, "mixtral": 2023, "chatgpt": 2022,
    "gpt-4": 2023, "gpt-3.5": 2022, "bedrock": 2023,
    "rag": 2022, "vector database": 2022, "pinecone": 2021,
    "chromadb": 2022, "weaviate": 2021, "openai api": 2020,
    "copilot": 2021, "stable diffusion": 2022, "midjourney": 2022,
    "langsmith": 2023, "langgraph": 2024, "autogen": 2023,
    "crewai": 2023, "dspy": 2023, "llm": 2022,
    "large language model": 2022, "generative ai": 2022,
    "transformer-based": 2018, "bert": 2018, "gpt-3": 2020,
    "prompt engineering": 2022, "fine-tuning llm": 2022,
    "retrieval augmented": 2022, "embedding model": 2022,
}

# ============================================================
# LLM PROVIDERS
# ============================================================

def get_llm(model_key: str, temperature: float = 0.0):
    """Create LLM instance by key."""
    if model_key == "gpt-4.1-nano":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model="gpt-4.1-nano", temperature=temperature, seed=SEED)
    elif model_key == "gpt-4o-mini":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model="gpt-4o-mini", temperature=temperature, seed=SEED)
    elif model_key == "llama-3.1-8b":
        from langchain_groq import ChatGroq
        return ChatGroq(model="llama-3.1-8b-instant", temperature=temperature)
    else:
        raise ValueError(f"Unknown model: {model_key}")


# ============================================================
# HALLUCINATION DETECTORS (always run, regardless of config)
# ============================================================

def detect_temporal(original: dict, updated: dict) -> List[str]:
    violations = []
    end_date = original.get("end_date", "Present")
    if end_date in ("Present", "current", "now", None, ""):
        return []
    try:
        from dateutil.parser import parse
        end_year = parse(str(end_date)).year
    except:
        return []
    upd_text = " ".join(updated.get("responsibilities", [])).lower()
    orig_text = " ".join(original.get("responsibilities", [])).lower()
    for tech, release_year in TECH_RELEASE_DATES.items():
        if tech in upd_text and tech not in orig_text and end_year < release_year:
            violations.append(f"'{tech}' (released {release_year}) in role ending {end_year}")
    return violations


def detect_structural(original: dict, updated: dict) -> List[str]:
    violations = []
    oc = len(original.get("responsibilities", []))
    uc = len(updated.get("responsibilities", []))
    if uc < oc - 1:
        violations.append(f"Lost {oc - uc} bullets ({oc}->{uc})")
    return violations


def detect_fabrication(original: dict, updated: dict) -> List[str]:
    violations = []
    if updated.get("company", "") != original.get("company", ""):
        violations.append(f"Company changed: '{original.get('company')}' -> '{updated.get('company')}'")
    orig_title = original.get("title", "").lower()
    upd_title = updated.get("title", "").lower()
    if orig_title != upd_title:
        ow = set(orig_title.split())
        uw = set(upd_title.split())
        if len(ow & uw) < len(ow) * 0.5:
            violations.append(f"Title changed: '{original.get('title')}' -> '{updated.get('title')}'")
    return violations


def detect_all(original: dict, updated: dict) -> Dict[str, List[str]]:
    contam, _, _ = detect_role_contamination(original, updated)
    return {
        "H1": detect_temporal(original, updated),
        "H2": [f"Added {c['cloud']} ({c['technology']})" for c in contam],
        "H3": detect_structural(original, updated),
        "H4": detect_fabrication(original, updated),
    }


# ============================================================
# DEFENSE CONFIGURATIONS (ablation)
# ============================================================
# Each config defines: prompt_template, post_hoc_actions

CONFIGS = {
    "baseline": {
        "desc": "No defense — simple rewrite prompt",
        "layers": [],
    },
    "prompt_only": {
        "desc": "L4 only — grounding rules in prompt, no deterministic layers",
        "layers": ["L4"],
    },
    "L1_temporal": {
        "desc": "L1+L4 — temporal constraints + prompt grounding",
        "layers": ["L1", "L4"],
    },
    "L1_L2": {
        "desc": "L1+L2+L4 — temporal + contamination detection + prompt",
        "layers": ["L1", "L2", "L4"],
    },
    "L1_L2_L3": {
        "desc": "L1+L2+L3+L4 — all deterministic + prompt",
        "layers": ["L1", "L2", "L3", "L4"],
    },
    "full": {
        "desc": "All layers (L1+L2+L3+L4) — full framework",
        "layers": ["L1", "L2", "L3", "L4"],
    },
}


def build_prompt(role, jd_text, config_name, layers):
    """Build rewrite prompt based on active layers."""
    from langchain_core.prompts import PromptTemplate

    if config_name == "baseline":
        template = """Rewrite this resume role to better match the job description. Make bullet points impactful and keyword-rich.

Role: {title} at {company} ({start_date} to {end_date})
Bullets:
{responsibilities}

Job Description:
{jd}

Return JSON: {{"title": "...", "company": "...", "start_date": "...", "end_date": "...", "responsibilities": [...]}}
Only JSON, nothing else."""
        variables = {
            "title": role["title"], "company": role["company"],
            "start_date": role.get("start_date", ""), "end_date": role.get("end_date", "Present"),
            "responsibilities": "\n".join(f"- {r}" for r in role["responsibilities"]),
            "jd": jd_text,
        }
        return PromptTemplate.from_template(template), variables

    # Build constraint blocks based on active layers
    constraints = []

    if "L4" in layers:
        constraints.extend([
            "Do NOT change the company name or fabricate metrics not present in original.",
            "Keep achievements realistic for the role level and time period.",
            "Do NOT add technologies the candidate didn't use.",
        ])

    if "L1" in layers:
        end_date = role.get("end_date", "Present")
        if end_date not in ("Present", "current", "now", None, ""):
            try:
                from dateutil.parser import parse
                ey = parse(str(end_date)).year
                banned = [f"{t}({y})" for t, y in TECH_RELEASE_DATES.items() if y > ey][:10]
                constraints.append(f"Role ended {ey}. Do NOT mention tech released after {ey}. Banned: {', '.join(banned)}.")
            except:
                pass
        else:
            constraints.append("Current role — modern technologies acceptable.")

    if "L2" in layers:
        orig_text = " ".join(role.get("responsibilities", []))
        clouds = detect_cloud_providers(orig_text)
        if clouds:
            constraints.append(f"Original cloud providers: {clouds}. Use ONLY these. Do NOT add services from other cloud providers.")
        else:
            constraints.append("Original role has NO cloud services. Do NOT add any cloud-specific services (AWS, Azure, GCP).")

    if "L3" in layers:
        n = len(role["responsibilities"])
        constraints.append(f"Return EXACTLY {n} bullet points. Do NOT reduce or add bullets.")

    constraints_text = "\n".join(f"{i+1}. {c}" for i, c in enumerate(constraints))

    template = """Rewrite this resume role to better match the job description.

STRICT RULES:
{constraints}

Role: {title} at {company} ({start_date} to {end_date})
Bullets:
{responsibilities}

Job Description:
{jd}

Return JSON: {{"title": "...", "company": "...", "start_date": "...", "end_date": "...", "responsibilities": [...]}}
Only JSON, nothing else."""

    variables = {
        "constraints": constraints_text,
        "title": role["title"], "company": role["company"],
        "start_date": role.get("start_date", ""), "end_date": role.get("end_date", "Present"),
        "responsibilities": "\n".join(f"- {r}" for r in role["responsibilities"]),
        "jd": jd_text,
    }
    return PromptTemplate.from_template(template), variables


def apply_post_hoc(role, result, layers):
    """Apply post-hoc defense actions (contamination revert, structural fix)."""
    if "L2" in layers:
        contam, _, _ = detect_role_contamination(role, result)
        if contam:
            result["responsibilities"] = role["responsibilities"]
            result["_reverted"] = True
    return result


# ============================================================
# TEST DATA (same 25 resumes + 5 JDs from v2)
# ============================================================

def generate_test_resumes() -> List[Dict]:
    resumes = []

    resumes.append({"id": "r01", "industry": "technology", "professional_experience": [
        {"title": "Senior Data Engineer", "company": "CloudData Inc", "start_date": "2022-01", "end_date": "Present",
         "responsibilities": ["Designed ETL pipelines using AWS Glue and Athena", "Managed S3 data lake with Lake Formation", "Built real-time streaming with Kinesis", "Optimized Redshift queries reducing costs by 30%", "Implemented CloudWatch monitoring for pipelines"]},
        {"title": "Data Engineer", "company": "DataFlow Systems", "start_date": "2019-06", "end_date": "2021-12",
         "responsibilities": ["Built batch ETL jobs using Python and Apache Spark", "Managed PostgreSQL databases with complex SQL", "Created data quality checks with Great Expectations", "Deployed pipelines using Docker containers"]},
    ]})

    resumes.append({"id": "r02", "industry": "technology", "professional_experience": [
        {"title": "Senior ML Engineer", "company": "AI Solutions Corp", "start_date": "2021-03", "end_date": "Present",
         "responsibilities": ["Built ML pipelines using Vertex AI and AutoML", "Deployed models to Cloud Run with auto-scaling", "Used BigQuery for feature engineering", "Managed training on GKE with GPU pools", "Stored models in Cloud Storage", "Implemented A/B testing framework"]},
        {"title": "Data Scientist", "company": "Analytics Startup", "start_date": "2018-09", "end_date": "2021-02",
         "responsibilities": ["Developed models using scikit-learn and XGBoost", "Built dashboards with Tableau", "Performed statistical analysis with Python and R", "Created reporting pipelines with pandas"]},
    ]})

    resumes.append({"id": "r03", "industry": "technology", "professional_experience": [
        {"title": "Senior DevOps Engineer", "company": "Enterprise Solutions Ltd", "start_date": "2020-05", "end_date": "Present",
         "responsibilities": ["Managed AKS clusters with 200+ microservices", "Built CI/CD in Azure DevOps with automated testing", "Implemented IaC with ARM templates and Bicep", "Configured Azure Monitor and Application Insights", "Managed Azure AD with Key Vault secrets"]},
        {"title": "Systems Administrator", "company": "MidSize Corp", "start_date": "2017-01", "end_date": "2020-04",
         "responsibilities": ["Administered Windows Server and Active Directory", "Managed VMware ESXi virtualization", "Implemented backup and disaster recovery", "Automated tasks with PowerShell scripts"]},
    ]})

    resumes.append({"id": "r04", "industry": "technology", "professional_experience": [
        {"title": "Senior Software Engineer", "company": "WebApp Studios", "start_date": "2021-01", "end_date": "Present",
         "responsibilities": ["Built React frontend with TypeScript and Next.js", "Developed REST APIs using Node.js and Express", "Managed PostgreSQL with Prisma ORM", "Implemented CI/CD with GitHub Actions and Docker", "Wrote unit and integration tests with Jest"]},
        {"title": "Software Developer", "company": "Digital Agency", "start_date": "2018-06", "end_date": "2020-12",
         "responsibilities": ["Developed Django web applications with Python", "Built responsive UIs with HTML CSS JavaScript", "Managed MySQL databases and migration scripts", "Deployed apps on Linux servers with Nginx"]},
    ]})

    resumes.append({"id": "r05", "industry": "technology", "professional_experience": [
        {"title": "AI Engineer", "company": "LLM Startup", "start_date": "2023-06", "end_date": "Present",
         "responsibilities": ["Built RAG applications using LangChain and Bedrock", "Deployed LLM endpoints on SageMaker", "Implemented vector search with OpenSearch", "Created LLM evaluation frameworks"]},
        {"title": "Backend Engineer", "company": "SaaS Platform", "start_date": "2019-03", "end_date": "2023-05",
         "responsibilities": ["Built microservices with Java Spring Boot", "Deployed on ECS with CodePipeline CI/CD", "Implemented API Gateway with Lambda authorizers", "Managed DynamoDB for session management"]},
    ]})

    resumes.append({"id": "r06", "industry": "technology", "professional_experience": [
        {"title": "Software Intern", "company": "Tech Startup Inc", "start_date": "2024-06", "end_date": "2024-09",
         "responsibilities": ["Developed REST APIs using Python Flask", "Wrote unit tests with pytest", "Participated in code reviews"]},
    ]})

    resumes.append({"id": "r07", "industry": "consulting", "professional_experience": [
        {"title": "Principal Cloud Architect", "company": "Cloud Consulting Partners", "start_date": "2020-01", "end_date": "Present",
         "responsibilities": ["Designed multi-cloud architectures using AWS and Azure", "Implemented Azure AD federation with AWS IAM", "Built pipelines spanning AWS Glue and Azure Data Factory", "Managed Terraform for multi-cloud provisioning", "Led cloud migration assessments", "Configured CloudWatch and Azure Monitor"]},
    ]})

    resumes.append({"id": "r08", "industry": "finance", "professional_experience": [
        {"title": "Senior Data Analyst", "company": "National Bank Corp", "start_date": "2017-03", "end_date": "2019-12",
         "responsibilities": ["Built financial reports using SQL Server Reporting Services", "Created ETL packages in SSIS for data warehouse", "Developed dashboards in Tableau for executives", "Wrote T-SQL stored procedures for transformations", "Performed ad-hoc analysis with Excel and Python"]},
        {"title": "Data Analyst", "company": "Regional Insurance Co", "start_date": "2015-06", "end_date": "2017-02",
         "responsibilities": ["Generated monthly reports using SQL on Oracle", "Built pivot tables in Excel for claims analysis", "Automated report distribution with VBA macros"]},
    ]})

    resumes.append({"id": "r09", "industry": "finance", "professional_experience": [
        {"title": "Quantitative Analyst", "company": "Investment Management LLC", "start_date": "2016-01", "end_date": "2019-08",
         "responsibilities": ["Developed risk models using Python and NumPy", "Built Monte Carlo simulations for portfolio optimization", "Created automated trading signals with statistical analysis", "Maintained Oracle database for historical market data", "Generated regulatory reports using Crystal Reports"]},
    ]})

    resumes.append({"id": "r10", "industry": "finance", "professional_experience": [
        {"title": "Senior Engineer", "company": "Fintech Startup", "start_date": "2022-03", "end_date": "Present",
         "responsibilities": ["Built payment processing APIs using AWS Lambda and API Gateway", "Managed DynamoDB tables for transaction records", "Implemented fraud detection with SageMaker", "Ensured PCI DSS compliance in all services", "Created real-time alerting with SNS and CloudWatch"]},
        {"title": "Software Engineer", "company": "Banking Platform", "start_date": "2018-07", "end_date": "2022-02",
         "responsibilities": ["Developed Java microservices for core banking", "Managed PostgreSQL databases with complex transactions", "Built REST APIs consumed by mobile applications", "Implemented automated testing with JUnit and Mockito"]},
    ]})

    resumes.append({"id": "r11", "industry": "healthcare", "professional_experience": [
        {"title": "Clinical Data Engineer", "company": "Metro Hospital System", "start_date": "2016-08", "end_date": "2019-11",
         "responsibilities": ["Built HL7 integration pipelines for EHR systems", "Managed Oracle database for clinical data warehouse", "Created Informatica ETL for patient data consolidation", "Developed SSRS reports for clinical quality metrics", "Ensured HIPAA compliance in data handling"]},
        {"title": "Database Administrator", "company": "Community Health Network", "start_date": "2014-01", "end_date": "2016-07",
         "responsibilities": ["Administered SQL Server for patient management", "Implemented database backup and recovery", "Optimized query performance for reporting"]},
    ]})

    resumes.append({"id": "r12", "industry": "healthcare", "professional_experience": [
        {"title": "Health Data Platform Lead", "company": "Digital Health Corp", "start_date": "2021-06", "end_date": "Present",
         "responsibilities": ["Built HIPAA-compliant data lake on S3 with encryption", "Processed patient records using AWS Glue and Athena", "Deployed ML models on SageMaker for readmission prediction", "Managed RDS PostgreSQL for clinical applications", "Implemented CloudTrail auditing for compliance"]},
    ]})

    resumes.append({"id": "r13", "industry": "manufacturing", "professional_experience": [
        {"title": "Senior IT Analyst", "company": "Global Manufacturing Co", "start_date": "2016-04", "end_date": "2020-08",
         "responsibilities": ["Managed SAP ERP integration with manufacturing systems", "Built reports using SQL Server and Crystal Reports", "Developed Python scripts for production data analysis", "Administered on-premise Windows Server infrastructure", "Implemented monitoring with Nagios and Grafana"]},
        {"title": "IT Support Specialist", "company": "Small Factory Inc", "start_date": "2013-09", "end_date": "2016-03",
         "responsibilities": ["Provided desktop and server support for 200+ users", "Managed Active Directory and Group Policy", "Maintained network infrastructure"]},
    ]})

    resumes.append({"id": "r14", "industry": "manufacturing", "professional_experience": [
        {"title": "IoT Platform Engineer", "company": "Smart Factory Systems", "start_date": "2020-11", "end_date": "Present",
         "responsibilities": ["Built IoT data ingestion with AWS IoT Core and Kinesis", "Processed sensor data with Lambda and stored in DynamoDB", "Created real-time dashboards with Grafana on ECS", "Implemented edge computing with Greengrass", "Managed time-series data in Timestream"]},
    ]})

    resumes.append({"id": "r15", "industry": "retail", "professional_experience": [
        {"title": "Senior Backend Engineer", "company": "E-Shop Platform", "start_date": "2020-02", "end_date": "Present",
         "responsibilities": ["Built product catalog API using Lambda and DynamoDB", "Implemented search with OpenSearch Service", "Managed order processing with SQS and Step Functions", "Deployed microservices on ECS Fargate", "Implemented A/B testing for recommendation engine", "Monitored with CloudWatch and X-Ray tracing"]},
        {"title": "Backend Developer", "company": "Retail Startup", "start_date": "2017-05", "end_date": "2020-01",
         "responsibilities": ["Developed Python Django e-commerce application", "Managed MySQL database with Redis caching layer", "Built payment gateway integrations with Stripe API", "Deployed on DigitalOcean droplets with Nginx"]},
    ]})

    resumes.append({"id": "r16", "industry": "education", "professional_experience": [
        {"title": "Platform Engineer", "company": "EdTech Learning Inc", "start_date": "2021-09", "end_date": "Present",
         "responsibilities": ["Built learning analytics pipeline with BigQuery and Dataflow", "Deployed student-facing APIs on Cloud Run", "Managed content storage in Cloud Storage with CDN", "Implemented Firebase authentication for user management", "Created Pub/Sub event system for real-time notifications"]},
    ]})

    resumes.append({"id": "r17", "industry": "energy", "professional_experience": [
        {"title": "Data Analyst", "company": "Regional Power Utility", "start_date": "2015-03", "end_date": "2018-12",
         "responsibilities": ["Built energy consumption reports using SAS and SQL", "Managed Teradata data warehouse for smart meter data", "Created Tableau dashboards for load forecasting", "Developed Python scripts for anomaly detection", "Maintained ETL processes with Informatica"]},
        {"title": "Junior Analyst", "company": "Energy Consultants LLC", "start_date": "2013-06", "end_date": "2015-02",
         "responsibilities": ["Performed data entry and validation for utility clients", "Generated Excel reports for regulatory submissions", "Assisted with database queries on Oracle"]},
    ]})

    resumes.append({"id": "r18", "industry": "government", "professional_experience": [
        {"title": "IT Specialist", "company": "State Department of Revenue", "start_date": "2016-09", "end_date": "2021-06",
         "responsibilities": ["Managed on-premise Oracle databases for tax processing", "Built SSIS ETL packages for data warehouse loading", "Developed SSRS reports for departmental analytics", "Implemented security controls per NIST frameworks", "Administered Windows Server 2016 infrastructure", "Created PowerShell automation for routine maintenance"]},
    ]})

    resumes.append({"id": "r19", "industry": "media", "professional_experience": [
        {"title": "Streaming Platform Engineer", "company": "MediaStream Corp", "start_date": "2021-01", "end_date": "Present",
         "responsibilities": ["Built video transcoding pipeline with Lambda and MediaConvert", "Managed content delivery through CloudFront CDN", "Stored media assets in S3 with lifecycle policies", "Implemented DRM with KMS encryption", "Built recommendation API with DynamoDB and Lambda"]},
        {"title": "Software Developer", "company": "Content Platform", "start_date": "2018-04", "end_date": "2020-12",
         "responsibilities": ["Developed content management system with Python Django", "Built search functionality with Elasticsearch", "Managed PostgreSQL databases for metadata storage", "Deployed services with Docker Compose on Linux"]},
    ]})

    resumes.append({"id": "r20", "industry": "logistics", "professional_experience": [
        {"title": "Supply Chain Data Engineer", "company": "Global Logistics Corp", "start_date": "2020-07", "end_date": "Present",
         "responsibilities": ["Built shipment tracking pipeline with Azure Data Factory", "Managed Azure SQL Database for warehouse operations", "Created Power BI dashboards for supply chain KPIs", "Implemented Azure Functions for real-time alerts", "Stored documents in Blob Storage with lifecycle management"]},
        {"title": "Business Analyst", "company": "Shipping Solutions Inc", "start_date": "2017-02", "end_date": "2020-06",
         "responsibilities": ["Analyzed shipment data using SQL and Excel", "Built operational reports with Crystal Reports", "Developed VBA tools for inventory forecasting", "Managed Microsoft Access databases for tracking"]},
    ]})

    resumes.append({"id": "r21", "industry": "telecom", "professional_experience": [
        {"title": "Data Platform Engineer", "company": "TeleCom Networks", "start_date": "2022-04", "end_date": "Present",
         "responsibilities": ["Built call detail record processing with Kafka and Spark on EMR", "Managed S3 data lake for network telemetry data", "Created Athena queries for ad-hoc network analysis", "Deployed ML anomaly detection on SageMaker"]},
        {"title": "Network Data Analyst", "company": "Regional Telecom", "start_date": "2017-08", "end_date": "2022-03",
         "responsibilities": ["Analyzed network performance using SQL and Python", "Built Hadoop-based processing for CDR records", "Created Tableau dashboards for NOC operations", "Managed Oracle database for billing data", "Automated report generation with Python scripts"]},
    ]})

    resumes.append({"id": "r22", "industry": "insurance", "professional_experience": [
        {"title": "Actuarial Analyst", "company": "National Insurance Group", "start_date": "2014-09", "end_date": "2018-05",
         "responsibilities": ["Built actuarial models using SAS and R", "Managed claims database on SQL Server", "Developed automated pricing tools with Excel VBA", "Created loss reserve reports for regulatory filings", "Performed statistical analysis on policy data"]},
    ]})

    resumes.append({"id": "r23", "industry": "real_estate", "professional_experience": [
        {"title": "Backend Engineer", "company": "PropTech Solutions", "start_date": "2021-05", "end_date": "Present",
         "responsibilities": ["Built property search API with Cloud Run and Firestore", "Processed listing images with Vision AI for auto-tagging", "Managed BigQuery data warehouse for market analytics", "Implemented Pub/Sub for real-time listing notifications"]},
        {"title": "Web Developer", "company": "Real Estate Agency", "start_date": "2018-10", "end_date": "2021-04",
         "responsibilities": ["Developed WordPress sites for property listings", "Built custom PHP plugins for MLS integration", "Managed MySQL databases for listing data"]},
    ]})

    resumes.append({"id": "r24", "industry": "cybersecurity", "professional_experience": [
        {"title": "Cloud Security Engineer", "company": "SecureCloud Inc", "start_date": "2022-01", "end_date": "Present",
         "responsibilities": ["Implemented GuardDuty and Security Hub across 50 AWS accounts", "Built automated remediation with Lambda and EventBridge", "Managed IAM policies with least-privilege enforcement", "Configured WAF rules for web application protection", "Created compliance dashboards with Athena and QuickSight"]},
        {"title": "Security Analyst", "company": "CyberDefense Corp", "start_date": "2018-06", "end_date": "2021-12",
         "responsibilities": ["Performed vulnerability assessments using Nessus and Qualys", "Managed SIEM platform with Splunk for threat detection", "Conducted incident response and forensic analysis", "Developed security automation scripts with Python", "Maintained compliance with SOC 2 and ISO 27001"]},
    ]})

    resumes.append({"id": "r25", "industry": "nonprofit", "professional_experience": [
        {"title": "Data Analyst", "company": "Community Impact Foundation", "start_date": "2019-01", "end_date": "2023-06",
         "responsibilities": ["Built donor tracking reports using SQL and Excel", "Created impact measurement dashboards with Tableau", "Developed Python scripts for grant data analysis", "Managed PostgreSQL database for program outcomes", "Automated email campaigns with Mailchimp integrations"]},
        {"title": "Program Coordinator", "company": "Youth Services Org", "start_date": "2016-05", "end_date": "2018-12",
         "responsibilities": ["Tracked program metrics in Excel spreadsheets", "Generated quarterly reports for board presentations", "Coordinated data collection across 5 program sites"]},
    ]})

    return resumes


def generate_job_descriptions() -> List[Dict]:
    return [
        {"id": "jd01_multicloud_ai", "description": "Senior Data Engineer - Multi-Cloud AI Platform. Requirements: 5+ years experience. Must have AWS (Glue, Athena, S3, Redshift) AND Azure (Data Factory, Synapse, Cosmos DB). Experience with LLM-based data quality using LangChain and RAG pipelines. Vector database experience (Pinecone, ChromaDB). Terraform, Kubernetes, CI/CD. Python, Spark, Kafka."},
        {"id": "jd02_gcp_ml", "description": "ML Engineer - Google Cloud Platform. Requirements: 3+ years ML on GCP. Vertex AI, BigQuery, GKE, Cloud Run. Experience with LLM fine-tuning, prompt engineering, and RAG systems. TensorFlow/PyTorch. A/B testing. MLOps with Kubeflow or Vertex Pipelines."},
        {"id": "jd03_aws_fullstack", "description": "Full Stack Engineer - AWS. Requirements: React/TypeScript frontend, Python/Node.js backend. AWS services: Lambda, DynamoDB, S3, API Gateway, ECS. Experience with generative AI integration, LLM APIs, and vector search. CI/CD, Docker, infrastructure as code."},
        {"id": "jd04_azure_data", "description": "Senior Data Analyst - Azure Cloud. Requirements: Azure Synapse, Data Factory, Power BI, Azure SQL. Advanced SQL, Python, statistical analysis. Experience with Azure Machine Learning for predictive analytics. Knowledge of data governance and Azure Purview. Databricks experience preferred."},
        {"id": "jd05_generic_senior", "description": "Senior Software Engineer. Requirements: 5+ years software development. Python, Java, or Go. Cloud experience (AWS, GCP, or Azure). Microservices architecture. Database design (SQL and NoSQL). CI/CD pipelines. Docker/Kubernetes. Experience with AI/ML integration and modern data tools."},
    ]


# ============================================================
# EXPERIMENT RUNNER
# ============================================================

def run_single_experiment(
    resumes: List[Dict],
    jds: List[Dict],
    model_key: str,
    config_name: str,
    temperature: float,
    max_retries: int = 2,
) -> Dict:
    """Run one experiment: model × config × temperature."""
    from langchain_core.output_parsers import JsonOutputParser

    llm = get_llm(model_key, temperature)
    layers = CONFIGS[config_name]["layers"]

    per_resume = defaultdict(lambda: {"H1": 0, "H2": 0, "H3": 0, "H4": 0, "roles": 0, "errors": 0})
    details = []
    total_latency = 0.0
    n_calls = 0

    assignments = [(r, jds[i % len(jds)]) for i, r in enumerate(resumes)]

    for resume, jd in assignments:
        rid = resume["id"]
        for role_idx, role in enumerate(resume["professional_experience"]):
            detail = {
                "resume_id": rid, "role_idx": role_idx,
                "title": role["title"], "period": f"{role.get('start_date','')}-{role.get('end_date','')}",
            }

            for attempt in range(max_retries + 1):
                try:
                    prompt_tmpl, variables = build_prompt(role, jd["description"], config_name, layers)
                    chain = prompt_tmpl | llm | JsonOutputParser()

                    t0 = time.time()
                    result = chain.invoke(variables)
                    latency = time.time() - t0
                    total_latency += latency
                    n_calls += 1

                    # Apply post-hoc defenses
                    result = apply_post_hoc(role, result, layers)

                    # Detect hallucinations (always, regardless of config)
                    halls = detect_all(role, result)
                    for k in ["H1", "H2", "H3", "H4"]:
                        cnt = len(halls[k])
                        per_resume[rid][k] += cnt
                    per_resume[rid]["roles"] += 1

                    detail["halls"] = {k: v for k, v in halls.items() if v}
                    detail["latency"] = round(latency, 2)
                    detail["reverted"] = result.get("_reverted", False)
                    break

                except Exception as e:
                    if attempt == max_retries:
                        per_resume[rid]["errors"] += 1
                        detail["error"] = str(e)[:150]
                    else:
                        time.sleep(1)

            details.append(detail)
            time.sleep(0.3)  # rate limit

    # Aggregate statistics
    resume_rates = []
    total = {"H1": 0, "H2": 0, "H3": 0, "H4": 0, "total": 0, "errors": 0}
    for rid, counts in per_resume.items():
        h = counts["H1"] + counts["H2"] + counts["H3"] + counts["H4"]
        resume_rates.append(h)
        for k in ["H1", "H2", "H3", "H4"]:
            total[k] += counts[k]
        total["total"] += h
        total["errors"] += counts["errors"]

    n = len(resumes)
    mean_hr = sum(resume_rates) / n if n else 0
    std_hr = math.sqrt(sum((r - mean_hr) ** 2 for r in resume_rates) / n) if n else 0
    ci_95 = 1.96 * std_hr / math.sqrt(n) if n else 0

    return {
        "model": model_key,
        "config": config_name,
        "temperature": temperature,
        "n_resumes": n,
        "total_halls": total,
        "mean_hr": round(mean_hr, 4),
        "std_hr": round(std_hr, 4),
        "ci_95": round(ci_95, 4),
        "resume_rates": resume_rates,
        "mean_latency": round(total_latency / n_calls, 3) if n_calls else 0,
        "n_calls": n_calls,
        "details": details,
    }


# ============================================================
# EXPERIMENT PLANS
# ============================================================

def plan_experiments() -> List[Dict]:
    """Generate the full experiment plan."""
    experiments = []

    # Experiment 1: Ablation study (all configs × primary model)
    for config in CONFIGS:
        experiments.append({
            "name": f"ablation_{config}",
            "model": "gpt-4.1-nano",
            "config": config,
            "temperature": 0.0,
        })

    # Experiment 2: Multi-model (baseline + full × 3 models)
    for model in ["gpt-4o-mini", "llama-3.1-8b"]:
        for config in ["baseline", "full"]:
            experiments.append({
                "name": f"model_{model}_{config}",
                "model": model,
                "config": config,
                "temperature": 0.0,
            })

    # Experiment 3: Temperature sensitivity (baseline + full × 4 temps × primary model)
    for temp in [0.3, 0.7, 1.0]:
        for config in ["baseline", "full"]:
            experiments.append({
                "name": f"temp_{temp}_{config}",
                "model": "gpt-4.1-nano",
                "config": config,
                "temperature": temp,
            })

    return experiments


# ============================================================
# MAIN
# ============================================================

def main():
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = Path(__file__).parent / "results_comprehensive"
    outdir.mkdir(exist_ok=True)

    print("=" * 70)
    print(f"COMPREHENSIVE EVALUATION — {ts}")
    print("Ablation + Multi-Model + Temperature Sensitivity")
    print("=" * 70)

    resumes = generate_test_resumes()
    jds = generate_job_descriptions()
    experiments = plan_experiments()

    print(f"\nDataset: {len(resumes)} resumes, {sum(len(r['professional_experience']) for r in resumes)} roles")
    print(f"Experiments planned: {len(experiments)}")
    print()

    all_results = {}

    for i, exp in enumerate(experiments):
        label = exp["name"]
        print(f"\n[{i+1}/{len(experiments)}] {label}: model={exp['model']}, config={exp['config']}, temp={exp['temperature']}")

        result = run_single_experiment(
            resumes, jds,
            model_key=exp["model"],
            config_name=exp["config"],
            temperature=exp["temperature"],
        )

        halls = result["total_halls"]
        print(f"  → HR={result['mean_hr']:.2f} ± {result['std_hr']:.2f} (95% CI: ±{result['ci_95']:.2f})")
        print(f"    H1={halls['H1']} H2={halls['H2']} H3={halls['H3']} H4={halls['H4']} total={halls['total']} errors={halls['errors']}")
        print(f"    Latency: {result['mean_latency']:.2f}s/role, {result['n_calls']} calls")

        # Save individual result
        with open(outdir / f"{label}.json", "w") as f:
            json.dump(result, f, indent=2)

        all_results[label] = {k: v for k, v in result.items() if k != "details"}

    # Save summary
    summary_path = outdir / f"summary_{ts}.json"
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)

    # Print formatted tables
    print("\n" + "=" * 70)
    print("TABLE 1: ABLATION STUDY (GPT-4.1-nano, temp=0)")
    print("=" * 70)
    print(f"{'Config':<20} {'HR':>6} {'±σ':>6} {'95%CI':>7} {'H1':>4} {'H2':>4} {'H3':>4} {'H4':>4} {'Total':>6}")
    print("-" * 70)
    for config in CONFIGS:
        key = f"ablation_{config}"
        if key in all_results:
            r = all_results[key]
            h = r["total_halls"]
            print(f"{config:<20} {r['mean_hr']:>6.2f} {r['std_hr']:>6.2f} {r['ci_95']:>7.2f} {h['H1']:>4} {h['H2']:>4} {h['H3']:>4} {h['H4']:>4} {h['total']:>6}")

    print("\n" + "=" * 70)
    print("TABLE 2: MULTI-MODEL (baseline vs full, temp=0)")
    print("=" * 70)
    print(f"{'Model':<20} {'Config':<15} {'HR':>6} {'±σ':>6} {'H1':>4} {'H2':>4} {'H3':>4} {'H4':>4} {'Total':>6}")
    print("-" * 70)
    for model in ["gpt-4.1-nano", "gpt-4o-mini", "llama-3.1-8b"]:
        for config in ["baseline", "full"]:
            if model == "gpt-4.1-nano":
                key = f"ablation_{config}"
            else:
                key = f"model_{model}_{config}"
            if key in all_results:
                r = all_results[key]
                h = r["total_halls"]
                print(f"{model:<20} {config:<15} {r['mean_hr']:>6.2f} {r['std_hr']:>6.2f} {h['H1']:>4} {h['H2']:>4} {h['H3']:>4} {h['H4']:>4} {h['total']:>6}")

    print("\n" + "=" * 70)
    print("TABLE 3: TEMPERATURE SENSITIVITY (GPT-4.1-nano)")
    print("=" * 70)
    print(f"{'Temp':<6} {'Config':<15} {'HR':>6} {'±σ':>6} {'H1':>4} {'H2':>4} {'H3':>4} {'H4':>4} {'Total':>6}")
    print("-" * 70)
    for temp in [0.0, 0.3, 0.7, 1.0]:
        for config in ["baseline", "full"]:
            if temp == 0.0:
                key = f"ablation_{config}"
            else:
                key = f"temp_{temp}_{config}"
            if key in all_results:
                r = all_results[key]
                h = r["total_halls"]
                print(f"{temp:<6.1f} {config:<15} {r['mean_hr']:>6.2f} {r['std_hr']:>6.2f} {h['H1']:>4} {h['H2']:>4} {h['H3']:>4} {h['H4']:>4} {h['total']:>6}")

    print(f"\n  Full results: {outdir}")
    print(f"  Summary: {summary_path}")


if __name__ == "__main__":
    main()
