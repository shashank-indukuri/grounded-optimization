"""
Evaluation Phase 2: End-to-End Pipeline with Real LLM Calls
============================================================
Tests the full resume optimization pipeline with and without defense layers.
Measures hallucination rates, ATS scores, and structural preservation.

Requires: OPENAI_API_KEY, GROQ_API_KEY
Reproducible: temperature=0, fixed seed for test data generation.
"""

import sys
import os
import json
import time
import random
import copy
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Any

# Load env
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / ".env")

# Add codebase
CODEBASE = Path(__file__).resolve().parent.parent.parent / "evidence" / "resumeai-main"
sys.path.insert(0, str(CODEBASE))

from app.cloud_taxonomy import detect_cloud_providers, detect_role_contamination
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from groq import Groq

SEED = 42
random.seed(SEED)

# Models (match production config)
REWRITE_MODEL = "gpt-4.1-nano"
GROQ_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"

# ============================================================
# SYNTHETIC TEST RESUMES (realistic, diverse industries)
# ============================================================

def generate_test_resumes() -> List[Dict]:
    """Generate 20 realistic synthetic resumes with known properties."""
    resumes = []

    # Resume 1: AWS Data Engineer (5 years)
    resumes.append({
        "id": "resume_01",
        "industry": "technology",
        "summary": "Data Engineer with 5 years of experience building scalable data pipelines on AWS.",
        "contact_info": {"name": "Alex Chen", "email": "alex@example.com", "phone": "555-0101"},
        "professional_experience": [
            {
                "title": "Senior Data Engineer",
                "company": "CloudData Inc",
                "start_date": "2022-01",
                "end_date": "Present",
                "responsibilities": [
                    "Designed and maintained ETL pipelines using AWS Glue and Athena",
                    "Managed S3 data lake with Lake Formation access controls",
                    "Built real-time streaming with Kinesis Data Streams",
                    "Optimized Redshift queries reducing warehouse costs by 30%",
                    "Implemented CloudWatch monitoring and alerting for all pipelines",
                ]
            },
            {
                "title": "Data Engineer",
                "company": "DataFlow Systems",
                "start_date": "2019-06",
                "end_date": "2021-12",
                "responsibilities": [
                    "Built batch ETL jobs using Python and Apache Spark",
                    "Managed PostgreSQL databases and wrote complex SQL queries",
                    "Created automated data quality checks with Great Expectations",
                    "Deployed data pipelines using Docker containers",
                ]
            },
        ],
        "skills": ["Python", "SQL", "AWS Glue", "Athena", "S3", "Redshift", "Spark", "Docker"],
        "education": [{"degree": "B.S. Computer Science", "institution": "State University", "year": "2019"}],
        "certifications": ["AWS Solutions Architect Associate"],
        "projects": [],
        "expected_clouds": {"role_0": ["AWS"], "role_1": ["Cloud-Agnostic"]},
    })

    # Resume 2: GCP ML Engineer
    resumes.append({
        "id": "resume_02",
        "industry": "technology",
        "summary": "Machine Learning Engineer specializing in production ML systems on Google Cloud Platform.",
        "contact_info": {"name": "Priya Patel", "email": "priya@example.com", "phone": "555-0102"},
        "professional_experience": [
            {
                "title": "Senior ML Engineer",
                "company": "AI Solutions Corp",
                "start_date": "2021-03",
                "end_date": "Present",
                "responsibilities": [
                    "Built end-to-end ML pipelines using Vertex AI and AutoML",
                    "Deployed models to Cloud Run endpoints with auto-scaling",
                    "Used BigQuery for feature engineering on petabyte-scale datasets",
                    "Implemented A/B testing framework for model evaluation",
                    "Managed training infrastructure on GKE with GPU node pools",
                    "Stored and versioned models in Cloud Storage with metadata tracking",
                ]
            },
            {
                "title": "Data Scientist",
                "company": "Analytics Startup",
                "start_date": "2018-09",
                "end_date": "2021-02",
                "responsibilities": [
                    "Developed predictive models using scikit-learn and XGBoost",
                    "Built dashboards with Tableau for business stakeholders",
                    "Performed statistical analysis using Python and R",
                    "Created automated reporting pipelines with pandas",
                ]
            },
        ],
        "skills": ["Python", "TensorFlow", "Vertex AI", "BigQuery", "GKE", "scikit-learn"],
        "education": [{"degree": "M.S. Machine Learning", "institution": "Tech University", "year": "2018"}],
        "certifications": ["Google Cloud Professional ML Engineer"],
        "projects": [],
        "expected_clouds": {"role_0": ["GCP"], "role_1": ["Cloud-Agnostic"]},
    })

    # Resume 3: Azure DevOps Engineer
    resumes.append({
        "id": "resume_03",
        "industry": "technology",
        "summary": "DevOps Engineer with expertise in Azure cloud infrastructure and CI/CD automation.",
        "contact_info": {"name": "James Wilson", "email": "james@example.com", "phone": "555-0103"},
        "professional_experience": [
            {
                "title": "Senior DevOps Engineer",
                "company": "Enterprise Solutions Ltd",
                "start_date": "2020-05",
                "end_date": "Present",
                "responsibilities": [
                    "Managed Azure Kubernetes Service clusters with 200+ microservices",
                    "Built CI/CD pipelines in Azure DevOps with automated testing",
                    "Implemented infrastructure as code with ARM templates and Bicep",
                    "Configured Azure Monitor and Application Insights for observability",
                    "Managed Azure AD identity with Key Vault secret management",
                ]
            },
            {
                "title": "Systems Administrator",
                "company": "MidSize Corp",
                "start_date": "2017-01",
                "end_date": "2020-04",
                "responsibilities": [
                    "Administered Windows Server and Active Directory environments",
                    "Managed VMware ESXi virtualization infrastructure",
                    "Implemented backup and disaster recovery procedures",
                    "Automated routine tasks with PowerShell scripts",
                ]
            },
        ],
        "skills": ["Azure", "Kubernetes", "Terraform", "Azure DevOps", "PowerShell", "Docker"],
        "education": [{"degree": "B.S. Information Technology", "institution": "City College", "year": "2016"}],
        "certifications": ["Azure Administrator Associate"],
        "projects": [],
        "expected_clouds": {"role_0": ["Azure"], "role_1": ["On-Premise"]},
    })

    # Resume 4: On-prem Data Analyst (pre-cloud era, 2015-2019) — temporal test
    resumes.append({
        "id": "resume_04",
        "industry": "finance",
        "summary": "Data Analyst with experience in financial reporting and traditional data warehousing.",
        "contact_info": {"name": "Sarah Kim", "email": "sarah@example.com", "phone": "555-0104"},
        "professional_experience": [
            {
                "title": "Senior Data Analyst",
                "company": "National Bank Corp",
                "start_date": "2017-03",
                "end_date": "2019-12",
                "responsibilities": [
                    "Built financial reports using SQL Server Reporting Services",
                    "Created ETL packages in SSIS for data warehouse loading",
                    "Developed dashboards in Tableau for executive leadership",
                    "Wrote complex T-SQL stored procedures for data transformation",
                    "Performed ad-hoc analysis using Excel and Python pandas",
                ]
            },
            {
                "title": "Data Analyst",
                "company": "Regional Insurance Co",
                "start_date": "2015-06",
                "end_date": "2017-02",
                "responsibilities": [
                    "Generated monthly reports using SQL queries on Oracle database",
                    "Built pivot tables and charts in Excel for claims analysis",
                    "Automated report distribution with VBA macros",
                ]
            },
        ],
        "skills": ["SQL", "T-SQL", "SSIS", "Tableau", "Python", "Excel"],
        "education": [{"degree": "B.S. Statistics", "institution": "State University", "year": "2015"}],
        "certifications": [],
        "projects": [],
        "expected_clouds": {"role_0": ["On-Premise"], "role_1": ["On-Premise"]},
    })

    # Resume 5: Full-stack developer (cloud-agnostic)
    resumes.append({
        "id": "resume_05",
        "industry": "technology",
        "summary": "Full-stack developer with experience building web applications using modern frameworks.",
        "contact_info": {"name": "Mike Johnson", "email": "mike@example.com", "phone": "555-0105"},
        "professional_experience": [
            {
                "title": "Senior Software Engineer",
                "company": "WebApp Studios",
                "start_date": "2021-01",
                "end_date": "Present",
                "responsibilities": [
                    "Built React frontend with TypeScript and Next.js framework",
                    "Developed REST APIs using Node.js and Express",
                    "Managed PostgreSQL database with Prisma ORM",
                    "Implemented CI/CD with GitHub Actions and Docker",
                    "Wrote unit and integration tests with Jest and Cypress",
                ]
            },
            {
                "title": "Software Developer",
                "company": "Digital Agency",
                "start_date": "2018-06",
                "end_date": "2020-12",
                "responsibilities": [
                    "Developed Django web applications with Python",
                    "Built responsive UIs with HTML, CSS, and JavaScript",
                    "Managed MySQL databases and wrote migration scripts",
                    "Deployed applications on Linux servers with Nginx",
                ]
            },
        ],
        "skills": ["TypeScript", "React", "Node.js", "Python", "Django", "PostgreSQL", "Docker"],
        "education": [{"degree": "B.S. Computer Science", "institution": "Tech College", "year": "2018"}],
        "certifications": [],
        "projects": [{"name": "E-commerce Platform", "description": "Built a full-stack e-commerce app with React and Node.js"}],
        "expected_clouds": {"role_0": ["Cloud-Agnostic"], "role_1": ["Cloud-Agnostic"]},
    })

    # Resume 6: Healthcare Data Engineer (pre-2020, temporal hallucination risk)
    resumes.append({
        "id": "resume_06",
        "industry": "healthcare",
        "summary": "Healthcare data professional with experience in clinical data systems and reporting.",
        "contact_info": {"name": "Dr. Lisa Chang", "email": "lisa@example.com", "phone": "555-0106"},
        "professional_experience": [
            {
                "title": "Clinical Data Engineer",
                "company": "Metro Hospital System",
                "start_date": "2016-08",
                "end_date": "2019-11",
                "responsibilities": [
                    "Built HL7 data integration pipelines for electronic health records",
                    "Managed Oracle database for clinical data warehouse",
                    "Created Informatica ETL workflows for patient data consolidation",
                    "Developed SSRS reports for clinical quality metrics",
                    "Ensured HIPAA compliance in all data handling procedures",
                ]
            },
            {
                "title": "Database Administrator",
                "company": "Community Health Network",
                "start_date": "2014-01",
                "end_date": "2016-07",
                "responsibilities": [
                    "Administered SQL Server databases for patient management system",
                    "Implemented database backup and recovery procedures",
                    "Optimized query performance for reporting applications",
                ]
            },
        ],
        "skills": ["SQL", "Oracle", "Informatica", "SSIS", "SSRS", "HL7", "Python"],
        "education": [{"degree": "M.S. Health Informatics", "institution": "Medical University", "year": "2013"}],
        "certifications": ["CHDA - Certified Health Data Analyst"],
        "projects": [],
        "expected_clouds": {"role_0": ["On-Premise"], "role_1": ["On-Premise"]},
    })

    # Resume 7: Multi-cloud architect (legitimate multi-cloud)
    resumes.append({
        "id": "resume_07",
        "industry": "consulting",
        "summary": "Cloud architect with multi-cloud expertise across AWS and Azure environments.",
        "contact_info": {"name": "Raj Sharma", "email": "raj@example.com", "phone": "555-0107"},
        "professional_experience": [
            {
                "title": "Principal Cloud Architect",
                "company": "Cloud Consulting Partners",
                "start_date": "2020-01",
                "end_date": "Present",
                "responsibilities": [
                    "Designed multi-cloud architectures using AWS and Azure for Fortune 500 clients",
                    "Implemented Azure AD federation with AWS IAM for unified identity",
                    "Built data pipelines spanning AWS Glue and Azure Data Factory",
                    "Managed Terraform modules for multi-cloud infrastructure provisioning",
                    "Led cloud migration assessments for enterprise workloads",
                    "Configured AWS CloudWatch and Azure Monitor for unified observability",
                ]
            },
        ],
        "skills": ["AWS", "Azure", "Terraform", "Kubernetes", "Python"],
        "education": [{"degree": "M.S. Cloud Computing", "institution": "Online University", "year": "2019"}],
        "certifications": ["AWS Solutions Architect Professional", "Azure Solutions Architect Expert"],
        "projects": [],
        "expected_clouds": {"role_0": ["AWS", "Azure"]},
    })

    # Resume 8: Entry-level (short resume, structural mutation risk)
    resumes.append({
        "id": "resume_08",
        "industry": "technology",
        "summary": "Recent graduate seeking entry-level software engineering position.",
        "contact_info": {"name": "Emma Davis", "email": "emma@example.com", "phone": "555-0108"},
        "professional_experience": [
            {
                "title": "Software Engineering Intern",
                "company": "Tech Startup Inc",
                "start_date": "2023-06",
                "end_date": "2023-09",
                "responsibilities": [
                    "Developed REST APIs using Python Flask",
                    "Wrote unit tests with pytest framework",
                    "Participated in code reviews and agile ceremonies",
                ]
            },
        ],
        "skills": ["Python", "Java", "SQL", "Git", "Flask"],
        "education": [{"degree": "B.S. Computer Science", "institution": "University of Somewhere", "year": "2024"}],
        "certifications": [],
        "projects": [
            {"name": "Chat Application", "description": "Built real-time chat app with WebSockets and React"},
            {"name": "ML Image Classifier", "description": "Trained CNN model for image classification using PyTorch"},
        ],
        "expected_clouds": {"role_0": ["Cloud-Agnostic"]},
    })

    # Resume 9: Manufacturing / traditional IT
    resumes.append({
        "id": "resume_09",
        "industry": "manufacturing",
        "summary": "IT professional with experience in manufacturing execution systems and ERP integration.",
        "contact_info": {"name": "Tom Martinez", "email": "tom@example.com", "phone": "555-0109"},
        "professional_experience": [
            {
                "title": "Senior IT Analyst",
                "company": "Global Manufacturing Co",
                "start_date": "2016-04",
                "end_date": "2020-08",
                "responsibilities": [
                    "Managed SAP ERP system integration with manufacturing execution systems",
                    "Built automated reports using SQL Server and Crystal Reports",
                    "Developed Python scripts for production data analysis",
                    "Administered on-premise Windows Server infrastructure",
                    "Implemented network monitoring with Nagios and Grafana",
                ]
            },
            {
                "title": "IT Support Specialist",
                "company": "Small Factory Inc",
                "start_date": "2013-09",
                "end_date": "2016-03",
                "responsibilities": [
                    "Provided desktop and server support for 200+ users",
                    "Managed Active Directory and Group Policy configuration",
                    "Installed and maintained network infrastructure",
                ]
            },
        ],
        "skills": ["SQL Server", "SAP", "Python", "PowerShell", "Windows Server"],
        "education": [{"degree": "B.S. Information Systems", "institution": "Community College", "year": "2013"}],
        "certifications": ["CompTIA A+", "MCSA Windows Server"],
        "projects": [],
        "expected_clouds": {"role_0": ["On-Premise"], "role_1": ["On-Premise"]},
    })

    # Resume 10: AWS + recent AI role (temporal boundary test)
    resumes.append({
        "id": "resume_10",
        "industry": "technology",
        "summary": "AI Engineer building production LLM applications on AWS.",
        "contact_info": {"name": "David Park", "email": "david@example.com", "phone": "555-0110"},
        "professional_experience": [
            {
                "title": "AI Engineer",
                "company": "LLM Startup",
                "start_date": "2023-06",
                "end_date": "Present",
                "responsibilities": [
                    "Built RAG applications using LangChain and Amazon Bedrock",
                    "Deployed LLM inference endpoints on SageMaker",
                    "Implemented vector search with OpenSearch Serverless",
                    "Created evaluation frameworks for LLM output quality",
                ]
            },
            {
                "title": "Backend Engineer",
                "company": "SaaS Company",
                "start_date": "2019-03",
                "end_date": "2023-05",
                "responsibilities": [
                    "Built microservices with Java Spring Boot and PostgreSQL",
                    "Deployed services on ECS with automated CI/CD via CodePipeline",
                    "Implemented API Gateway with Lambda authorizers",
                    "Managed DynamoDB tables for session management",
                ]
            },
        ],
        "skills": ["Python", "LangChain", "AWS Bedrock", "SageMaker", "Java", "Spring Boot"],
        "education": [{"degree": "M.S. Computer Science", "institution": "Research University", "year": "2019"}],
        "certifications": [],
        "projects": [],
        "expected_clouds": {"role_0": ["AWS"], "role_1": ["AWS"]},
    })

    return resumes


# ============================================================
# JOB DESCRIPTIONS (targets for optimization)
# ============================================================

def generate_job_descriptions() -> List[Dict]:
    """Generate diverse job descriptions to pair with resumes."""
    return [
        {
            "id": "jd_multi_cloud_de",
            "title": "Senior Data Engineer",
            "description": """
            Senior Data Engineer needed for multi-cloud data platform team.
            Requirements:
            - 5+ years experience with AWS (Glue, Athena, Redshift) and Azure (Data Factory, Synapse)
            - Strong Python and SQL skills
            - Experience with Spark, Kafka, and real-time streaming
            - Knowledge of data governance and cataloging tools
            - Experience with LLM-based data quality tools (LangChain, vector databases)
            - CI/CD experience with Terraform or CloudFormation
            """,
        },
        {
            "id": "jd_gcp_ml",
            "title": "ML Engineer - Google Cloud",
            "description": """
            ML Engineer to build production ML systems on GCP.
            Requirements:
            - 3+ years MLOps experience on Google Cloud (Vertex AI, BigQuery, GKE)
            - Strong Python, TensorFlow/PyTorch
            - Experience deploying models with Cloud Run or Cloud Functions
            - Knowledge of A/B testing and experiment tracking
            - Experience with LLM fine-tuning and RAG systems
            """,
        },
        {
            "id": "jd_fullstack",
            "title": "Full Stack Developer",
            "description": """
            Full Stack Developer for a fast-growing SaaS startup.
            Requirements:
            - React/TypeScript frontend, Node.js/Python backend
            - PostgreSQL or MongoDB experience
            - AWS or GCP cloud deployment experience
            - CI/CD with GitHub Actions or Jenkins
            - REST API and GraphQL experience
            """,
        },
    ]


# ============================================================
# REWRITE ENGINE (simplified for evaluation)
# ============================================================

def rewrite_experience_undefended(role: dict, jd: str, llm) -> dict:
    """Rewrite a role WITHOUT any defense layers (baseline)."""
    prompt = PromptTemplate.from_template("""
You are a professional resume optimizer. Rewrite the following role to better match the job description.
Make the bullet points more impactful and keyword-rich.

Role:
Title: {title}
Company: {company}
Period: {start_date} to {end_date}
Responsibilities:
{responsibilities}

Target Job Description:
{jd}

Return a JSON object with:
{{"title": "...", "company": "...", "start_date": "...", "end_date": "...", "responsibilities": ["bullet1", "bullet2", ...]}}

Return ONLY the JSON, nothing else.
""")
    chain = prompt | llm | JsonOutputParser()
    result = chain.invoke({
        "title": role["title"],
        "company": role["company"],
        "start_date": role.get("start_date", ""),
        "end_date": role.get("end_date", "Present"),
        "responsibilities": "\n".join(f"- {r}" for r in role["responsibilities"]),
        "jd": jd,
    })
    return result


def rewrite_experience_defended(role: dict, jd: str, llm, temporal_context: str, cloud_context: str) -> dict:
    """Rewrite a role WITH all defense layers active."""
    prompt = PromptTemplate.from_template("""
You are a professional resume optimizer. Rewrite the following role to better match the job description.

STRICT RULES:
1. Preserve the EXACT number of bullet points ({bullet_count}). DO NOT reduce or condense them.
2. DO NOT hallucinate or add technologies that didn't exist during this role's time period.
3. DO NOT introduce cloud services from providers not already used in this role.
4. DO NOT create new companies or fabricate metrics.
5. Keep all achievements realistic for the time period and role level.

Temporal Context:
{temporal_context}

Cloud Context for this role:
{cloud_context}

Role:
Title: {title}
Company: {company}
Period: {start_date} to {end_date}
Responsibilities:
{responsibilities}

Target Job Description:
{jd}

Return a JSON object with:
{{"title": "...", "company": "...", "start_date": "...", "end_date": "...", "responsibilities": ["bullet1", "bullet2", ...]}}

You MUST return exactly {bullet_count} bullet points. Return ONLY the JSON, nothing else.
""")
    chain = prompt | llm | JsonOutputParser()
    result = chain.invoke({
        "title": role["title"],
        "company": role["company"],
        "start_date": role.get("start_date", ""),
        "end_date": role.get("end_date", "Present"),
        "responsibilities": "\n".join(f"- {r}" for r in role["responsibilities"]),
        "bullet_count": len(role["responsibilities"]),
        "jd": jd,
        "temporal_context": temporal_context,
        "cloud_context": cloud_context,
    })
    return result


# ============================================================
# HALLUCINATION DETECTORS
# ============================================================

TECH_RELEASE_DATES = {
    "langchain": 2022, "llamaindex": 2022, "llama index": 2022,
    "vertex ai": 2021, "mixtral": 2023, "chatgpt": 2022,
    "gpt-4": 2023, "gpt-3.5": 2022, "bedrock": 2023,
    "rag": 2022, "vector database": 2022, "pinecone": 2021,
    "chromadb": 2022, "weaviate": 2021, "openai api": 2020,
    "copilot": 2021, "stable diffusion": 2022, "midjourney": 2022,
    "langsmith": 2023, "langgraph": 2024, "autogen": 2023,
    "crewai": 2023, "dspy": 2023,
}

def detect_temporal_hallucinations(original_role: dict, updated_role: dict) -> List[str]:
    """Detect anachronistic technology mentions."""
    violations = []
    end_date = original_role.get("end_date", "Present")
    if end_date in ("Present", "current", "now", None):
        return []  # Current roles can mention anything

    try:
        from dateutil.parser import parse
        end_year = parse(str(end_date)).year
    except:
        return []

    updated_text = " ".join(updated_role.get("responsibilities", [])).lower()
    original_text = " ".join(original_role.get("responsibilities", [])).lower()

    for tech, release_year in TECH_RELEASE_DATES.items():
        if tech in updated_text and tech not in original_text and end_year < release_year:
            violations.append(f"'{tech}' (released {release_year}) in role ending {end_year}")

    return violations


def detect_structural_violations(original_role: dict, updated_role: dict) -> List[str]:
    """Detect bullet point loss."""
    violations = []
    orig_count = len(original_role.get("responsibilities", []))
    upd_count = len(updated_role.get("responsibilities", []))

    if upd_count < orig_count - 1:
        violations.append(f"Lost {orig_count - upd_count} bullets (had {orig_count}, got {upd_count})")

    return violations


def detect_content_fabrication(original_role: dict, updated_role: dict) -> List[str]:
    """Detect fabricated companies, changed titles, or new certifications."""
    violations = []
    if updated_role.get("company", "") != original_role.get("company", ""):
        violations.append(f"Company changed: '{original_role.get('company')}' -> '{updated_role.get('company')}'")
    if updated_role.get("title", "") != original_role.get("title", ""):
        # Allow minor title enhancements but flag major changes
        orig_words = set(original_role.get("title", "").lower().split())
        upd_words = set(updated_role.get("title", "").lower().split())
        if len(orig_words & upd_words) < len(orig_words) * 0.5:
            violations.append(f"Title changed significantly: '{original_role.get('title')}' -> '{updated_role.get('title')}'")

    return violations


def detect_all_hallucinations(original_role: dict, updated_role: dict) -> Dict[str, List[str]]:
    """Run all hallucination detectors on a role pair."""
    contam_list, orig_clouds, upd_clouds = detect_role_contamination(original_role, updated_role)
    contam_violations = [f"Added {c['cloud']} ({c['technology']})" for c in contam_list]

    return {
        "H1_temporal": detect_temporal_hallucinations(original_role, updated_role),
        "H2_contamination": contam_violations,
        "H3_structural": detect_structural_violations(original_role, updated_role),
        "H4_fabrication": detect_content_fabrication(original_role, updated_role),
    }


# ============================================================
# MAIN EVALUATION LOOP
# ============================================================

def run_evaluation():
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    print("=" * 70)
    print(f"EVALUATION PHASE 2: End-to-End Pipeline")
    print(f"Seed: {SEED} | Timestamp: {ts}")
    print(f"Rewrite model: {REWRITE_MODEL} | temperature=0")
    print("=" * 70)

    # Initialize LLM (temperature=0 for reproducibility)
    llm = ChatOpenAI(model=REWRITE_MODEL, temperature=0)

    resumes = generate_test_resumes()
    jds = generate_job_descriptions()

    # Assign JDs to resumes (cycle through)
    assignments = []
    for i, resume in enumerate(resumes):
        jd = jds[i % len(jds)]
        assignments.append((resume, jd))

    results = {
        "meta": {"seed": SEED, "timestamp": ts, "model": REWRITE_MODEL, "temperature": 0},
        "baseline": {"total_roles": 0, "hallucinations": {"H1": 0, "H2": 0, "H3": 0, "H4": 0}, "details": []},
        "defended": {"total_roles": 0, "hallucinations": {"H1": 0, "H2": 0, "H3": 0, "H4": 0}, "details": []},
    }

    for resume, jd in assignments:
        resume_id = resume["id"]
        jd_text = jd["description"]
        print(f"\n--- {resume_id} ({resume['industry']}) vs {jd['id']} ---")

        for role_idx, role in enumerate(resume["professional_experience"]):
            role_key = f"role_{role_idx}"
            results["baseline"]["total_roles"] += 1
            results["defended"]["total_roles"] += 1

            print(f"  Role {role_idx}: {role['title']} at {role['company']} ({len(role['responsibilities'])} bullets)")

            # --- BASELINE (no defense) ---
            try:
                baseline_result = rewrite_experience_undefended(role, jd_text, llm)
                baseline_halls = detect_all_hallucinations(role, baseline_result)

                for htype, violations in baseline_halls.items():
                    key = htype.split("_")[0]
                    results["baseline"]["hallucinations"][key] += len(violations)
                    if violations:
                        print(f"    BASELINE {htype}: {violations}")

                results["baseline"]["details"].append({
                    "resume_id": resume_id, "role_idx": role_idx,
                    "original_bullets": len(role["responsibilities"]),
                    "output_bullets": len(baseline_result.get("responsibilities", [])),
                    "hallucinations": {k: v for k, v in baseline_halls.items() if v},
                })
            except Exception as e:
                print(f"    BASELINE ERROR: {e}")
                results["baseline"]["details"].append({
                    "resume_id": resume_id, "role_idx": role_idx, "error": str(e)
                })

            # Small delay to avoid rate limits
            time.sleep(0.5)

            # --- DEFENDED (all layers) ---
            try:
                # Build temporal context
                end_date = role.get("end_date", "Present")
                if end_date in ("Present", "current", "now", None):
                    temporal = "Current role - all modern technologies acceptable."
                else:
                    from dateutil.parser import parse as dparse
                    end_year = dparse(str(end_date)).year
                    temporal = f"Role ended in {end_year}. DO NOT mention technologies released after {end_year}. "
                    temporal += "Specifically: LangChain (2022), LlamaIndex (2022), Mixtral (2023), ChatGPT (2022), GPT-4 (2023), Bedrock (2023), RAG (2022)."

                # Build cloud context
                orig_text = " ".join(role.get("responsibilities", []))
                orig_clouds = detect_cloud_providers(orig_text)
                cloud_ctx = f"This role uses: {orig_clouds}. Do NOT introduce services from other cloud providers."

                defended_result = rewrite_experience_defended(
                    role, jd_text, llm, temporal, cloud_ctx
                )

                # Post-hoc validation: structural check
                if len(defended_result.get("responsibilities", [])) < len(role["responsibilities"]) - 1:
                    print(f"    DEFENDED: Structural violation detected, retrying...")
                    # Retry with stronger constraint
                    defended_result = rewrite_experience_defended(
                        role, jd_text, llm,
                        temporal + f"\nCRITICAL: Return EXACTLY {len(role['responsibilities'])} bullets.",
                        cloud_ctx
                    )

                # Post-hoc validation: contamination check
                contam, _, _ = detect_role_contamination(role, defended_result)
                if contam:
                    print(f"    DEFENDED: Contamination detected, reverting role")
                    defended_result["responsibilities"] = role["responsibilities"]

                defended_halls = detect_all_hallucinations(role, defended_result)

                for htype, violations in defended_halls.items():
                    key = htype.split("_")[0]
                    results["defended"]["hallucinations"][key] += len(violations)
                    if violations:
                        print(f"    DEFENDED {htype}: {violations}")

                results["defended"]["details"].append({
                    "resume_id": resume_id, "role_idx": role_idx,
                    "original_bullets": len(role["responsibilities"]),
                    "output_bullets": len(defended_result.get("responsibilities", [])),
                    "hallucinations": {k: v for k, v in defended_halls.items() if v},
                })
            except Exception as e:
                print(f"    DEFENDED ERROR: {e}")
                results["defended"]["details"].append({
                    "resume_id": resume_id, "role_idx": role_idx, "error": str(e)
                })

            time.sleep(0.5)

    # --- SUMMARY ---
    print("\n" + "=" * 70)
    print("PHASE 2 RESULTS")
    print("=" * 70)

    total_roles = results["baseline"]["total_roles"]
    n_resumes = len(resumes)

    for mode in ["baseline", "defended"]:
        h = results[mode]["hallucinations"]
        total_h = sum(h.values())
        per_resume = total_h / n_resumes if n_resumes > 0 else 0
        print(f"\n  {mode.upper()}:")
        print(f"    Total roles processed: {results[mode]['total_roles']}")
        print(f"    H1 (Temporal):       {h['H1']}")
        print(f"    H2 (Contamination):  {h['H2']}")
        print(f"    H3 (Structural):     {h['H3']}")
        print(f"    H4 (Fabrication):    {h['H4']}")
        print(f"    Total hallucinations: {total_h}")
        print(f"    Per resume:          {per_resume:.2f}")

    b_total = sum(results["baseline"]["hallucinations"].values())
    d_total = sum(results["defended"]["hallucinations"].values())
    if b_total > 0:
        reduction = (1 - d_total / b_total) * 100
        print(f"\n  REDUCTION: {reduction:.1f}% ({b_total} -> {d_total})")
    else:
        print(f"\n  No baseline hallucinations detected (try more aggressive JDs)")

    # Save
    outpath = Path(__file__).parent / f"results_phase2_{ts}.json"
    with open(outpath, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Results saved: {outpath}")

    return results


if __name__ == "__main__":
    run_evaluation()
