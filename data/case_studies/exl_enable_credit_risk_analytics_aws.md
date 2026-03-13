# Enable Credit Risk Analytics: D&A in the Cloud with AWS Migration

**Type:** Case Study  
**Company:** EXL  
**Client:** UK Financial Services Provider

---

## Overview

EXL helped a UK financial services provider transform its credit risk analytics function through successful migration onto the cloud using Amazon Web Services (AWS), resulting in a fivefold performance increase in data extraction, processing, and report creation.

---

## Challenge

EXL's client, a UK financial services provider, knew its credit risk analytics function needed to optimize its costs, balance growth and risk, and compete with digital-first organizations. The best way of accomplishing this was by adopting new cloud technologies that could host its data and analytics while lowering expenses, improving performance, enhancing security, and providing scalability. This robust, controlled data platform would then act as a single source of truth for the client's business and analytics needs.

Standing in the way of realizing this goal were several obstacles:

- Legacy platforms created issues including poor data latency
- Slow query performance
- Frequent load errors

The presence of multiple complex data sources and numerous redundant data marts rendered data analysis inefficient and ineffective, while missing capabilities in data visualization and operational automation caused these processes to be time- and cost-intensive. Inconsistent data incorporated from years of past portfolio acquisitions created regulatory risks. A lack of shared metrics led to different teams returning different results from their analysis, leaving the business without the single source of truth it required.

---

## Solution

Leveraging its decades of finance domain expertise, deep analytics experience, and extensive suite of credit risk modeling and strategic solutions, EXL set out to accomplish the client's business goals.

### Data Platform

EXL implemented a data platform utilizing Amazon Web Services for cloud-based data storage to ensure scalability. A Collibra-powered data dictionary organized and catalogued the metadata, as well as provided standard definitions of key performance indicators. Additionally, all payment card data held within the database was tokenized to ensure payment card industry (PCI) compliance.

Data from multiple external sources was cleaned and placed into a data lake. User access to this data was based on a subscription-based governance model in compliance with all relevant PCI and personally identifiable information (PII) regulations. Additionally, a modeled data mart allowed this data to be pre-processed, defined, and analyzed as needed.

### Open Source Tooling

An extensive suite of tools enhanced the solution:

- **Data extraction and processing:** Dremio and DG
- **Data analytics:** Jupyter, Python, and PC
- **Automation:** Airflow-based automation
- **Presentation:** Tableau data visualization

### Solution Architecture Overview

| Layer | Component | Description |
|-------|-----------|-------------|
| Data Platform | Cloud based data storage | AWS cloud based scalability |
| Data Platform | Data Dictionary | Collibra cataloguing tool to hold metadata and standard definitions of key KPI metrics |
| Data Platform | PCI Compliance | Card tokenization |
| Presentation Layer | Modelled Data Mart | Pre-processed, defined flags and calculations |
| Presentation Layer | Un-Modelled Data Lake | Cleansed data from multiple external sources |
| Presentation Layer | Access Control | Governed subscription based access to different areas (Non PII/PCI, PII, PCI) |
| Open Source Tooling | Data extraction & processing | Dremio, DG |
| Open Source Tooling | Data analytics | Jupyter, Python, PC |
| Open Source Tooling | Automation | Apache Airflow |
| Open Source Tooling | Presentation | Tableau |

---

## Outcome

The successful cloud migration enhanced the effectiveness of the client's credit risk function while lowering its costs.

- Open-source applications and a "pay only what you use" data platform resulted in decreased spending
- Powerful data and analytics interventions provided **up to a fivefold (5x) performance increase** in data extraction, processing, and report creation
- Intuitive tools and self-service dashboards minimized the complexity of the system, supported a broad range of skill levels, and improved overall user experience
- Provided the client with the single source of truth they needed
- Positioned the client to better balance risk and growth in the future

---

## About EXL

EXL (NASDAQ: EXLS) is a global analytics and digital solutions company that partners with clients to improve business outcomes and unlock growth. Bringing together deep domain expertise with robust data, powerful analytics, cloud, and AI, EXL creates agile, scalable solutions and executes complex operations for the world's leading corporations in industries including insurance, healthcare, banking and financial services, media, and retail, among others. Headquartered in New York, the team is over 33,000 strong, with more than 50 offices spanning six continents.

**Headquarters:** 320 Park Avenue, 29th Floor, New York, New York 10022  
**Website:** exlservice.com
