---
title: TNSR.AI - Building a Scalable Media Enhancement Platform
description: "A deep dive into building a full-stack media enhancement platform using modern cloud technologies"
date: "July 27 2024"
demoURL: "https://tnsr.ai"
---

**What is TNSR.AI and why did I build it?**

TNSR.AI is a web-based platform that leverages deep learning models to enhance various types of media. The project was born from the need to make advanced AI-powered media enhancement accessible through a user-friendly interface, while ensuring scalability and performance.

![VISIT tnsr.ai](/projects/landingpage.jpg)

## Architecture Overview

The platform is built using a modern tech stack, designed with scalability and performance in mind. Here's a high-level overview of the system architecture:


### Frontend Implementation
The frontend is built using **Next.js** and **TailwindCSS**, focusing on a responsive and intuitive user experience. A key feature is real-time job status updates via WebSocket connections, allowing users to monitor their media enhancement tasks live without refreshing the page.

```javascript
// Next.js component for real-time job status updates
import { useState, useEffect } from 'react';

const JobStatus = ({ jobId }) => {
    const [status, setStatus] = useState('initializing');
    const [progress, setProgress] = useState(0);

    useEffect(() => {
        const ws = new WebSocket(`wss://api.tnsr.ai/ws/${jobId}`);
        ws.onmessage = (event) => {
            const data = JSON.parse(event.data);
            setStatus(data?.status);
            setProgress(data?.progress);
        };
        return () => ws.close();
    }, [jobId]);

    return <div>Job Status: {status} {progress}%</div>;
};
```

![TNSR.AI Job Status](/projects/ws_design.png)

### Backend Architecture

The backend is powered by FastAPI, chosen for its high performance and async capabilities. The system uses Celery with Redis for handling asynchronous tasks, particularly important for managing long-running media enhancement jobs and sending dynamic emails to users.

![TNSR.AI Backend Architecture](/projects/backend.png)

### Storage and File Management

File management is handled through Cloudflare R2, providing a cost-effective and reliable object storage solution. The system generates signed URLs for secure file uploads and downloads:

![TNSR.AI File Management](/projects/upload_design.png)

### Monitoring and Observability

One of the key aspects of maintaining a production system is comprehensive monitoring. The platform uses OpenTelemetry for distributed tracing, along with Grafana, Loki, Tempo, and Prometheus for monitoring and visualization.

![TNSR.AI Monitoring](/projects/observability.png)

### Payment Processing and User Management

The platform integrates Stripe for handling payments and subscriptions. User authentication is implemented using JWT tokens with Google SSO support:

### Notification System

The platform features a robust notification system using MJML for beautiful email templates and Discord webhooks for instant updates:


## Deployment and CI/CD

The entire application is containerized using Docker and deployed using Docker Compose. GitHub Actions handles the CI/CD pipeline:

### Example GitHub Actions Workflow

Below is a sample GitHub Actions workflow used to deploy the platform to an Ubuntu server. This workflow checks out the code, sets up Node.js and Python environments, installs dependencies, runs tests, builds the frontend, and finally deploys to the server via SSH:

```yaml
name: Deploy to Ubuntu Server

on:
    push:
        branches:
            - main

jobs:
    deploy:
        runs-on: ubuntu-latest

        steps:
        - name: Checkout code
            uses: actions/checkout@v2

        - name: Set up Node.js for Frontend
            uses: actions/setup-node@v3
            with:
                node-version: '20'

        - name: Install Frontend Dependencies
            run: |
                cd frontend
                npm install

        - name: Run Frontend Tests
            run: |
                cd frontend
                npm test
                npm run cypress:run

        - name: Build Frontend
            run: |
                cd frontend
                npm run build

        - name: Set up Python for Backend
            uses: actions/setup-python@v4
            with:
                python-version: '3.9'

        - name: Install Backend Dependencies
            run: |
                cd backend
                pip install -r requirements.txt

        - name: Run Backend Tests
            run: |
                cd backend
                pytest

        - name: Deploy to Server
            uses: appleboy/ssh-action@v1.1.0
            with:
                host: ${{ secrets.SERVER_HOST }}
                username: ${{ secrets.SERVER_USER }}
                key: ${{ secrets.SERVER_SSH_KEY }}
                port: 22
                script: |
                    ./home/tnsr/deploy.sh
```

## Scaling and Future Improvements

The platform is designed to scale horizontally, with each component being independently scalable. Future improvements include:

- Adding support for more AI models
- Implementing a microservices architecture
- Adding real-time collaboration features
- Expanding the API for third-party integrations


The complete source code and detailed documentation are available in the project repository.

**[Visit TNSR.AI](https://tnsr.ai)**
