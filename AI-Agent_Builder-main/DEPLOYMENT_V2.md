# 🚀 Deployment Guide (OpenAI Edition)

This guide helps you deploy your **Multi-Modal Document Intelligence Platform** using **Groq** for fast inference.

---

## 🏗️ Phase 1: Backend Deployment (Render)

1.  **Push to GitHub**: Ensure your latest code is pushed to your repository.
2.  **Create Service**:
    - Go to [dashboard.render.com](https://dashboard.render.com).
    - Click **New +** -> **Web Service**.
    - Connect your `ai_agent_builder` repo.
3.  **Configure**:
    - **Name**: `ai-document-backend`
    - **Runtime**: **Docker**
    - **Region**: Closest to you (e.g., Singapore, Frankfurt).
    - **Branch**: `main`
    - **Initial Deployment**: It will start building automatically.
4.  **Environment Variables** (Scroll down to "Environment"):
    Add these keys:
    - `LLM_PROVIDER`: `openai`
    - `OPENAI_API_KEY`: `your_openai_api_key_here`
    - `LLM_MODEL`: `gpt-4o-mini` (Fast & Cost-effective)
    - `QDRANT_HOST`: `your_qdrant_cloud_url` (or `localhost` if testing without persistence, but Cloud is recommended)
    - `QDRANT_API_KEY`: `your_qdrant_key` (if using Cloud)
    - `PORT`: `8000`
5.  **Deploy**: Click **Create Web Service**.
6.  **Wait**: The build might take 5-10 minutes.
7.  **Copy URL**: Once live, copy your backend URL (e.g., `https://ai-document-backend.onrender.com`).

---

## ⚛️ Phase 2: Frontend Deployment (Vercel)

1.  **Create Project**:
    - Go to [vercel.com/new](https://vercel.com/new).
    - Import your `ai_agent_builder` repo.
2.  **Frontend Settings**:
    - **Framework Preset**: Vite
    - **Root Directory**: Click "Edit" and select `frontend`.
3.  **Environment Variables**:
    - `VITE_API_BASE_URL`: Paste your **Render Backend URL** (e.g., `https://ai-document-backend.onrender.com`).
4.  **Deploy**: Click **Deploy**.

---

## ✅ Phase 3: Verification

1.  Open your **Vercel App URL**.
2.  Upload a PDF.
3.  Ask a question.
    - *Success*: You get a fast response from Groq!
    - *Failure*: Check the browser console (F12) for errors (CORS, 404, etc).

> **Troubleshooting CORS**:
> If you see CORS errors in the browser console, go back to Render -> Environment Variables and add:
> `CORS_ORIGINS`: `https://your-vercel-app.vercel.app` (The URL assigned to you by Vercel).
