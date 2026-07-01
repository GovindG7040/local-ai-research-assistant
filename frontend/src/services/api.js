import axios from "axios";

// ==========================================
// AXIOS INSTANCE
// ==========================================

const api = axios.create({
    baseURL: "http://127.0.0.1:8000",
});

// ==========================================
// CHAT API
// ==========================================

export const askQuestion = async (question) => {

    const response = await api.post("/chat/", {
        question,
    });

    return response.data;
};

// ==========================================
// INGEST API
// ==========================================

export const ingestKnowledgeBase = async () => {

    const response = await api.post("/ingest/");

    return response.data;
};