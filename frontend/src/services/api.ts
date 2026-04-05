import axios from "axios";
import type {
  ChatResponse,
  ChatHistoryResponse,
  QuizAnswerResponse,
  MultiUploadResponse,
  DocumentListResponse,
} from "../types";

const api = axios.create({
  baseURL: import.meta.env.VITE_API_BASE_URL || "/api",
  timeout: 120_000,
  headers: { "Content-Type": "application/json" },
});

// ── Chat ────────────────────────────────────────────────────

export async function sendChat(
  userQuery: string,
  userId = "default",
  sessionId = "default",
): Promise<ChatResponse> {
  const { data } = await api.post<ChatResponse>("/chat/", {
    user_query: userQuery,
    user_id: userId,
    session_id: sessionId,
  });
  return data;
}

export async function getChatHistory(
  userId = "default",
  limit = 50,
): Promise<ChatHistoryResponse> {
  const { data } = await api.get<ChatHistoryResponse>("/chat/history", {
    params: { user_id: userId, limit },
  });
  return data;
}

// ── Quiz ────────────────────────────────────────────────────

export async function submitQuizAnswer(
  topic: string,
  userAnswer: string,
  correctAnswer: string,
  userId = "default",
): Promise<QuizAnswerResponse> {
  const { data } = await api.post<QuizAnswerResponse>("/quiz/answer", {
    user_id: userId,
    topic,
    user_answer: userAnswer,
    correct_answer: correctAnswer,
  });
  return data;
}

// ── Ingest / Documents ──────────────────────────────────────

export async function uploadFiles(files: File[]): Promise<MultiUploadResponse> {
  const form = new FormData();
  for (const file of files) {
    form.append("files", file);
  }
  const { data } = await api.post<MultiUploadResponse>("/ingest/upload", form, {
    headers: { "Content-Type": "multipart/form-data" },
    timeout: 300_000,
  });
  return data;
}

export async function getDocuments(): Promise<DocumentListResponse> {
  const { data } = await api.get<DocumentListResponse>("/ingest/documents");
  return data;
}

export async function deleteDocument(filename: string): Promise<void> {
  await api.delete(`/ingest/documents/${encodeURIComponent(filename)}`);
}

export default api;
