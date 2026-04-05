import axios from "axios";
import type {
  ChatResponse,
  QuizAnswerResponse,
  IngestResponse,
} from "../types";

const api = axios.create({
  baseURL: import.meta.env.VITE_API_BASE_URL || "/api",
  timeout: 120_000,
  headers: { "Content-Type": "application/json" },
});

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

export async function uploadFile(file: File): Promise<IngestResponse> {
  const form = new FormData();
  form.append("file", file);
  const { data } = await api.post<IngestResponse>("/ingest/upload", form, {
    headers: { "Content-Type": "multipart/form-data" },
    timeout: 300_000,
  });
  return data;
}

export default api;
