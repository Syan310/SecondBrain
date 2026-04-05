import axios from "axios";
import type {
  ChatResponse,
  ChatStreamStatusEvent,
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

interface SendChatStreamHandlers {
  onStatus?: (event: ChatStreamStatusEvent) => void;
  onToken?: (text: string) => void;
  onFinal?: (response: ChatResponse) => void;
}

function parseSseEvent(
  rawEvent: string,
  handlers: SendChatStreamHandlers,
): void {
  const lines = rawEvent
    .split("\n")
    .map((line) => line.trim())
    .filter(Boolean);

  if (lines.length === 0) return;

  let eventType = "message";
  const dataLines: string[] = [];

  for (const line of lines) {
    if (line.startsWith("event:")) {
      eventType = line.slice(6).trim();
      continue;
    }
    if (line.startsWith("data:")) {
      dataLines.push(line.slice(5).trim());
    }
  }

  if (dataLines.length === 0) return;

  const payload = JSON.parse(dataLines.join("\n"));

  if (eventType === "status") {
    handlers.onStatus?.(payload as ChatStreamStatusEvent);
    return;
  }
  if (eventType === "token") {
    handlers.onToken?.(String(payload.text ?? ""));
    return;
  }
  if (eventType === "final") {
    handlers.onFinal?.(payload as ChatResponse);
    return;
  }
  if (eventType === "error") {
    throw new Error(String(payload.message ?? "流式请求失败"));
  }
}

export async function sendChatStream(
  userQuery: string,
  userId = "default",
  sessionId = "default",
  handlers: SendChatStreamHandlers = {},
): Promise<void> {
  const baseUrl = import.meta.env.VITE_API_BASE_URL || "/api";
  const response = await fetch(`${baseUrl}/chat/stream`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      user_query: userQuery,
      user_id: userId,
      session_id: sessionId,
    }),
  });

  if (!response.ok) {
    throw new Error(`请求失败: ${response.status}`);
  }
  if (!response.body) {
    throw new Error("浏览器不支持流式响应");
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { value, done } = await reader.read();
    buffer += decoder.decode(value || new Uint8Array(), { stream: !done });

    let boundary = buffer.indexOf("\n\n");
    while (boundary !== -1) {
      const rawEvent = buffer.slice(0, boundary);
      buffer = buffer.slice(boundary + 2);
      parseSseEvent(rawEvent, handlers);
      boundary = buffer.indexOf("\n\n");
    }

    if (done) {
      break;
    }
  }

  if (buffer.trim()) {
    parseSseEvent(buffer, handlers);
  }
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
