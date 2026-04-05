export interface StepResult {
  step: number;
  intent: string;
  result: string;
}

export interface ChatResponse {
  answer: string;
  step_results: StepResult[];
}

export interface ChatStreamStatusEvent {
  stage: string;
  message: string;
  step?: number;
  intent?: string;
  total_steps?: number;
}

export interface QuizQuestion {
  topic: string;
  question: string;
  choices: string[];
  correct_answer: string;
  explanation: string;
  source_chunks?: string[];
}

export interface QuizData {
  questions: QuizQuestion[];
  topics: string[];
}

export interface MasteryUpdate {
  topic: string;
  old_effective_mastery: number;
  new_mastery_score: number;
  total_attempts: number;
  correct_count: number;
}

export interface QuizAnswerResponse {
  is_correct: boolean;
  correct_answer: string;
  mastery_update: MasteryUpdate;
}

export interface IngestResponse {
  filename: string;
  parents: number;
  chunks_inserted: number;
  concepts_total: number;
  concepts_new: number;
}

export interface MultiUploadResult {
  filename: string;
  success: boolean;
  parents: number;
  chunks_inserted: number;
  concepts_total: number;
  concepts_new: number;
  error: string | null;
}

export interface MultiUploadResponse {
  results: MultiUploadResult[];
}

export interface DocumentInfo {
  source_doc: string;
  concept_count: number;
}

export interface DocumentListResponse {
  documents: DocumentInfo[];
}

export interface HistoryMessage {
  role: "user" | "assistant";
  content: string;
}

export interface ChatHistoryResponse {
  messages: HistoryMessage[];
}

export interface ChatMessage {
  role: "user" | "assistant";
  content: string;
  quizData?: QuizData;
  stages?: string[];
}

export type Panel = "chat" | "upload";
