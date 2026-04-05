import { useState, useRef, useEffect } from "react";
import ReactMarkdown from "react-markdown";
import rehypeKatex from "rehype-katex";
import remarkGfm from "remark-gfm";
import remarkMath from "remark-math";
import "katex/dist/katex.min.css";
import { sendChatStream, getChatHistory } from "../services/api";
import QuizModal from "./QuizModal";
import type { ChatMessage, QuizData } from "../types";

const INTRO_CARDS = [
  {
    icon: "💬",
    title: "知识问答",
    desc: "基于你的笔记进行 RAG 检索问答，支持简单和多跳复杂问题",
    example: "什么是 Attention 机制？",
  },
  {
    icon: "📝",
    title: "Quiz 复习",
    desc: "根据掌握度自适应出题，答完自动更新掌握度分数",
    example: "来几道 NLP 的题",
  },
  {
    icon: "🔍",
    title: "盲区诊断",
    desc: "交叉对比知识库与掌握度，找出薄弱和未接触的知识点",
    example: "分析我在机器学习方面的知识盲区",
  },
];

export default function ChatPanel() {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [quizData, setQuizData] = useState<QuizData | null>(null);
  const bottomRef = useRef<HTMLDivElement>(null);

  // Load chat history on mount
  useEffect(() => {
    getChatHistory()
      .then((res) => {
        if (res.messages.length > 0) {
          setMessages(
            res.messages.map((m) => ({ role: m.role, content: m.content })),
          );
        }
      })
      .catch(() => {});
  }, []);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const appendStage = (message: string) => {
    setMessages((prev) => {
      const next = [...prev];
      const last = next[next.length - 1];
      if (!last || last.role !== "assistant") return prev;

      const stages = last.stages ?? [];
      if (stages[stages.length - 1] === message) return prev;

      next[next.length - 1] = {
        ...last,
        stages: [...stages, message],
      };
      return next;
    });
  };

  const appendAssistantToken = (text: string) => {
    if (!text) return;

    setMessages((prev) => {
      const next = [...prev];
      const last = next[next.length - 1];
      if (!last || last.role !== "assistant") return prev;

      next[next.length - 1] = {
        ...last,
        content: `${last.content}${text}`,
      };
      return next;
    });
  };

  const handleSend = async (override?: string) => {
    const query = (override ?? input).trim();
    if (!query || loading) return;

    setMessages((prev) => [
      ...prev,
      { role: "user", content: query },
      { role: "assistant", content: "", stages: ["已发送请求，准备开始处理..."] },
    ]);
    if (!override) setInput("");
    setLoading(true);

    try {
      let finalResponse: {
        answer: string;
        step_results: { intent: string; result: string }[];
      } | null = null;

      await sendChatStream(query, "default", "default", {
        onStatus: (event) => appendStage(event.message),
        onToken: (text) => appendAssistantToken(text),
        onFinal: (response) => {
          finalResponse = response;
        },
      });

      const res = finalResponse;
      if (!res) {
        throw new Error("未收到最终响应");
      }

      const quizStep = res.step_results.find((s) => s.intent === "quiz");
      let parsedQuiz: QuizData | null = null;
      if (quizStep) {
        try {
          parsedQuiz = JSON.parse(quizStep.result) as QuizData;
          setQuizData(parsedQuiz);
        } catch {
          parsedQuiz = null;
        }
      }

      const answerText =
        parsedQuiz && res.answer.trim().startsWith("{")
          ? "已为你生成测验，请在弹窗中作答。"
          : res.answer;

      setMessages((prev) => {
        const next = [...prev];
        const last = next[next.length - 1];
        if (!last || last.role !== "assistant") return prev;

        next[next.length - 1] = {
          ...last,
          content: answerText || last.content,
          quizData: parsedQuiz ?? undefined,
        };
        return next;
      });
    } catch (err) {
      const msg = err instanceof Error ? err.message : "请求失败";
      setMessages((prev) => {
        const next = [...prev];
        const last = next[next.length - 1];
        if (!last || last.role !== "assistant") {
          return [...prev, { role: "assistant", content: `错误: ${msg}` }];
        }

        next[next.length - 1] = {
          ...last,
          content: `错误: ${msg}`,
        };
        return next;
      });
    } finally {
      setLoading(false);
    }
  };

  return (
    <>
      <div className="flex flex-col h-full">
        {/* Header */}
        <div className="shrink-0 px-6 py-4 border-b border-gray-200 bg-white">
          <h1 className="text-lg font-semibold text-gray-900">SecondBrain</h1>
          <p className="text-sm text-gray-500">
            基于你的笔记，问答 · 复习 · 诊断
          </p>
        </div>

        {/* Messages */}
        <div className="flex-1 overflow-y-auto px-6 py-4 space-y-4">
          {messages.length === 0 && (
            <div className="mt-12 max-w-2xl mx-auto">
              <p className="text-center text-gray-500 mb-8 text-sm">
                我可以帮你做以下事情，直接输入或点击示例开始：
              </p>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                {INTRO_CARDS.map((card) => (
                  <button
                    key={card.title}
                    onClick={() => handleSend(card.example)}
                    className="text-left p-4 bg-white border border-gray-200 rounded-xl hover:border-indigo-300 hover:shadow-sm transition-all cursor-pointer"
                  >
                    <div className="text-2xl mb-2">{card.icon}</div>
                    <h3 className="text-sm font-semibold text-gray-900 mb-1">
                      {card.title}
                    </h3>
                    <p className="text-xs text-gray-500 mb-3">{card.desc}</p>
                    <p className="text-xs text-indigo-500 italic">
                      &ldquo;{card.example}&rdquo;
                    </p>
                  </button>
                ))}
              </div>
            </div>
          )}
          {messages.map((msg, i) => (
            <div
              key={i}
              className={`flex ${msg.role === "user" ? "justify-end" : "justify-start"}`}
            >
              <div
                className={`max-w-[75%] rounded-2xl px-4 py-3 text-sm leading-relaxed ${
                  msg.role === "user"
                    ? "bg-indigo-600 text-white"
                    : "bg-white border border-gray-200 text-gray-800"
                }`}
              >
                {msg.role === "assistant" && msg.stages && msg.stages.length > 0 && (
                  <div className="mb-3 space-y-1">
                    {msg.stages.map((stage, idx) => (
                      <div key={idx} className="text-xs text-gray-500">
                        {stage}
                      </div>
                    ))}
                  </div>
                )}
                {msg.role === "assistant" ? (
                  <div className="break-words [&>*:first-child]:mt-0 [&>*:last-child]:mb-0 [&_a]:text-indigo-600 [&_a]:underline [&_blockquote]:my-3 [&_blockquote]:border-l-4 [&_blockquote]:border-gray-300 [&_blockquote]:pl-4 [&_code]:rounded-md [&_code]:bg-gray-100 [&_code]:px-1.5 [&_code]:py-0.5 [&_code]:font-mono [&_code]:text-[0.9em] [&_h1]:mb-3 [&_h1]:text-lg [&_h1]:font-semibold [&_h2]:mb-3 [&_h2]:text-base [&_h2]:font-semibold [&_h3]:mb-2 [&_h3]:text-sm [&_h3]:font-semibold [&_hr]:my-4 [&_li]:mb-1 [&_ol]:mb-3 [&_ol]:list-decimal [&_ol]:pl-5 [&_p]:mb-3 [&_pre]:my-3 [&_pre]:overflow-x-auto [&_pre]:rounded-xl [&_pre]:bg-gray-900 [&_pre]:px-4 [&_pre]:py-3 [&_pre]:text-gray-100 [&_pre_code]:bg-transparent [&_pre_code]:p-0 [&_pre_code]:text-inherit [&_pre_code]:rounded-none [&_table]:mb-3 [&_table]:w-full [&_table]:border-collapse [&_td]:border [&_td]:border-gray-200 [&_td]:px-3 [&_td]:py-2 [&_th]:border [&_th]:border-gray-200 [&_th]:bg-gray-50 [&_th]:px-3 [&_th]:py-2 [&_ul]:mb-3 [&_ul]:list-disc [&_ul]:pl-5">
                    <ReactMarkdown
                      remarkPlugins={[remarkGfm, remarkMath]}
                      rehypePlugins={[rehypeKatex]}
                    >
                      {msg.content}
                    </ReactMarkdown>
                  </div>
                ) : (
                  <div className="whitespace-pre-wrap">{msg.content}</div>
                )}
              </div>
            </div>
          ))}
          <div ref={bottomRef} />
        </div>

        {/* Input */}
        <div className="shrink-0 px-6 py-4 border-t border-gray-200 bg-white">
          <div className="flex gap-3">
            <input
              type="text"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) =>
                e.key === "Enter" && !e.shiftKey && handleSend()
              }
              placeholder="输入你的问题..."
              disabled={loading}
              className="flex-1 px-4 py-2.5 border border-gray-300 rounded-xl text-sm focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:border-transparent disabled:opacity-50"
            />
            <button
              onClick={() => handleSend()}
              disabled={loading || !input.trim()}
              className="px-5 py-2.5 bg-indigo-600 text-white text-sm font-medium rounded-xl hover:bg-indigo-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
            >
              发送
            </button>
          </div>
        </div>
      </div>

      {/* Quiz Modal */}
      {quizData && (
        <QuizModal quizData={quizData} onClose={() => setQuizData(null)} />
      )}
    </>
  );
}
