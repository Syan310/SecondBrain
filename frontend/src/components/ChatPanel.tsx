import { useState, useRef, useEffect } from "react";
import { sendChat } from "../services/api";
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

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const handleSend = async (override?: string) => {
    const query = (override ?? input).trim();
    if (!query || loading) return;

    setMessages((prev) => [...prev, { role: "user", content: query }]);
    if (!override) setInput("");
    setLoading(true);

    try {
      const res = await sendChat(query);

      // Check if any step is a quiz intent
      const quizStep = res.step_results.find((s) => s.intent === "quiz");
      if (quizStep) {
        try {
          const parsed: QuizData = JSON.parse(quizStep.result);
          setMessages((prev) => [
            ...prev,
            { role: "assistant", content: res.answer, quizData: parsed },
          ]);
          setQuizData(parsed);
          return;
        } catch {
          // Not valid quiz JSON, treat as normal answer
        }
      }

      setMessages((prev) => [
        ...prev,
        { role: "assistant", content: res.answer },
      ]);
    } catch (err) {
      const msg = err instanceof Error ? err.message : "请求失败";
      setMessages((prev) => [
        ...prev,
        { role: "assistant", content: `错误: ${msg}` },
      ]);
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
                className={`max-w-[75%] rounded-2xl px-4 py-3 text-sm leading-relaxed whitespace-pre-wrap ${
                  msg.role === "user"
                    ? "bg-indigo-600 text-white"
                    : "bg-white border border-gray-200 text-gray-800"
                }`}
              >
                {msg.content}
              </div>
            </div>
          ))}
          {loading && (
            <div className="flex justify-start">
              <div className="bg-white border border-gray-200 rounded-2xl px-4 py-3 text-sm text-gray-400 animate-pulse">
                思考中...
              </div>
            </div>
          )}
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
