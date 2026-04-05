import { useState } from "react";
import { submitQuizAnswer } from "../services/api";
import type { QuizData, QuizAnswerResponse } from "../types";

interface QuizModalProps {
  quizData: QuizData;
  onClose: () => void;
}

interface AnswerResult {
  questionIndex: number;
  response: QuizAnswerResponse;
}

export default function QuizModal({ quizData, onClose }: QuizModalProps) {
  const [currentIndex, setCurrentIndex] = useState(0);
  const [selected, setSelected] = useState<string | null>(null);
  const [results, setResults] = useState<AnswerResult[]>([]);
  const [submitting, setSubmitting] = useState(false);
  const [showResult, setShowResult] = useState(false);
  const [currentResult, setCurrentResult] = useState<QuizAnswerResponse | null>(
    null,
  );

  const question = quizData.questions[currentIndex];
  const isFinished = currentIndex >= quizData.questions.length;
  const choiceLetter = (idx: number) => String.fromCharCode(65 + idx);

  const handleSubmit = async () => {
    if (!selected || submitting) return;
    setSubmitting(true);

    try {
      const res = await submitQuizAnswer(
        question.topic,
        selected,
        question.correct_answer,
      );
      setCurrentResult(res);
      setResults((prev) => [
        ...prev,
        { questionIndex: currentIndex, response: res },
      ]);
      setShowResult(true);
    } catch {
      setCurrentResult({
        is_correct: false,
        correct_answer: question.correct_answer,
        mastery_update: {
          topic: question.topic,
          old_effective_mastery: 0,
          new_mastery_score: 0,
          total_attempts: 0,
          correct_count: 0,
        },
      });
      setShowResult(true);
    } finally {
      setSubmitting(false);
    }
  };

  const handleNext = () => {
    setSelected(null);
    setShowResult(false);
    setCurrentResult(null);
    setCurrentIndex((prev) => prev + 1);
  };

  // Summary view
  if (isFinished) {
    const correctCount = results.filter((r) => r.response.is_correct).length;
    return (
      <div
        className="fixed inset-0 z-50 flex items-center justify-center bg-black/40"
        onClick={onClose}
      >
        <div
          className="bg-white rounded-2xl shadow-xl w-full max-w-lg max-h-[80vh] overflow-y-auto mx-4"
          onClick={(e) => e.stopPropagation()}
        >
          <div className="px-6 py-5 border-b border-gray-100 flex items-center justify-between">
            <h2 className="text-lg font-semibold text-gray-900">Quiz 完成</h2>
            <button
              onClick={onClose}
              className="text-gray-400 hover:text-gray-600 text-xl leading-none"
            >
              &times;
            </button>
          </div>
          <div className="px-6 py-6">
            <div className="text-center mb-6">
              <div className="text-4xl font-bold text-indigo-600">
                {correctCount}/{quizData.questions.length}
              </div>
              <p className="text-sm text-gray-500 mt-1">答对题数</p>
            </div>
            <div className="space-y-3">
              {results.map((r, i) => {
                const q = quizData.questions[r.questionIndex];
                return (
                  <div
                    key={i}
                    className={`p-3 rounded-xl border ${r.response.is_correct ? "border-green-200 bg-green-50" : "border-red-200 bg-red-50"}`}
                  >
                    <div className="flex items-start gap-2">
                      <span
                        className={`shrink-0 mt-0.5 text-sm font-medium ${r.response.is_correct ? "text-green-600" : "text-red-600"}`}
                      >
                        {r.response.is_correct ? "✓" : "✗"}
                      </span>
                      <div className="text-sm">
                        <p className="font-medium text-gray-800">
                          {q.question}
                        </p>
                        {!r.response.is_correct && (
                          <p className="text-gray-500 mt-1">
                            正确答案: {r.response.correct_answer}
                          </p>
                        )}
                        <p className="text-xs text-gray-400 mt-1">
                          掌握度:{" "}
                          {(
                            r.response.mastery_update.new_mastery_score * 100
                          ).toFixed(0)}
                          %
                        </p>
                      </div>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
          <div className="px-6 py-4 border-t border-gray-100">
            <button
              onClick={onClose}
              className="w-full py-2.5 bg-indigo-600 text-white text-sm font-medium rounded-xl hover:bg-indigo-700 transition-colors"
            >
              关闭
            </button>
          </div>
        </div>
      </div>
    );
  }

  // Question view
  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/40"
      onClick={onClose}
    >
      <div
        className="bg-white rounded-2xl shadow-xl w-full max-w-lg max-h-[80vh] overflow-y-auto mx-4"
        onClick={(e) => e.stopPropagation()}
      >
        {/* Header */}
        <div className="px-6 py-5 border-b border-gray-100 flex items-center justify-between">
          <div>
            <h2 className="text-lg font-semibold text-gray-900">Quiz</h2>
            <p className="text-xs text-gray-500">主题: {question.topic}</p>
          </div>
          <span className="text-sm text-gray-400">
            {currentIndex + 1} / {quizData.questions.length}
          </span>
        </div>

        {/* Body */}
        <div className="px-6 py-5">
          <p className="text-sm font-medium text-gray-900 mb-5">
            {question.question}
          </p>

          <div className="space-y-2.5">
            {question.choices.map((choice, idx) => {
              const letter = choiceLetter(idx);
              const isSelected = selected === letter;
              let borderColor = "border-gray-200";
              let bgColor = "bg-white";

              if (showResult && currentResult) {
                if (letter === currentResult.correct_answer) {
                  borderColor = "border-green-400";
                  bgColor = "bg-green-50";
                } else if (isSelected && !currentResult.is_correct) {
                  borderColor = "border-red-400";
                  bgColor = "bg-red-50";
                }
              } else if (isSelected) {
                borderColor = "border-indigo-500";
                bgColor = "bg-indigo-50";
              }

              return (
                <button
                  key={idx}
                  onClick={() => !showResult && setSelected(letter)}
                  disabled={showResult}
                  className={`w-full text-left px-4 py-3 rounded-xl border-2 text-sm transition-colors ${borderColor} ${bgColor} ${!showResult ? "cursor-pointer hover:border-indigo-300" : ""} disabled:cursor-default`}
                >
                  {choice}
                </button>
              );
            })}
          </div>

          {showResult && currentResult && (
            <div
              className={`mt-5 p-4 rounded-xl text-sm ${currentResult.is_correct ? "bg-green-50 border border-green-200 text-green-800" : "bg-red-50 border border-red-200 text-red-800"}`}
            >
              <p className="font-medium mb-1">
                {currentResult.is_correct ? "回答正确!" : "回答错误"}
              </p>
              {question.explanation && (
                <p className="text-gray-600">{question.explanation}</p>
              )}
              <p className="text-xs text-gray-400 mt-2">
                掌握度:{" "}
                {(
                  currentResult.mastery_update.old_effective_mastery * 100
                ).toFixed(0)}
                % →{" "}
                {(
                  currentResult.mastery_update.new_mastery_score * 100
                ).toFixed(0)}
                %
              </p>
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="px-6 py-4 border-t border-gray-100">
          {!showResult ? (
            <button
              onClick={handleSubmit}
              disabled={!selected || submitting}
              className="w-full py-2.5 bg-indigo-600 text-white text-sm font-medium rounded-xl hover:bg-indigo-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
            >
              {submitting ? "提交中..." : "提交答案"}
            </button>
          ) : (
            <button
              onClick={handleNext}
              className="w-full py-2.5 bg-indigo-600 text-white text-sm font-medium rounded-xl hover:bg-indigo-700 transition-colors"
            >
              {currentIndex + 1 < quizData.questions.length
                ? "下一题"
                : "查看结果"}
            </button>
          )}
        </div>
      </div>
    </div>
  );
}
