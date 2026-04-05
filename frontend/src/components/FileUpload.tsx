import { useState, useRef, useEffect, useCallback } from "react";
import { uploadSingleFile, getDocuments, deleteDocument } from "../services/api";
import type { MultiUploadResult, DocumentInfo } from "../types";

const ACCEPTED = ".pdf,.docx,.doc,.md";

type UploadStatus = "pending" | "uploading" | "processing" | "done" | "error";

interface FileTask {
  id: string;
  name: string;
  size: number;
  status: UploadStatus;
  progress: number; // 0-100 upload bytes progress
  result?: MultiUploadResult;
  error?: string;
}

let taskIdCounter = 0;

export default function FileUpload() {
  const [dragging, setDragging] = useState(false);
  const [tasks, setTasks] = useState<FileTask[]>([]);
  const [documents, setDocuments] = useState<DocumentInfo[]>([]);
  const [deleting, setDeleting] = useState<string | null>(null);
  const fileRef = useRef<HTMLInputElement>(null);
  const processingRef = useRef(false);
  const queueRef = useRef<{ task: FileTask; file: File }[]>([]);

  // Load existing documents on mount
  useEffect(() => {
    getDocuments()
      .then((res) => setDocuments(res.documents))
      .catch(() => {});
  }, []);

  const updateTask = useCallback(
    (id: string, patch: Partial<FileTask>) =>
      setTasks((prev) => prev.map((t) => (t.id === id ? { ...t, ...patch } : t))),
    [],
  );

  const processQueue = useCallback(async () => {
    if (processingRef.current) return;
    processingRef.current = true;

    while (queueRef.current.length > 0) {
      const item = queueRef.current.shift()!;
      const { task, file } = item;

      updateTask(task.id, { status: "uploading", progress: 0 });

      try {
        const res = await uploadSingleFile(file, (percent) => {
          if (percent >= 100) {
            updateTask(task.id, { status: "processing", progress: 100 });
          } else {
            updateTask(task.id, { progress: percent });
          }
        });

        const result = res.results[0];
        if (result?.success) {
          updateTask(task.id, { status: "done", progress: 100, result });
        } else {
          updateTask(task.id, {
            status: "error",
            progress: 100,
            error: result?.error || "上传失败",
          });
        }
      } catch (err) {
        updateTask(task.id, {
          status: "error",
          progress: 100,
          error: err instanceof Error ? err.message : "上传失败",
        });
      }
    }

    // Refresh document list after all uploads
    try {
      const docs = await getDocuments();
      setDocuments(docs.documents);
    } catch {}

    processingRef.current = false;
  }, [updateTask]);

  const handleFiles = useCallback(
    (files: FileList | File[]) => {
      const fileArray = Array.from(files);
      if (fileArray.length === 0) return;

      const newTasks: FileTask[] = fileArray.map((f) => ({
        id: `upload-${++taskIdCounter}`,
        name: f.name,
        size: f.size,
        status: "pending" as const,
        progress: 0,
      }));

      setTasks((prev) => [...newTasks, ...prev]);

      for (let i = 0; i < fileArray.length; i++) {
        queueRef.current.push({ task: newTasks[i], file: fileArray[i] });
      }

      processQueue();
    },
    [processQueue],
  );

  const onDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setDragging(false);
    handleFiles(e.dataTransfer.files);
  };

  const onFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files) handleFiles(e.target.files);
    e.target.value = "";
  };

  const handleDelete = async (filename: string) => {
    setDeleting(filename);
    try {
      await deleteDocument(filename);
      setDocuments((prev) => prev.filter((d) => d.source_doc !== filename));
    } catch (err) {
      // ignore
    } finally {
      setDeleting(null);
    }
  };

  const clearCompleted = () => {
    setTasks((prev) => prev.filter((t) => t.status !== "done" && t.status !== "error"));
  };

  const hasActive = tasks.some((t) => t.status !== "done" && t.status !== "error");
  const hasFinished = tasks.some((t) => t.status === "done" || t.status === "error");

  return (
    <div className="flex flex-col h-full">
      {/* Header */}
      <div className="shrink-0 px-6 py-4 border-b border-gray-200 bg-white">
        <h1 className="text-lg font-semibold text-gray-900">文档管理</h1>
        <p className="text-sm text-gray-500">
          上传 PDF / Word / Markdown 文档，支持多文件同时上传
        </p>
      </div>

      <div className="flex-1 overflow-y-auto px-6 py-8">
        <div className="max-w-lg mx-auto space-y-6">
          {/* Upload area */}
          <div
            onDragOver={(e) => {
              e.preventDefault();
              setDragging(true);
            }}
            onDragLeave={() => setDragging(false)}
            onDrop={onDrop}
            onClick={() => !hasActive && fileRef.current?.click()}
            className={`border-2 border-dashed rounded-2xl p-12 text-center transition-colors ${
              hasActive
                ? "border-gray-200 bg-gray-50 cursor-default"
                : dragging
                  ? "border-indigo-500 bg-indigo-50 cursor-pointer"
                  : "border-gray-300 hover:border-indigo-400 hover:bg-gray-50 cursor-pointer"
            }`}
          >
            <div className="text-4xl mb-3 text-gray-400">
              {hasActive ? "⏳" : "📁"}
            </div>
            {hasActive ? (
              <p className="text-sm text-gray-500 animate-pulse">
                正在处理文件...
              </p>
            ) : (
              <>
                <p className="text-sm text-gray-600 font-medium">
                  拖拽文件到这里，或点击选择
                </p>
                <p className="text-xs text-gray-400 mt-2">
                  支持 PDF, DOCX, DOC, Markdown，可多选
                </p>
              </>
            )}
          </div>
          <input
            ref={fileRef}
            type="file"
            accept={ACCEPTED}
            multiple
            onChange={onFileChange}
            className="hidden"
          />

          {/* Upload task list */}
          {tasks.length > 0 && (
            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <h2 className="text-sm font-semibold text-gray-700">
                  上传进度
                </h2>
                {hasFinished && !hasActive && (
                  <button
                    onClick={clearCompleted}
                    className="text-xs text-gray-400 hover:text-gray-600 transition-colors"
                  >
                    清除已完成
                  </button>
                )}
              </div>
              {tasks.map((task) => (
                <TaskItem key={task.id} task={task} />
              ))}
            </div>
          )}

          {/* Document list */}
          {documents.length > 0 && (
            <div>
              <h2 className="text-sm font-semibold text-gray-700 mb-3">
                已上传文档 ({documents.length})
              </h2>
              <div className="space-y-2">
                {documents.map((doc) => (
                  <div
                    key={doc.source_doc}
                    className="flex items-center justify-between p-3 bg-white border border-gray-200 rounded-xl"
                  >
                    <div className="min-w-0 flex-1">
                      <p className="text-sm font-medium text-gray-900 truncate">
                        {doc.source_doc}
                      </p>
                      <p className="text-xs text-gray-500">
                        {doc.concept_count} 个概念
                      </p>
                    </div>
                    <button
                      onClick={() => handleDelete(doc.source_doc)}
                      disabled={deleting === doc.source_doc}
                      className="ml-3 shrink-0 px-3 py-1.5 text-xs text-red-600 bg-red-50 border border-red-200 rounded-lg hover:bg-red-100 disabled:opacity-50 transition-colors"
                    >
                      {deleting === doc.source_doc ? "删除中..." : "删除"}
                    </button>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

// ── Task progress item ─────────────────────────────────────

function formatSize(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function TaskItem({ task }: { task: FileTask }) {
  const statusConfig: Record<
    UploadStatus,
    { label: string; color: string; barColor: string }
  > = {
    pending: { label: "等待中", color: "text-gray-400", barColor: "bg-gray-300" },
    uploading: { label: `上传中 ${task.progress}%`, color: "text-blue-600", barColor: "bg-blue-500" },
    processing: { label: "服务端处理中...", color: "text-amber-600", barColor: "bg-amber-500" },
    done: { label: "完成", color: "text-green-600", barColor: "bg-green-500" },
    error: { label: "失败", color: "text-red-600", barColor: "bg-red-500" },
  };

  const cfg = statusConfig[task.status];

  return (
    <div className="p-3 bg-white border border-gray-200 rounded-xl">
      {/* File info row */}
      <div className="flex items-center justify-between mb-1.5">
        <p className="text-sm font-medium text-gray-900 truncate mr-3">
          {task.name}
        </p>
        <span className={`text-xs font-medium shrink-0 ${cfg.color}`}>
          {cfg.label}
        </span>
      </div>

      {/* Progress bar */}
      {(task.status === "uploading" || task.status === "processing" || task.status === "pending") && (
        <div className="w-full h-1.5 bg-gray-100 rounded-full overflow-hidden">
          <div
            className={`h-full rounded-full transition-all duration-300 ${cfg.barColor} ${
              task.status === "processing" ? "animate-pulse" : ""
            }`}
            style={{ width: `${task.status === "pending" ? 0 : task.progress}%` }}
          />
        </div>
      )}

      {/* Result details */}
      {task.status === "done" && task.result && (
        <p className="text-xs text-gray-500 mt-1">
          {task.result.chunks_inserted} 个块，{task.result.concepts_total} 个概念
          (新增 {task.result.concepts_new})
        </p>
      )}
      {task.status === "error" && task.error && (
        <p className="text-xs text-red-500 mt-1">{task.error}</p>
      )}

      {/* File size */}
      <p className="text-xs text-gray-400 mt-1">{formatSize(task.size)}</p>
    </div>
  );
}
