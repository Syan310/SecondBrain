import { useState, useRef, useEffect } from "react";
import { uploadFiles, getDocuments, deleteDocument } from "../services/api";
import type { MultiUploadResult, DocumentInfo } from "../types";

const ACCEPTED = ".pdf,.docx,.doc,.md";

export default function FileUpload() {
  const [dragging, setDragging] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [results, setResults] = useState<MultiUploadResult[]>([]);
  const [error, setError] = useState("");
  const [documents, setDocuments] = useState<DocumentInfo[]>([]);
  const [deleting, setDeleting] = useState<string | null>(null);
  const fileRef = useRef<HTMLInputElement>(null);

  // Load existing documents on mount
  useEffect(() => {
    getDocuments()
      .then((res) => setDocuments(res.documents))
      .catch(() => {});
  }, []);

  const handleFiles = async (files: FileList | File[]) => {
    const fileArray = Array.from(files);
    if (fileArray.length === 0) return;

    setError("");
    setResults([]);
    setUploading(true);

    try {
      const res = await uploadFiles(fileArray);
      setResults(res.results);
      // Refresh document list
      const docs = await getDocuments();
      setDocuments(docs.documents);
    } catch (err) {
      setError(err instanceof Error ? err.message : "上传失败");
    } finally {
      setUploading(false);
    }
  };

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
      setError(err instanceof Error ? err.message : "删除失败");
    } finally {
      setDeleting(null);
    }
  };

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
            onClick={() => fileRef.current?.click()}
            className={`border-2 border-dashed rounded-2xl p-12 text-center cursor-pointer transition-colors ${
              dragging
                ? "border-indigo-500 bg-indigo-50"
                : "border-gray-300 hover:border-indigo-400 hover:bg-gray-50"
            }`}
          >
            <div className="text-4xl mb-3 text-gray-400">
              {uploading ? "⏳" : "📁"}
            </div>
            {uploading ? (
              <p className="text-sm text-gray-500 animate-pulse">
                正在上传并解析...
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

          {/* Upload results */}
          {results.length > 0 && (
            <div className="space-y-2">
              {results.map((r) => (
                <div
                  key={r.filename}
                  className={`p-4 rounded-xl text-sm border ${
                    r.success
                      ? "bg-green-50 border-green-200 text-green-800"
                      : "bg-red-50 border-red-200 text-red-700"
                  }`}
                >
                  <p className="font-medium">
                    {r.success ? "✓" : "✗"} {r.filename}
                  </p>
                  {r.success ? (
                    <p className="text-gray-600 mt-1">
                      {r.chunks_inserted} 个块，{r.concepts_total} 个概念
                      (新增 {r.concepts_new})
                    </p>
                  ) : (
                    <p className="mt-1">{r.error}</p>
                  )}
                </div>
              ))}
            </div>
          )}

          {/* Error */}
          {error && (
            <div className="p-4 bg-red-50 border border-red-200 rounded-xl text-sm text-red-700">
              {error}
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
