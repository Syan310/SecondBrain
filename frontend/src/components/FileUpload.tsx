import { useState, useRef } from "react";
import { uploadFile } from "../services/api";
import type { IngestResponse } from "../types";

const ACCEPTED = ".pdf,.docx,.doc,.md";

export default function FileUpload() {
  const [dragging, setDragging] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [result, setResult] = useState<IngestResponse | null>(null);
  const [error, setError] = useState("");
  const fileRef = useRef<HTMLInputElement>(null);

  const handleFile = async (file: File) => {
    setError("");
    setResult(null);
    setUploading(true);

    try {
      const res = await uploadFile(file);
      setResult(res);
    } catch (err) {
      setError(err instanceof Error ? err.message : "上传失败");
    } finally {
      setUploading(false);
    }
  };

  const onDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setDragging(false);
    const file = e.dataTransfer.files[0];
    if (file) handleFile(file);
  };

  const onFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) handleFile(file);
    e.target.value = "";
  };

  return (
    <div className="flex flex-col h-full">
      {/* Header */}
      <div className="shrink-0 px-6 py-4 border-b border-gray-200 bg-white">
        <h1 className="text-lg font-semibold text-gray-900">文档上传</h1>
        <p className="text-sm text-gray-500">
          上传 PDF / Word / Markdown 文档，系统自动解析入库
        </p>
      </div>

      {/* Upload area */}
      <div className="flex-1 overflow-y-auto px-6 py-8">
        <div className="max-w-lg mx-auto">
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
                  支持 PDF, DOCX, DOC, Markdown
                </p>
              </>
            )}
          </div>
          <input
            ref={fileRef}
            type="file"
            accept={ACCEPTED}
            onChange={onFileChange}
            className="hidden"
          />

          {/* Result */}
          {result && (
            <div className="mt-6 p-4 bg-green-50 border border-green-200 rounded-xl text-sm text-green-800">
              <p className="font-medium mb-2">上传成功</p>
              <ul className="space-y-1 text-gray-600">
                <li>文件: {result.filename}</li>
                <li>父块: {result.parents} 个</li>
                <li>子块写入: {result.chunks_inserted} 个</li>
                <li>
                  概念: {result.concepts_total} 个 (新增 {result.concepts_new})
                </li>
              </ul>
            </div>
          )}

          {/* Error */}
          {error && (
            <div className="mt-6 p-4 bg-red-50 border border-red-200 rounded-xl text-sm text-red-700">
              {error}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
