import type { Panel } from "../types";

const NAV_ITEMS: { key: Panel; label: string; icon: string }[] = [
  { key: "chat", label: "聊天问答", icon: "💬" },
  { key: "upload", label: "文档上传", icon: "📄" },
];

interface LayoutProps {
  activePanel: Panel;
  onNavigate: (panel: Panel) => void;
  children: React.ReactNode;
}

export default function Layout({
  activePanel,
  onNavigate,
  children,
}: LayoutProps) {
  return (
    <div className="flex h-screen bg-gray-50">
      {/* Sidebar */}
      <nav className="w-56 shrink-0 bg-gray-900 text-gray-300 flex flex-col">
        <div className="px-5 py-6 text-xl font-bold text-white tracking-tight">
          SecondBrain
        </div>
        <ul className="flex-1 space-y-1 px-3">
          {NAV_ITEMS.map(({ key, label, icon }) => (
            <li key={key}>
              <button
                onClick={() => onNavigate(key)}
                className={`w-full flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm transition-colors cursor-pointer ${
                  activePanel === key
                    ? "bg-gray-700 text-white"
                    : "hover:bg-gray-800 hover:text-white"
                }`}
              >
                <span className="text-base">{icon}</span>
                <span>{label}</span>
              </button>
            </li>
          ))}
        </ul>
      </nav>

      {/* Content */}
      <main className="flex-1 overflow-hidden">{children}</main>
    </div>
  );
}
