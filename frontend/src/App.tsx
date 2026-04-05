import { useState } from "react";
import Layout from "./components/Layout";
import ChatPanel from "./components/ChatPanel";
import FileUpload from "./components/FileUpload";
import type { Panel } from "./types";

export default function App() {
  const [activePanel, setActivePanel] = useState<Panel>("chat");

  return (
    <Layout activePanel={activePanel} onNavigate={setActivePanel}>
      {activePanel === "chat" && <ChatPanel />}
      {activePanel === "upload" && <FileUpload />}
    </Layout>
  );
}
