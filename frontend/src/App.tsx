import ChatPanel from "./components/ChatPanel";
import GapAnalysis from "./components/GapAnalysis";
import QuizPanel from "./components/QuizPanel";
import FileUpload from "./components/FileUpload";
import "./App.css";

function App() {
  return (
    <div className="app">
      <header>
        <h1>SecondBrain</h1>
      </header>
      <main>
        <ChatPanel />
        <GapAnalysis />
        <QuizPanel />
        <FileUpload />
      </main>
    </div>
  );
}

export default App;
