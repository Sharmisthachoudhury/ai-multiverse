import "./App.css";
import ChatBox from "./Component/ChatBox";

function App() {
  const chatBoxInstances = [1, 2, 3];
  return (
    <div className="container-fluid">
      <div className="row">
        {chatBoxInstances.map((id) => (
          <ChatBox key={id} modelKey={id} />
        ))}
      </div>
    </div>
  );
}

export default App;
