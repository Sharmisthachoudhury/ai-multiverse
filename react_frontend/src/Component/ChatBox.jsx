import React, { useState, useEffect, useRef, useMemo } from "react";
import PropTypes from "prop-types";

const ChatBox = ({ modelKey }) => {
  const [selectedModel, setSelectedModel] = useState("gpt-3.5-turbo");
  const [stableInput, setStableInput] = useState("");
  const [stableInputMessage, setStableInputMessage] = useState([]);
  const [sendButtonDisabled, setSendButtonDisabled] = useState(false);
  const [spinner, setSpinner] = useState(false);
  const [stableResponseMessage, setStableResponseMessage] = useState("");
  const [syncChat, setSyncChat] = useState(false);
  const [showError, setShowError] = useState(false);
  const sendButtonRef = useRef("");

  ChatBox.propTypes = {
    modelKey: PropTypes.number.isRequired,
  };

  const handleModelChange = (event) => {
    setSelectedModel(event.target.value);
  };

  const sendStableInput = async () => {
    if (!stableInput.trim()) {
      setShowError(true);
      setTimeout(() => setShowError(false), 3000);
      return;
    }

    setStableInputMessage([
      ...stableInputMessage,
      {
        id: Date.now(),
        answer: stableInput,
        model: "User",
      },
    ]);

    setSendButtonDisabled(true);
    setSpinner(true);

    try {
      const response = await fetch(`http://localhost:5000/getres`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          prompt: stableInput,
          model: selectedModel,
        }),
      });
      const data = await response.json();
      setStableResponseMessage({
        id: Date.now(),
        answer: data.answer,
        model: selectedModel,
      });
      setStableInput("");
    } catch (error) {
      console.error("Error fetching response from the server:", error);
    } finally {
      setSpinner(false);
      setSendButtonDisabled(false);
    }
  };

  const modelOptions = useMemo(
    () => [
      { value: "command-nightly", label: "command-nightly" },
      { value: "gpt-3.5-turbo", label: "gpt-3.5-turbo" },
      { value: "stable-diffusion", label: "stable-diffusion" },
    ],
    []
  );

  const handleKeyDown = (event) => {
    if (event.keyCode === 13) {
      event.preventDefault();
      sendButtonRef.current.click();
    }
  };

  useEffect(() => {
    if (stableResponseMessage) {
      setStableInputMessage([...stableInputMessage, stableResponseMessage]);
      setStableResponseMessage("");
    }
  }, [stableResponseMessage, stableInputMessage]);

  return (
    <div className="col-md-4" style={{ height: "95vh" }}>
      <div className="d-flex justify-content-start">
        <select
          className="btn btn-dark dropdown-toggle sticky-top"
          value={selectedModel}
          onChange={handleModelChange}
        >
          {modelOptions.map((option) => (
            <option key={option.value} value={option.value}>
              {option.label}
            </option>
          ))}
        </select>
        <label className="form-check-label pt-1 ms-5">Sync Chats</label>
        <input
          className="form-check-input pt-1 ms-2"
          type="checkbox"
          checked={syncChat}
          onChange={(e) => setSyncChat(e.target.checked)}
          id="flexCheckChecked"
        />
      </div>
      {showError && (
        <div
          className="alert alert-danger alert-dismissible fade show"
          role="alert"
        >
          <strong>Error!</strong> Your message can't be empty.
          <button
            type="button"
            className="btn-close"
            data-bs-dismiss="alert"
            aria-label="Close"
            onClick={() => setShowError(false)}
          ></button>
        </div>
      )}
      <div className="overflow-y-auto mb-3" style={{ height: "85%" }}>
        {stableInputMessage &&
          stableInputMessage.length !== 0 &&
          stableInputMessage.map((item) => (
            <div
              key={item.id}
              className={`${
                item.model === "User"
                  ? "d-flex justify-content-end"
                  : "d-flex justify-content-start"
              }`}
            >
              <div className="card mask-custom w-50">
                <div className="card-body">
                  {item.model === "stable-diffusion" &&
                  item.answer.startsWith("https://") ? (
                    <img
                      className="img-fluid"
                      src={item.answer}
                      alt={item.answer}
                    />
                  ) : (
                    <p className="fw-bold">{item.answer}</p>
                  )}
                  <em className="small d-flex justify-content-end">
                    {item.model}
                  </em>
                </div>
              </div>
            </div>
          ))}
      </div>
      <div className="d-flex">
        <input
          className="form-control rounded-pill"
          placeholder="Type your message..."
          type="text"
          value={stableInput}
          onChange={(e) => setStableInput(e.target.value)}
          onKeyDown={handleKeyDown}
          required
          disabled={spinner || sendButtonDisabled}
        />
        <button
          ref={sendButtonRef}
          className="btn btn-primary rounded-pill ms-2"
          type="button"
          onClick={sendStableInput}
          hidden={sendButtonDisabled}
        >
          Send&ensp;🚀
        </button>
        {spinner && (
          <div className="spinner-border text-secondary" role="status">
            <span className="visually-hidden">Loading...</span>
          </div>
        )}
      </div>
    </div>
  );
};

export default ChatBox;
