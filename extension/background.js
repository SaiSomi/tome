chrome.runtime.onInstalled.addListener(() => {
  chrome.contextMenus.create({
    id: "saveHighlight",
    title: "Save Highlight to Notes.txt",
    contexts: ["selection"]
  });
});

chrome.contextMenus.onClicked.addListener((info, tab) => {
  if (info.menuItemId === "saveHighlight") {
    if (tab.url.startsWith("chrome://") || tab.url.startsWith("chrome-extension://")) {
      console.warn("Cannot access chrome:// or extension pages.");
      return;
    }

    chrome.scripting.executeScript({
      target: { tabId: tab.id },
      function: getSelectionAndSend
    });
  }
});

function getSelectionAndSend() {
  const selection = window.getSelection().toString().trim();
  if (!selection) return;

  const payload = {
    text: selection,
    url: window.location.href,
    title: document.title
  };

  fetch("http://localhost:5000/highlight", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload)
  }).then(() => {
    console.log("✅ Highlight saved.");
  }).catch(() => {
    console.error("❌ Failed to save highlight. Make sure your local server is running.");
  });
}
