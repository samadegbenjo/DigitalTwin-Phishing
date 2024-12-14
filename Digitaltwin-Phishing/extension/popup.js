document.getElementById('evaluate').addEventListener('click', () => {
  chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
    chrome.tabs.sendMessage(tabs[0].id, { action: 'evaluate' }, (response) => {
      const resultDiv = document.getElementById('result');
      if (response && response.phishing !== undefined) {
        resultDiv.innerText = response.phishing ? 'Phishing detected!' : 'No phishing detected.';
      } else {
        resultDiv.innerText = 'Error evaluating the page.';
      }
    });
  });
});