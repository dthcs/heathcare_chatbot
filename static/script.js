document.querySelector('form').addEventListener('submit', async function (event) {
    event.preventDefault();

    const formData = new FormData(this);

    try {
        const response = await fetch('/upload', {
            method: 'POST',
            body: formData,
        });

        if (response.ok) {
            const result = await response.json();
            console.log(result);
            displayResult(result);
        } else {
            console.error('Server response not OK');
        }
    } catch (error) {
        console.error('Error during fetch:', error);
    }
});

function displayResult(result) {
    const resultContainer1 = document.getElementById('result-container1');
    const resultContainer2 = document.getElementById('result-container2');
    
    resultContainer1.textContent = `Predictions: ${result["prediction"]}`;
    resultContainer2.textContent = `Recommendations: ${result["recommendation"]}`;
}

