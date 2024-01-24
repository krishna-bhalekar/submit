document.getElementById("myForm").addEventListener("submit", function(event) {
    event.preventDefault();

    // Collect form data
    var formData = new FormData(event.target);
    var formDataObject = {};
    formData.forEach(function(value, key){
        formDataObject[key] = value;
    });

    // Call a function to send data to Google Sheets
    sendDataToGoogleSheets(formDataObject);
});

function sendDataToGoogleSheets(formData) {
    // Use Google Sheets API to append data to your sheet
    // Example API call using fetch or XMLHttpRequest
    // Replace with your actual API endpoint and credentials
    fetch('https://docs.google.com/spreadsheets/d/1pW0lLK4nha6uKStqe_0s2zRKAF0ivy-z_YorXVTRON8/edit#gid=0', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
            // Include any necessary authentication headers
        },
        body: JSON.stringify(formData),
    })
    .then(response => response.json())
    .then(data => {
        console.log('Success:', data);
        // Optionally, redirect the user or show a success message
    })
    .catch((error) => {
        console.error('Error:', error);
        // Handle errors
    });
}
