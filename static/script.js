document.addEventListener('DOMContentLoaded', () =>{
    document.getElementById("predict-boiler-btn").addEventListener("click", async () => {
        try {
            const response = await fetch("/predict_boiler", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json"
                },
                body: JSON.stringify({
                    "ip_address": "0.0.0.0"
                })
            });
            
            if (!response.ok) {
                throw new Error(`HTTP error! Status: ${response.status}`);
            }
            
            const data = await response.json();
            console.log("Prediction Response:", data);
            document.getElementById('boiler-prediction').innerHTML = `Boiler Run Time (in minutes): ${data.predicted_run_time_minutes}<br>Boiler Temp: ${data.predicted_boiler_temperature}<br>Predicted Energy (kW): ${data.predicted_energy_usage}`;
            document.getElementById('room-temp').innerHTML = `Room Temperature: ${data.room_temperature}`;
            document.getElementById('outside-temp').innerHTML = `Outside Temperature: ${data.outside_temperature}`;
        } catch (error) {
            console.error("Error making prediction request:", error);
        }
    });
});