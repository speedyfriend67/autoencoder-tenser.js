const inputData = [];
let model;
let trainingInProgress = false;
let lossValues = [];

document.getElementById('visualizeButton').addEventListener('click', visualize);
document.getElementById('resetButton').addEventListener('click', clearInput);
document.getElementById('clearButton').addEventListener('click', clearInput);
document.getElementById('epochs').addEventListener('input', function() {
  document.getElementById('epochsValue').innerText = this.value;
});

async function visualize() {
  const inputDataRaw = document.getElementById('inputData').value.trim();
  if (!inputDataRaw) {
    alert('Please enter input data.');
    return;
  }
  
  inputData.length = 0;
  inputDataRaw.split(',').forEach(value => inputData.push(parseFloat(value)));

  const encodingUnits = parseInt(document.getElementById('encodingUnits').value, 10);
  const epochs = parseInt(document.getElementById('epochs').value, 10);
  const learningRate = parseFloat(document.getElementById('learningRate').value);
  const optimizerName = document.getElementById('optimizer').value;
  const lossMetric = document.getElementById('lossMetric').value;

  if (isNaN(encodingUnits) || isNaN(epochs) || isNaN(learningRate)) {
    alert('Please enter valid numeric values for encoding units, epochs, and learning rate.');
    return;
  }

  const optimizer = optimizerName === 'adam' ? tf.train.adam(learningRate) : tf.train.sgd(learningRate);

  if (trainingInProgress) return;
  trainingInProgress = true;

  // Clear previous values
  lossValues = [];
  clearInput();

  // Define the model architecture
  model = tf.sequential();
  model.add(tf.layers.dense({ inputShape: [inputData.length], units: encodingUnits, activation: 'relu' }));
  model.add(tf.layers.dense({ units: inputData.length, activation: 'sigmoid' }));

  // Compile the model
  model.compile({ optimizer, loss: lossMetric });

  // Train the model with random data (no real training data provided)
  const trainingData = tf.randomNormal([10, inputData.length]);
  const history = await model.fit(trainingData, trainingData, {
    epochs,
    verbose: 1,
    callbacks: {
      onEpochEnd: async (epoch, logs) => {
        lossValues.push(logs.loss);
        updateLossChart();
        await tf.nextFrame();
      }
    }
  });

  trainingInProgress = false;

  // Encode and decode input data
  const encodedData = model.predict(tf.tensor2d([inputData]));
  const decodedData = model.predict(encodedData);

  // Display encoded data
  displayEncodedData(encodedData.dataSync());

  // Display decoded data
  displayDecodedData(decodedData.dataSync());

  // Calculate reconstruction error
  const error = tf.metrics.meanSquaredError(tf.tensor2d([inputData]), decodedData).dataSync()[0];

  // Display reconstruction error
  document.getElementById('error').innerText = `Reconstruction Error: ${error.toFixed(4)}`;
  document.getElementById('error').style.display = 'block';
}

function clearInput() {
  document.getElementById('inputData').value = '';
}

function updateLossChart() {
  const ctx = document.getElementById('lossChart').getContext('2d');
  const chart = new Chart(ctx, {
    type: 'line',
    data: {
      labels: Array.from(Array(lossValues.length).keys()),
      datasets: [{
        label: 'Loss',
        data: lossValues,
        borderColor: 'blue',
        borderWidth: 1,
        fill: false
      }]
    },
    options: {
      scales: {
        y: {
          beginAtZero: true
        }
      }
    }
  });
}

function displayEncodedData(data) {
  const encodedContainer = document.getElementById('encodedContainer');
  encodedContainer.innerHTML = '<p><strong>Encoded Data:</strong></p>';
  const ul = document.createElement('ul');
  data.forEach(value => {
    const li = document.createElement('li');
    li.textContent = value.toFixed(4);
    ul.appendChild(li);
  });
  encodedContainer.appendChild(ul);
}

function displayDecodedData(data) {
  const decodedContainer = document.getElementById('decodedContainer');
  decodedContainer.innerHTML = '<p><strong>Decoded Data:</strong></p>';
  const ul = document.createElement('ul');
  data.forEach(value => {
    const li = document.createElement('li');
    li.textContent = value.toFixed(4);
    ul.appendChild(li);
  });
  decodedContainer.appendChild(ul);
}


