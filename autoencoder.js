document.getElementById('visualizeButton').addEventListener('click', visualize);

async function visualize() {
  // Retrieve input data and parameters
  const inputDataRaw = document.getElementById('inputData').value.trim();
  if (!inputDataRaw) {
    alert('Please enter 5 numbers separated by commas.');
    return;
  }
  const inputData = inputDataRaw.split(',').map(Number);
  if (inputData.length !== 5 || inputData.some(isNaN)) {
    alert('Please enter 5 valid numbers separated by commas.');
    return;
  }
  const encodingUnits = parseInt(document.getElementById('encodingUnits').value, 10);
  const epochs = parseInt(document.getElementById('epochs').value, 10);
  const learningRate = parseFloat(document.getElementById('learningRate').value);
  const optimizerName = document.getElementById('optimizer').value;
  const lossMetric = document.getElementById('lossMetric').value;
  let lossFunction;
  if (lossMetric === 'custom') {
    const customLoss = document.getElementById('customLoss').value.trim();
    if (!customLoss) {
      alert('Please enter a custom loss function.');
      return;
    }
    try {
      eval(`lossFunction = ${customLoss}`);
    } catch (error) {
      alert('Invalid custom loss function.');
      return;
    }
  }
  const l1Regularization = document.getElementById('l1Regularization').checked;
  const l2Regularization = document.getElementById('l2Regularization').checked;
  const dropoutRate = parseFloat(document.getElementById('dropoutRate').value);
  const earlyStopping = document.getElementById('earlyStopping').checked;
  const learningRateScheduler = document.getElementById('learningRateScheduler').value;
  const batchNormalization = document.getElementById('batchNormalization').checked;

  // Construct the model architecture
  const model = tf.sequential();
  model.add(tf.layers.dense({ inputShape: [inputData.length], units: encodingUnits, activation: 'relu' }));
  if (batchNormalization) {
    model.add(tf.layers.batchNormalization());
  }
  if (l1Regularization) {
    model.add(tf.layers.dropout({ rate: dropoutRate }));
  }
  if (l2Regularization) {
    model.add(tf.layers.dropout({ rate: dropoutRate }));
  }
  model.add(tf.layers.dense({ units: inputData.length, activation: 'sigmoid' }));

  // Compile the model
  const optimizer = optimizerName === 'adam' ? tf.train.adam(learningRate) : tf.train.sgd(learningRate);
  const loss = lossMetric === 'custom' ? lossFunction : lossMetric;
  model.compile({ optimizer, loss });

  // Display model summary
  document.getElementById('modelSummary').innerText = '';
  model.summary().print();
}



  // Train the model with random data (no real training data provided)
  const trainingData = tf.randomNormal([10, inputData.length]);
  const history = await model.fit(trainingData, trainingData, {
    epochs,
    verbose: 1,
    callbacks: {
      onEpochEnd: async (epoch, logs) => {
        lossValues.push(logs.loss);
        accuracyValues.push(logs.acc);
        updateLossChart();
        updateTrainingMetrics(epoch + 1, logs.loss, logs.acc);
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

  // Visualize input data as a line chart
  visualizeLineChart(inputData);
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

function updateTrainingMetrics(epoch, loss, accuracy) {
  const trainingMetrics = document.getElementById('trainingMetrics');
  trainingMetrics.innerHTML = `<p>Epoch: ${epoch}, Loss: ${loss.toFixed(4)}, Accuracy: ${(accuracy * 100).toFixed(2)}%</p>`;
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

function visualizeLineChart(data) {
  const ctx = document.getElementById('lineChart').getContext('2d');
  const chart = new Chart(ctx, {
    type: 'line',
    data: {
      labels: ['1', '2', '3', '4', '5'],
      datasets: [{
        label: 'Input Data',
        data: data,
        borderColor: 'green',
        borderWidth: 2,
        fill: false
      }]
    },
    options: {
      scales: {
        x: {
          display: true,
          title: {
            display: true,
            text: 'Number Index'
          }
        },
        y: {
          display: true,
          title: {
            display: true,
            text: 'Number Value'
          }
        }
      }
    }
  });
}

