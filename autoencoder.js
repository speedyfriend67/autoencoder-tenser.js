let model;
let trainingInProgress = false;
let lossValues = [];

async function visualize() {
  const inputData = document.getElementById('inputData').value.split(',').map(Number);
  const encodingUnits = parseInt(document.getElementById('encodingUnits').value, 10);
  const epochs = parseInt(document.getElementById('epochs').value, 10);
  const learningRate = parseFloat(document.getElementById('learningRate').value);

  // Prevent starting multiple training sessions simultaneously
  if (trainingInProgress) return;
  trainingInProgress = true;

  // Define the model architecture
  model = tf.sequential();
  model.add(tf.layers.dense({ inputShape: [inputData.length], units: encodingUnits, activation: 'relu' }));
  model.add(tf.layers.dense({ units: inputData.length, activation: 'sigmoid' }));

  // Compile the model
  const optimizer = tf.train.adam(learningRate);
  model.compile({ optimizer, loss: 'meanSquaredError' });

  // Train the model with random data (no real training data provided)
  const trainingData = tf.randomNormal([10, inputData.length]);
  const history = await model.fit(trainingData, trainingData, {
    epochs,
    verbose: 0,
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

  // Calculate reconstruction error
  const error = tf.metrics.meanSquaredError(tf.tensor2d([inputData]), decodedData).dataSync()[0];

  // Visualize encoded data
  const canvas = document.getElementById('canvas');
  const ctx = canvas.getContext('2d');
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  const encodedValues = encodedData.dataSync();
  ctx.fillStyle = 'blue';
  ctx.fillRect(0, 0, encodedValues[0] * 100, encodedValues[1] * 100);

  // Visualize decoded data
  const decodedCanvas = document.getElementById('decodedCanvas');
  const decodedCtx = decodedCanvas.getContext('2d');
  decodedCtx.clearRect(0, 0, decodedCanvas.width, decodedCanvas.height);
  const decodedValues = decodedData.dataSync();
  const imageData = decodedCtx.createImageData(decodedCanvas.width, decodedCanvas.height);
  for (let i = 0; i < decodedValues.length; i++) {
    const value = Math.round(decodedValues[i] * 255);
    imageData.data[i * 4] = value;
    imageData.data[i * 4 + 1] = value;
    imageData.data[i * 4 + 2] = value;
    imageData.data[i * 4 + 3] = 255;
  }
  decodedCtx.putImageData(imageData, 0, 0);

  // Display reconstruction error
  const errorElement = document.getElementById('error');
  errorElement.innerText = `Reconstruction Error: ${error.toFixed(4)}`;
  errorElement.style.display = 'none';
}

function resetCanvas() {
  const canvas = document.getElementById('canvas');
  const ctx = canvas.getContext('2d');
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  const decodedCanvas = document.getElementById('decodedCanvas');
  const decodedCtx = decodedCanvas.getContext('2d');
  decodedCtx.clearRect(0, 0, decodedCanvas.width, decodedCanvas.height);

  document.getElementById('error').style.display = 'none';
}

function clearInput() {
  document.getElementById('inputData').value = '';
}

function toggleEncoded() {
  const canvas = document.getElementById('canvas');
  canvas.style.display = canvas.style.display === 'none' ? 'block' : 'none';
}

function toggleDecoded() {
  const decodedCanvas = document.getElementById('decodedCanvas');
  decodedCanvas.style.display = decodedCanvas.style.display === 'none' ? 'block' : 'none';
}

function toggleError() {
  const errorElement = document.getElementById('error');
  errorElement.style.display = errorElement.style.display === 'none' ? 'block' : 'none';
}

function updateLossChart() {
  const lossCanvas = document.getElementById('lossChart');
  const ctx = lossCanvas.getContext('2d');
  const scaleFactor = 50;
  ctx.clearRect(0, 0, lossCanvas.width, lossCanvas.height);
  ctx.beginPath();
  ctx.moveTo(0, lossCanvas.height - lossValues[0] * scaleFactor);
  for (let i = 1; i < lossValues.length; i++) {
    ctx.lineTo(i * 10, lossCanvas.height - lossValues[i] * scaleFactor);
  }
  ctx.strokeStyle = 'red';
  ctx.stroke();
}

document.getElementById('visualizeButton').addEventListener('click', visualize);
document.getElementById('resetButton').addEventListener('click', resetCanvas);
document.getElementById('clearButton').addEventListener('click', clearInput);
document.getElementById('toggleEncodedButton').addEventListener('click', toggleEncoded);
document.getElementById('toggleDecodedButton').addEventListener('click', toggleDecoded);
document.getElementById('toggleErrorButton').addEventListener('click', toggleError);
document.getElementById('epochs').addEventListener('input', function() {
  document.getElementById('epochsValue').innerText = this.value;
});
document.getElementById('trainPauseResumeButton').addEventListener('click', function() {
  if (trainingInProgress) {
    model.stopTraining = true;
    this.innerText = 'Resume Training';
  } else {
    model.stopTraining = false;
    visualize();
    this.innerText = 'Pause Training';
  }
});
