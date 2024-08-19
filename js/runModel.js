const { spawn } = require('child_process');

// Dữ liệu cần truyền
const data_to_pass_in = { message: 'Hello from Node.js!' };

// Chuyển đổi dữ liệu thành chuỗi JSON
const dataString = JSON.stringify(data_to_pass_in);

// Khởi tạo quá trình con Python
const pythonProcess = spawn('python', ['../python/main.py', dataString]);

// Xử lý dữ liệu đầu ra từ Python
pythonProcess.stdout.on('data', (data) => {
  const output = data.toString().trim();
  console.log('Data received from Python:', output);

  // Phân tích chuỗi JSON từ Python
  try {
    const parsedOutput = JSON.parse(output);
    console.log('Parsed output:', parsedOutput);
  } catch (error) {
    console.error('Error parsing JSON:', error);
  }
});

// Xử lý lỗi từ Python
pythonProcess.stderr.on('data', (data) => {
  console.error('Error from Python:', data.toString());
});

// Xử lý khi quá trình Python kết thúc
pythonProcess.on('close', (code) => {
  console.log(`Python process exited with code ${code}`);
});