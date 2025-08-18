const { app, BrowserWindow, globalShortcut, ipcMain } = require('electron')
const path = require('path');
const { spawn } = require('child_process');

let isInteractive = false;
let win;

function runModel() {
  const scriptPath = path.join(__dirname, 'listen.py');
  pyProcess = spawn('python3', [scriptPath]);

  pyProcess.stdout.on('data', (data) => {
    const prediction = data.toString().trim();
    console.log(`Prediction from Python: ${prediction}`);
    win.webContents.send('prediction', prediction);

  });
  pyProcess.stderr.on('data', (data) => {
    console.error('Python error:', data.toString());
  });
  
  pyProcess.on('close', (code) => {
    console.log(`Python process exited with code ${code}`);
  });
  
  pyProcess.on('error', (error) => {
    console.error('Failed to start Python process:', error);
  });
}

const createWindow = () => {
  win = new BrowserWindow({
    width: 800,
    height: 600,
    alwaysOnTop: true,
    transparent: true,
    autoHideMenuBar: true,
    fullscreen: true,
    focusable: true,
    webPreferences: {
      nodeIntegration: true,
      contextIsolation: false,
      backgroundThrottling: false 
    }
  })

  win.setIgnoreMouseEvents(true);
  win.loadFile('./Overlay/index.html')
}

app.whenReady().then(() => {
  createWindow()

  ipcMain.on('start-listen', () => {
    console.log("Listen button clicked!");
    runModel();
  });

  ipcMain.on('close', () => {
    console.log("Closing");
    app.quit()
  });
  
  globalShortcut.register('CommandOrControl+Shift+O', () => {
    if (win.isVisible()) {
      win.hide();
    } else {
      win.show();
    }
  });

  globalShortcut.register('Tab', () => {
    isInteractive = !isInteractive;

    if (isInteractive) {
      win.setIgnoreMouseEvents(false, { forward: true });
      win.focus();
    } else {
      win.setIgnoreMouseEvents(true);
    }
  });

})

