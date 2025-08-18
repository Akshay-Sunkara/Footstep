document.addEventListener('keydown', (event) => {
  const movableDiv = document.querySelector('.overlayContainer');
  if (!movableDiv) return;

  let currentTop = parseInt(movableDiv.style.top) || 30;
  let currentLeft = parseInt(movableDiv.style.left) || 800;
  const moveAmount = 10;

  switch (event.key) {
    case 'ArrowLeft':
      movableDiv.style.left = (currentLeft - moveAmount) + 'px';
      break;
    case 'ArrowUp':
      movableDiv.style.top = (currentTop - moveAmount) + 'px';
      break;
    case 'ArrowRight':
      movableDiv.style.left = (currentLeft + moveAmount) + 'px';
      break;
    case 'ArrowDown':
      movableDiv.style.top = (currentTop + moveAmount) + 'px';
      break;
    case '/':
      win.minimize()
      break;
  }
});



