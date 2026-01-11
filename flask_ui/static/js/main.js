// static/js/main.js
document.addEventListener('DOMContentLoaded', () => {
  // hide flash messages after 5s
  setTimeout(() => {
    const flashes = document.querySelectorAll('.flash-item');
    flashes.forEach(f => f.style.display = 'none');
  }, 5000);
});
