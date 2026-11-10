// Navigation and progress are enhancements; every document works without JavaScript.
const progress = document.querySelector('.reading-progress');
const links = [...document.querySelectorAll('.contents nav a')];
const sections = links.map(link => document.getElementById(decodeURIComponent(link.hash.slice(1))));
let scheduled = false;
function updateReadingPosition() {
  const range = document.documentElement.scrollHeight - window.innerHeight;
  progress.style.width = `${range > 0 ? Math.min(100, window.scrollY / range * 100) : 0}%`;
  let current = -1;
  sections.forEach((section, index) => {
    if (section && section.getBoundingClientRect().top <= 160) current = index;
  });
  links.forEach((link, index) => {
    link.classList.toggle('active', index === current);
    if (index === current) link.setAttribute('aria-current', 'location');
    else link.removeAttribute('aria-current');
  });
  scheduled = false;
}
function scheduleUpdate() {
  if (!scheduled) { scheduled = true; requestAnimationFrame(updateReadingPosition); }
}
window.addEventListener('scroll', scheduleUpdate, {passive: true});
window.addEventListener('resize', scheduleUpdate);
window.addEventListener('load', scheduleUpdate);
updateReadingPosition();
