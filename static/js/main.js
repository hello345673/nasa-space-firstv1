// Main JavaScript for Exoplanet Hunter AI

document.addEventListener('DOMContentLoaded', function() {
    console.log('Exoplanet Hunter AI Loaded! 🚀');
    
    // Modern Hero Animations
    initHeroAnimations();
});

function initHeroAnimations() {
    // Animate stats on scroll
    const statValues = document.querySelectorAll('.stat-value');
    if (statValues.length > 0) {
        const observer = new IntersectionObserver((entries) => {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    animateNumber(entry.target);
                }
            });
        }, { threshold: 0.5 });
        
        statValues.forEach(stat => observer.observe(stat));
    }
    
    // Parallax effect for gradient orbs
    window.addEventListener('scroll', () => {
        const scrolled = window.pageYOffset;
        const orbs = document.querySelectorAll('.gradient-orb');
        
        orbs.forEach((orb, index) => {
            const speed = 0.5 + (index * 0.1);
            orb.style.transform = `translateY(${scrolled * speed}px) rotate(-45deg)`;
        });
    });
}

function animateNumber(element) {
    const target = parseInt(element.textContent.replace(/[^\d]/g, ''));
    const duration = 2000;
    const start = performance.now();
    
    function updateNumber(currentTime) {
        const elapsed = currentTime - start;
        const progress = Math.min(elapsed / duration, 1);
        const current = Math.floor(progress * target);
        
        if (element.textContent.includes('%')) {
            element.textContent = current + '%';
        } else if (element.textContent.includes('K')) {
            element.textContent = current + 'K+';
        } else {
            element.textContent = current;
        }
        
        if (progress < 1) {
            requestAnimationFrame(updateNumber);
        }
    }
    
    requestAnimationFrame(updateNumber);
}

