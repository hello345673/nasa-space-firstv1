// Main JavaScript for Exoplanet Hunter AI

document.addEventListener('DOMContentLoaded', function() {
    console.log('Exoplanet Hunter AI Loaded! 🚀');
    
    // Modern Hero Animations
    initHeroAnimations();
});

function initHeroAnimations() {
    // Animate chart bars on scroll
    const chartBars = document.querySelectorAll('.chart-bar');
    if (chartBars.length > 0) {
        const observer = new IntersectionObserver((entries) => {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    entry.target.style.animation = 'chartGrow 2s ease-out';
                }
            });
        }, { threshold: 0.5 });
        
        chartBars.forEach(bar => observer.observe(bar));
    }
    
            // Animate stats on scroll (skip demo section stats)
            const statValues = document.querySelectorAll('.stat-value');
            if (statValues.length > 0) {
                const observer = new IntersectionObserver((entries) => {
                    entries.forEach(entry => {
                        // Skip demo section stats - they should show real values immediately
                        if (entry.isIntersecting && !entry.target.closest('.demo-stats')) {
                            animateNumber(entry.target);
                        }
                    });
                }, { threshold: 0.5 });
                
                statValues.forEach(stat => {
                    // Don't animate demo stats
                    if (!stat.closest('.demo-stats')) {
                        observer.observe(stat);
                    }
                });
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

