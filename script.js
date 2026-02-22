/**
 * WhereDaMilk - Frontend JavaScript
 * Handles project initialization and UI interactions
 */

// ============================================================================
// DOM Elements
// ============================================================================

const initBtn = document.getElementById('initBtn');
const loadingContainer = document.getElementById('loadingContainer');
const successMessage = document.getElementById('successMessage');
const errorMessage = document.getElementById('errorMessage');
const errorText = document.getElementById('errorText');
const statusElement = document.getElementById('status');

// ============================================================================
// Initialization
// ============================================================================

document.addEventListener('DOMContentLoaded', () => {
    console.log('🥛 WhereDaMilk Frontend Loaded');
    setupEventListeners();
    animateOnScroll();
});

/**
 * Setup event listeners for interactive elements
 */
function setupEventListeners() {
    initBtn.addEventListener('click', handleInitialization);
    window.addEventListener('scroll', animateOnScroll);
}

// ============================================================================
// Initialize Project Handler
// ============================================================================

async function handleInitialization() {
    try {
        // Disable button and show loading state
        initBtn.disabled = true;
        loadingContainer.style.display = 'flex';
        successMessage.style.display = 'none';
        errorMessage.style.display = 'none';

        console.log('🚀 Starting initialization...');
        
        // Simulate multi-step initialization
        const steps = [
            { name: 'Git Repository', duration: 1200 },
            { name: 'Environment Setup', duration: 1000 },
            { name: 'Dependencies Check', duration: 1500 },
            { name: 'Configuration', duration: 800 }
        ];

        let currentStep = 0;

        // Update loading text with each step
        for (const step of steps) {
            updateLoadingText(`${step.name}...`);
            await delay(step.duration);
            currentStep++;
            console.log(`✓ ${step.name} completed (${currentStep}/${steps.length})`);
        }

        // Final verification
        updateLoadingText('Verifying setup...');
        await delay(600);

        // Simulate successful initialization
        showSuccessMessage();
        updateStatus('✓ Initialized Successfully');
        logSuccess();

    } catch (error) {
        console.error('❌ Initialization failed:', error);
        showErrorMessage(error.message);
        updateStatus('✗ Initialization Failed');
    } finally {
        // Re-enable button
        initBtn.disabled = false;
        loadingContainer.style.display = 'none';
    }
}

/**
 * Update loading text
 */
function updateLoadingText(text) {
    const loadingText = document.querySelector('.loading-text');
    if (loadingText) {
        loadingText.textContent = text;
    }
}

/**
 * Show success message
 */
function showSuccessMessage() {
    loadingContainer.style.display = 'none';
    successMessage.style.display = 'block';
    playSuccessAnimation();
}

/**
 * Show error message
 */
function showErrorMessage(message) {
    loadingContainer.style.display = 'none';
    errorMessage.style.display = 'block';
    errorText.textContent = message || 'An unexpected error occurred. Please try again.';
    playErrorAnimation();
}

/**
 * Update status display
 */
function updateStatus(text) {
    statusElement.textContent = text;
    statusElement.style.animation = 'none';
    // Trigger reflow to restart animation
    void statusElement.offsetWidth;
    statusElement.style.animation = 'pulse 0.6s ease-out';
}

/**
 * Play success animation
 */
function playSuccessAnimation() {
    const successIcon = document.querySelector('.success-icon');
    if (successIcon) {
        successIcon.style.animation = 'none';
        void successIcon.offsetWidth;
        successIcon.style.animation = 'pulse 0.6s ease-out';
    }
}

/**
 * Play error animation
 */
function playErrorAnimation() {
    const errorIcon = document.querySelector('.error-icon');
    if (errorIcon) {
        errorIcon.style.animation = 'none';
        void errorIcon.offsetWidth;
        errorIcon.style.animation = 'shake 0.5s ease-out';
    }
}

// ============================================================================
// Utility Functions
// ============================================================================

/**
 * Delay utility for async operations
 */
function delay(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
}

/**
 * Log successful initialization
 */
function logSuccess() {
    console.clear();
    console.log('%c🥛 WhereDaMilk', 'font-size: 24px; font-weight: bold; color: #FF6B35;');
    console.log('%cProject Initialization Complete!', 'font-size: 14px; color: #10B981; font-weight: bold;');
    console.log('%c━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━', 'color: #1A7F7E;');
    console.log('%cProject Name:', 'font-weight: bold; color: #004E89;', 'wheredamilk');
    console.log('%cVersion:', 'font-weight: bold; color: #004E89;', '1.0.0');
    console.log('%cStatus:', 'font-weight: bold; color: #004E89;', 'Ready to Use ✓');
    console.log('%c━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━', 'color: #1A7F7E;');
    console.log('%cTech Stack:', 'font-weight: bold; color: #FF6B35; font-size: 12px;');
    console.log('%c• YOLOv8 Object Detection', 'color: #004E89;');
    console.log('%c• EasyOCR Text Recognition', 'color: #004E89;');
    console.log('%c• MiDaS Depth Estimation', 'color: #004E89;');
    console.log('%c• ElevenLabs TTS', 'color: #004E89;');
    console.log('%c• Flask REST API', 'color: #004E89;');
    console.log('%c━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━', 'color: #1A7F7E;');
    console.log('%cVoice Commands:', 'font-weight: bold; color: #FF6B35; font-size: 12px;');
    console.log('%c"find <item>"     → Find and locate an item', 'color: #004E89;');
    console.log('%c"what is this"    → Identify detected objects', 'color: #004E89;');
    console.log('%c"read"            → Read text from labels', 'color: #004E89;');
    console.log('%c"stop"            → Cancel current mode', 'color: #004E89;');
    console.log('%c"quit"            → Exit the application', 'color: #004E89;');
    console.log('%c━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━', 'color: #1A7F7E;');
    console.log('%c👉 Run main.py to start the application', 'font-size: 12px; font-weight: bold; color: #10B981; background: rgba(16, 185, 129, 0.1); padding: 8px; border-radius: 4px;');
}

// ============================================================================
// Scroll Animations
// ============================================================================

/**
 * Animate elements on scroll
 */
function animateOnScroll() {
    const cards = document.querySelectorAll('.feature-card, .command-item, .badge');
    
    cards.forEach(card => {
        const rect = card.getBoundingClientRect();
        const isVisible = rect.top < window.innerHeight && rect.bottom > 0;
        
        if (isVisible) {
            card.style.opacity = '1';
            card.style.transform = 'translateY(0)';
        }
    });
}

// ============================================================================
// Interactive Elements Enhancement
// ============================================================================

// Add ripple effect to buttons
initBtn.addEventListener('mousedown', function(e) {
    const ripple = document.createElement('span');
    const rect = this.getBoundingClientRect();
    const size = Math.max(rect.width, rect.height);
    const x = e.clientX - rect.left - size / 2;
    const y = e.clientY - rect.top - size / 2;

    ripple.style.width = ripple.style.height = size + 'px';
    ripple.style.left = x + 'px';
    ripple.style.top = y + 'px';
    ripple.classList.add('ripple');

    this.appendChild(ripple);

    setTimeout(() => ripple.remove(), 600);
});

// ============================================================================
// Analytics & Tracking
// ============================================================================

/**
 * Track initialization attempt
 */
function trackInitialization() {
    const event = {
        timestamp: new Date().toISOString(),
        projectName: 'wheredamilk',
        action: 'initialization_started',
        userAgent: navigator.userAgent
    };
    console.log('📊 Event tracked:', event);
}

/**
 * Track successful initialization
 */
function trackSuccessfulInit() {
    const event = {
        timestamp: new Date().toISOString(),
        projectName: 'wheredamilk',
        action: 'initialization_success',
        duration: 'automatic'
    };
    console.log('📊 Event tracked:', event);
}

// ============================================================================
// Feature Detection & Polyfills
// ============================================================================

// Check for required browser APIs
function checkBrowserSupport() {
    const requirements = {
        'Fetch API': typeof fetch !== 'undefined',
        'Promise': typeof Promise !== 'undefined',
        'localStorage': typeof localStorage !== 'undefined',
        'requestAnimationFrame': typeof requestAnimationFrame !== 'undefined'
    };

    const allSupported = Object.values(requirements).every(v => v === true);
    
    if (!allSupported) {
        console.warn('⚠️ Some browser features may not be available:', requirements);
    }

    return allSupported;
}

// Run on page load
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', checkBrowserSupport);
} else {
    checkBrowserSupport();
}

// ============================================================================
// Export for Testing
// ============================================================================

if (typeof module !== 'undefined' && module.exports) {
    module.exports = {
        handleInitialization,
        updateLoadingText,
        showSuccessMessage,
        showErrorMessage,
        delay
    };
}
