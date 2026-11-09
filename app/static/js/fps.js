window.DetectionFpsIndicator = {
  update(element, processingTimeMs) {
    const duration = Number(processingTimeMs);
    if (!Number.isFinite(duration) || duration <= 0) {
      this.reset(element);
      return;
    }

    element.textContent = `FPS: ${(1000 / duration).toFixed(1)}`;
  },

  reset(element) {
    element.textContent = 'FPS: 0';
  },
};
