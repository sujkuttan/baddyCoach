let videoFile: File | null = null;
let currentUrl: string | null = null;

export function setVideoFile(f: File | null) {
  if (currentUrl) {
    URL.revokeObjectURL(currentUrl);
    currentUrl = null;
  }
  videoFile = f;
  if (f) {
    currentUrl = URL.createObjectURL(f);
  }
}

export function getVideoObjectURL(): string | null {
  return currentUrl;
}
