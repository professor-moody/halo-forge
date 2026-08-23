export type DatasetSourcePickerRequest = {
  kind: "file" | "folder";
  multiple?: boolean;
};

export type DatasetSourcePickerResult = {
  paths: string[];
};

export type HaloForgeDesktopBridge = {
  pickDatasetSource: (
    request: DatasetSourcePickerRequest,
  ) => Promise<DatasetSourcePickerResult | null>;
};

declare global {
  interface Window {
    haloForgeDesktop?: HaloForgeDesktopBridge;
    __TAURI__?: unknown;
    __TAURI_INTERNALS__?: unknown;
  }
}

export function isDesktopRuntime(): boolean {
  return typeof window !== "undefined" && Boolean(window.__TAURI_INTERNALS__ || window.__TAURI__);
}

async function openNativeDatasetSource(
  request: DatasetSourcePickerRequest,
): Promise<DatasetSourcePickerResult | null> {
  const { open } = await import("@tauri-apps/plugin-dialog");
  const selected = await open({
    title: request.kind === "folder" ? "Choose a dataset folder" : "Choose a dataset file",
    directory: request.kind === "folder",
    multiple: Boolean(request.multiple),
  });
  const paths = (Array.isArray(selected) ? selected : selected ? [selected] : [])
    .map(String)
    .filter(Boolean);
  return paths.length ? { paths } : null;
}

const nativeBridge: HaloForgeDesktopBridge = Object.freeze({
  pickDatasetSource: openNativeDatasetSource,
});

/**
 * Open the desktop-native dataset chooser when Halo Forge is running inside
 * Tauri. Browser callers receive null and should use their normal upload or
 * workstation-path flow. Cancellation also resolves to null.
 */
export async function pickDatasetSource(
  request: DatasetSourcePickerRequest,
): Promise<DatasetSourcePickerResult | null> {
  if (typeof window === "undefined") return null;
  if (window.haloForgeDesktop && window.haloForgeDesktop !== nativeBridge) {
    return window.haloForgeDesktop.pickDatasetSource(request);
  }
  if (!isDesktopRuntime()) return null;
  return nativeBridge.pickDatasetSource(request);
}

if (typeof window !== "undefined" && isDesktopRuntime() && !window.haloForgeDesktop) {
  window.haloForgeDesktop = nativeBridge;
}

