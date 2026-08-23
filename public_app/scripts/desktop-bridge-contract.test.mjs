import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

const appSource = (path) => readFile(new URL(`../${path}`, import.meta.url), "utf8");
const repoSource = (path) => readFile(new URL(`../../${path}`, import.meta.url), "utf8");

test("dataset picker has one stable desktop and browser contract", async () => {
  const bridge = await appSource("src/lib/desktop-bridge.ts");
  assert.match(bridge, /export function isDesktopRuntime/);
  assert.match(bridge, /export async function pickDatasetSource/);
  assert.match(bridge, /kind: "file" \| "folder"/);
  assert.match(bridge, /Promise<DatasetSourcePickerResult \| null>/);
  assert.match(bridge, /window\.haloForgeDesktop/);
  assert.match(bridge, /directory: request\.kind === "folder"/);
  assert.match(bridge, /multiple: Boolean\(request\.multiple\)/);
});

test("Tauri grants the dialog only to its loopback dashboard window", async () => {
  const capability = JSON.parse(await repoSource("apps/desktop-tauri/src-tauri/capabilities/default.json"));
  assert.deepEqual(capability.windows, ["main"]);
  assert.deepEqual(capability.remote?.urls, ["http://127.0.0.1:8765/*"]);
  assert.ok(capability.permissions.includes("dialog:allow-open"));
  assert.ok(!capability.permissions.includes("dialog:allow-save"));

  const cargo = await repoSource("apps/desktop-tauri/src-tauri/Cargo.toml");
  const rust = await repoSource("apps/desktop-tauri/src-tauri/src/main.rs");
  assert.match(cargo, /tauri-plugin-dialog = "2"/);
  assert.match(rust, /plugin\(tauri_plugin_dialog::init\(\)\)/);
});

test("dashboard bundles the matching JavaScript dialog plugin", async () => {
  const packageJson = JSON.parse(await appSource("package.json"));
  assert.match(packageJson.dependencies?.["@tauri-apps/plugin-dialog"] ?? "", /^\^2\./);
});

