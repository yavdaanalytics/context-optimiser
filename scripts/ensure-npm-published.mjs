#!/usr/bin/env node
/**
 * Verify (and optionally wait for) a package version on the npm registry.
 * Usage: node scripts/ensure-npm-published.mjs [--wait] [--timeout-ms 120000]
 */
import { readFileSync } from "node:fs";
import { resolve } from "node:path";

function argValue(flag, fallback) {
  const idx = process.argv.indexOf(flag);
  return idx === -1 ? fallback : (process.argv[idx + 1] ?? fallback);
}

const pkgJson = JSON.parse(readFileSync(resolve(process.cwd(), "package.json"), "utf-8"));
const name = argValue("--package", pkgJson.name);
const version = argValue("--version", pkgJson.version);
const wait = process.argv.includes("--wait");
const timeoutMs = Number(argValue("--timeout-ms", "90000"));
const pollMs = Number(argValue("--poll-ms", "5000"));

function registryUrl(pkgName, pkgVersion) {
  const pathName = pkgName.startsWith("@")
    ? `${pkgName.replace("/", "%2f")}/${pkgVersion}`
    : `${pkgName}/${pkgVersion}`;
  return `https://registry.npmjs.org/${pathName}`;
}

async function fetchPublishedVersion() {
  const res = await fetch(registryUrl(name, version), { headers: { accept: "application/json" } });
  if (res.status === 404) return null;
  if (!res.ok) throw new Error(`registry ${res.status} for ${registryUrl(name, version)}`);
  const body = await res.json();
  return typeof body.version === "string" ? body.version : null;
}

const started = Date.now();
let found = await fetchPublishedVersion();
while (!found && wait && Date.now() - started < timeoutMs) {
  process.stderr.write(`waiting for ${name}@${version} on registry...\n`);
  await new Promise((r) => setTimeout(r, pollMs));
  found = await fetchPublishedVersion();
}

const result = { ok: found === version, name, version, found, elapsed_ms: Date.now() - started };
process.stdout.write(`${JSON.stringify(result)}\n`);
if (!result.ok) {
  process.stderr.write(
    `${name}@${version} is not on the npm registry. Fix CI, push the fix, move the tag, and re-check.\n`,
  );
  process.exit(1);
}
