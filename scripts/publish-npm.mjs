#!/usr/bin/env node
/**
 * npm publish wrapper that classifies common failures for automation.
 * Usage (CI): node scripts/publish-npm.mjs --access public --provenance
 */
import { spawnSync } from "node:child_process";
import { readFileSync } from "node:fs";
import { resolve } from "node:path";

const npmCmd = process.platform === "win32" ? "npm.cmd" : "npm";
const pkg = JSON.parse(readFileSync(resolve(process.cwd(), "package.json"), "utf-8"));
const leaf = pkg.name.split("/").pop();
const scope = pkg.name.startsWith("@") ? pkg.name.split("/")[0].slice(1) : undefined;

const args = ["publish", ...process.argv.slice(2)];
const result = spawnSync(npmCmd, args, {
  encoding: "utf8",
  stdio: ["ignore", "pipe", "pipe"],
  env: process.env,
  shell: process.platform === "win32",
});

const stdout = result.stdout ?? "";
const stderr = result.stderr ?? "";
if (stdout) process.stdout.write(stdout);
if (stderr) process.stderr.write(stderr);
const combined = `${stdout}\n${stderr}`;
const code = result.status ?? 1;

if (code === 0) {
  process.stdout.write(`published ${pkg.name}@${pkg.version}\n`);
  process.exit(0);
}
if (/bypass 2fa|Two-factor authentication/i.test(combined)) {
  process.stderr.write(
    [
      "",
      `publish blocked for ${pkg.name}@${pkg.version}: npm requires a granular access token with "Bypass 2FA" (or interactive 2FA).`,
      "Programmatic fix:",
      "  1. https://www.npmjs.com/settings/~/tokens → Granular Access Token",
      `  2. Packages: Read and write (scope ${scope ? "@" + scope : "<scope>"}), enable Bypass two-factor authentication`,
      `  3. gh secret set NPM_TOKEN -R <org>/${leaf}`,
      "  4. gh workflow run release-tag.yml --ref v" + pkg.version,
      "  5. npm run release:ensure-published -- --wait",
      "",
    ].join("\n"),
  );
  process.exit(403);
}
if (/ENEEDAUTH|need auth/i.test(combined)) {
  process.stderr.write(
    "publish blocked: missing registry auth. In CI, setup-node must set registry-url and NODE_AUTH_TOKEN from secrets.NPM_TOKEN.\n",
  );
  process.exit(401);
}
if (/EPUBLISHCONFLICT|cannot publish over/i.test(combined)) {
  process.stderr.write(
    `${pkg.name}@${pkg.version} already exists on the registry. Bump package.json version (and tag) instead of republishing.\n`,
  );
  process.exit(409);
}
process.exit(code);
