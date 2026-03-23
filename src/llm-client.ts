/**
 * LLM Client for memory extraction and dedup decisions.
 * Uses OpenAI-compatible API (reuses the embedding provider config).
 */

import OpenAI from "openai";
import {
  buildOauthEndpoint,
  extractOutputTextFromSse,
  loadOAuthSession,
  needsRefresh,
  normalizeOauthModel,
  refreshOAuthSession,
  saveOAuthSession,
} from "./llm-oauth.js";

export interface LlmClientConfig {
  apiKey?: string;
  model: string;
  baseURL?: string;
  auth?: "api-key" | "oauth";
  oauthPath?: string;
  oauthProvider?: string;
  timeoutMs?: number;
  log?: (msg: string) => void;
}

export interface LlmClient {
  /** Send a prompt and parse the JSON response. Returns null on failure. */
  completeJson<T>(prompt: string, label?: string): Promise<T | null>;
  /** Best-effort diagnostics for the most recent failure, if any. */
  getLastError(): string | null;
}

// 4F.4: strip unsafe control characters (NUL, BEL, BS, etc.) before JSON parsing
function stripControlChars(text: string): string {
  return text.replace(/[\u0000-\u0008\u000B\u000C\u000E-\u001F\u007F]/g, "");
}

/**
 * String-escape-aware balanced brace/bracket extraction.
 * Handles escaped quotes inside JSON string literals so braces within
 * strings don't break the depth counter.
 */
function extractBalancedJson(text: string, startIndex: number): string | null {
  const opener = text[startIndex];
  const closer = opener === "{" ? "}" : opener === "[" ? "]" : null;
  if (!closer) return null;

  let depth = 0;
  let inString = false;
  let escaped = false;

  for (let i = startIndex; i < text.length; i++) {
    const char = text[i];
    if (inString) {
      if (escaped) { escaped = false; continue; }
      if (char === "\\") { escaped = true; continue; }
      if (char === '"') { inString = false; }
      continue;
    }
    if (char === '"') { inString = true; continue; }
    if (char === opener) { depth++; continue; }
    if (char === closer) {
      depth--;
      if (depth === 0) return text.slice(startIndex, i + 1);
    }
  }
  return null;
}

/**
 * 4F.4: Layered JSON defense parsing for LLM responses.
 * Layer 1: Strip control chars
 * Layer 2: Direct JSON.parse
 * Layer 3: Fenced JSON extraction (```json ... ```)
 * Layer 4: Balanced brace extraction (string-escape aware)
 * Layer 5: Graceful degradation (return null)
 */
function extractJsonFromResponse(text: string): string | null {
  // Layer 1: sanitize control characters
  const cleaned = stripControlChars(text).trim();

  // Layer 2: try direct parse (fastest path for well-formed JSON)
  try {
    JSON.parse(cleaned);
    return cleaned;
  } catch {}

  // Layer 3: try markdown code fence (```json ... ``` or ``` ... ```)
  const fenceMatch = cleaned.match(/```(?:json)?\s*\n?([\s\S]*?)```/);
  if (fenceMatch) {
    const fenced = fenceMatch[1].trim();
    try {
      JSON.parse(fenced);
      return fenced;
    } catch {}
  }

  // Layer 4: balanced brace extraction (string-escape aware)
  for (let i = 0; i < cleaned.length; i++) {
    const char = cleaned[i];
    if (char !== "{" && char !== "[") continue;
    const candidate = extractBalancedJson(cleaned, i);
    if (!candidate) continue;
    try {
      JSON.parse(candidate);
      return candidate;
    } catch {
      continue;
    }
  }

  // Layer 5: graceful degradation
  return null;
}

function previewText(value: string, maxLen = 200): string {
  const normalized = value.replace(/\s+/g, " ").trim();
  if (normalized.length <= maxLen) return normalized;
  return `${normalized.slice(0, maxLen - 3)}...`;
}

export function createLlmClient(config: LlmClientConfig): LlmClient {
  const baseURL = (config.baseURL || "http://127.0.0.1:11434/v1").replace(/\/+$/, "");
  // 4F.1: default 45s, hard cap 50s
  const timeoutMs = Math.min(config.timeoutMs ?? 45000, 50000);
  const log = config.log ?? (() => {});

  return {
    async completeJson<T>(prompt: string, label = "generic"): Promise<T | null> {
      lastError = null;
      try {
        const response = await client.chat.completions.create({
          model: config.model,
          messages: [
            {
              role: "system",
              content:
                "You are a memory extraction assistant. Always respond with valid JSON only.",
            },
            { role: "user", content: prompt },
          ],
          temperature: 0.1,
        });

        const raw = response.choices?.[0]?.message?.content;
        if (!raw) {
          lastError =
            `memory-lancedb-pro: llm-client [${label}] empty response content from model ${config.model}`;
          log(lastError);
          return null;
        }
        if (typeof raw !== "string") {
          lastError =
            `memory-lancedb-pro: llm-client [${label}] non-string response content type=${Array.isArray(raw) ? "array" : typeof raw} from model ${config.model}`;
          log(lastError);
          return null;
        }

        const jsonStr = extractJsonFromResponse(raw);
        if (!jsonStr) {
          lastError =
            `memory-lancedb-pro: llm-client [${label}] no JSON object found (chars=${raw.length}, preview=${JSON.stringify(previewText(raw))})`;
          log(lastError);
          return null;
        }

        try {
          return JSON.parse(jsonStr) as T;
        } catch (err) {
          const repairedJsonStr = repairCommonJson(jsonStr);
          if (repairedJsonStr !== jsonStr) {
            try {
              const repaired = JSON.parse(repairedJsonStr) as T;
              log(
                `memory-lancedb-pro: llm-client [${label}] recovered malformed JSON via heuristic repair (jsonChars=${jsonStr.length})`,
              );
              return repaired;
            } catch (repairErr) {
              lastError =
                `memory-lancedb-pro: llm-client [${label}] JSON.parse failed: ${err instanceof Error ? err.message : String(err)}; repair failed: ${repairErr instanceof Error ? repairErr.message : String(repairErr)} (jsonChars=${jsonStr.length}, jsonPreview=${JSON.stringify(previewText(jsonStr))})`;
              log(lastError);
              return null;
            }
          }
          lastError =
            `memory-lancedb-pro: llm-client [${label}] JSON.parse failed: ${err instanceof Error ? err.message : String(err)} (jsonChars=${jsonStr.length}, jsonPreview=${JSON.stringify(previewText(jsonStr))})`;
          log(lastError);
          return null;
        }
      } catch (err) {
        lastError =
          `memory-lancedb-pro: llm-client [${label}] request failed for model ${config.model}: ${err instanceof Error ? err.message : String(err)}`;
        log(lastError);
        return null;
      }
    },
    getLastError(): string | null {
      return lastError;
    },
  };
}

function createOauthClient(config: LlmClientConfig, log: (msg: string) => void): LlmClient {
  if (!config.oauthPath) {
    throw new Error("LLM oauth mode requires llm.oauthPath");
  }

  let cachedSessionPromise: Promise<Awaited<ReturnType<typeof loadOAuthSession>>> | null = null;
  let lastError: string | null = null;

  async function getSession() {
    if (!cachedSessionPromise) {
      cachedSessionPromise = loadOAuthSession(config.oauthPath!).catch((error) => {
        cachedSessionPromise = null;
        throw error;
      });
    }
    let session = await cachedSessionPromise;
    if (needsRefresh(session)) {
      session = await refreshOAuthSession(session, config.timeoutMs);
      await saveOAuthSession(config.oauthPath!, session);
      cachedSessionPromise = Promise.resolve(session);
    }
    return session;
  }

  return {
    async completeJson<T>(prompt: string, label = "generic"): Promise<T | null> {
      lastError = null;
      try {
        const session = await getSession();
        const { signal, dispose } = createTimeoutSignal(config.timeoutMs);
        const endpoint = buildOauthEndpoint(config.baseURL, config.oauthProvider);
        try {
          const response = await fetch(endpoint, {
            method: "POST",
            headers: {
              Authorization: `Bearer ${session.accessToken}`,
              "Content-Type": "application/json",
              Accept: "text/event-stream",
              "OpenAI-Beta": "responses=experimental",
              "chatgpt-account-id": session.accountId,
              originator: "codex_cli_rs",
            },
            signal,
            body: JSON.stringify({
              model: normalizeOauthModel(config.model),
              instructions:
                "You are a memory extraction assistant. Always respond with valid JSON only.",
              input: [
                {
                  role: "user",
                  content: [
                    {
                      type: "input_text",
                      text: prompt,
                    },
                  ],
                },
              ],
              store: false,
              stream: true,
              text: {
                format: { type: "text" },
              },
            }),
          });

          if (!response.ok) {
            const detail = await response.text().catch(() => "");
            throw new Error(`HTTP ${response.status} ${response.statusText}: ${detail.slice(0, 500)}`);
          }

          const bodyText = await response.text();
          const raw = (
            response.headers.get("content-type")?.includes("text/event-stream") ||
            looksLikeSseResponse(bodyText)
          )
            ? extractOutputTextFromSse(bodyText)
            : (() => {
                try {
                  const parsed = JSON.parse(bodyText) as Record<string, unknown>;
                  const output = Array.isArray(parsed.output) ? parsed.output : [];
                  const first = output.find(
                    (item) =>
                      item &&
                      typeof item === "object" &&
                      Array.isArray((item as Record<string, unknown>).content),
                  ) as Record<string, unknown> | undefined;
                  if (!first) return null;
                  const content = (first.content as Array<Record<string, unknown>>).find(
                    (part) => part?.type === "output_text" && typeof part.text === "string",
                  );
                  return typeof content?.text === "string" ? content.text : null;
                } catch {
                  return null;
                }
              })();

          if (!raw) {
            lastError =
              `memory-lancedb-pro: llm-client [${label}] empty OAuth response content from model ${config.model}`;
            log(lastError);
            return null;
          }

          const jsonStr = extractJsonFromResponse(raw);
          if (!jsonStr) {
            lastError =
              `memory-lancedb-pro: llm-client [${label}] no JSON object found in OAuth response (chars=${raw.length}, preview=${JSON.stringify(previewText(raw))})`;
            log(lastError);
            return null;
          }

          try {
            return JSON.parse(jsonStr) as T;
          } catch (err) {
            const repairedJsonStr = repairCommonJson(jsonStr);
            if (repairedJsonStr !== jsonStr) {
              try {
                const repaired = JSON.parse(repairedJsonStr) as T;
                log(
                  `memory-lancedb-pro: llm-client [${label}] recovered malformed OAuth JSON via heuristic repair (jsonChars=${jsonStr.length})`,
                );
                return repaired;
              } catch (repairErr) {
                lastError =
                  `memory-lancedb-pro: llm-client [${label}] OAuth JSON.parse failed: ${err instanceof Error ? err.message : String(err)}; repair failed: ${repairErr instanceof Error ? repairErr.message : String(repairErr)} (jsonChars=${jsonStr.length}, jsonPreview=${JSON.stringify(previewText(jsonStr))})`;
                log(lastError);
                return null;
              }
            }
            lastError =
              `memory-lancedb-pro: llm-client [${label}] OAuth JSON.parse failed: ${err instanceof Error ? err.message : String(err)} (jsonChars=${jsonStr.length}, jsonPreview=${JSON.stringify(previewText(jsonStr))})`;
            log(lastError);
            return null;
          }
        } finally {
          dispose();
        }
      } catch (err) {
        lastError =
          `memory-lancedb-pro: llm-client [${label}] OAuth request failed for model ${config.model}: ${err instanceof Error ? err.message : String(err)}`;
        log(lastError);
        return null;
      }
    },
    getLastError(): string | null {
      return lastError;
    },
  };
}

export function createLlmClient(config: LlmClientConfig): LlmClient {
  const log = config.log ?? (() => {});
  if (config.auth === "oauth") {
    return createOauthClient(config, log);
  }
  return createApiKeyClient(config, log);
}

export { extractJsonFromResponse, repairCommonJson };
