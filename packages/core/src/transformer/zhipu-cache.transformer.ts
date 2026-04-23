import { LLMProvider, UnifiedChatRequest } from "@/types/llm";
import { Transformer } from "@/types/transformer";

interface CacheEntry {
  cache_id: string;
  expiresAt: number;
  systemLength: number;
}

const TTL_MS = 7 * 24 * 60 * 60 * 1000;
const HASH_THRESHOLD = 500;

// Track which requests used cache, keyed by a request identifier
// { hash -> { cache_id, systemLength } }
const requestCacheMap = new Map<string, { cache_id: string; systemLength: number; hit: boolean }>();

export class ZhipuCacheTransformer implements Transformer {
  name = "zhipu-cache";

  private cache = new Map<string, CacheEntry>();
  private logger: any;
  private cacheBaseUrl: string;

  constructor(options?: any) {
    this.logger = options?.logger || console;
    // Default: zhipu OpenAI-compatible endpoint for cache API
    // Can be overridden via config: { "use": ["zhipu-cache"], "options": { "cacheBaseUrl": "..." } }
    this.cacheBaseUrl = options?.cacheBaseUrl || "https://open.bigmodel.cn/api/paas/v4";
  }

  private simpleHash(str: string): string {
    let hash = 0;
    for (let i = 0; i < str.length; i++) {
      const char = str.charCodeAt(i);
      hash = ((hash << 5) - hash) + char;
      hash = hash & hash;
    }
    return hash.toString(36);
  }

  private getSystemContent(request: UnifiedChatRequest): string | null {
    if (!Array.isArray(request.messages)) return null;
    for (const msg of request.messages) {
      if (msg.role === "system" && typeof msg.content === "string" && msg.content.length > HASH_THRESHOLD) {
        return msg.content;
      }
    }
    return null;
  }

  private removeSystemMessages(request: UnifiedChatRequest): void {
    if (!Array.isArray(request.messages)) return;
    request.messages = request.messages.filter((msg) => msg.role !== "system");
  }

  private async createCache(systemContent: string, provider: LLMProvider): Promise<string | null> {
    try {
      const url = `${this.cacheBaseUrl.replace(/\/$/, "")}/cache/create`;
      const apiKey = provider.apiKey || "";

      const response = await fetch(url, {
        method: "POST",
        headers: {
          "Authorization": `Bearer ${apiKey}`,
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          messages: [{ role: "system", content: systemContent }],
          ttl: 10080,
        }),
      });

      if (!response.ok) {
        this.logger.warn(`[zhipu-cache] Create cache failed: HTTP ${response.status}`);
        return null;
      }

      const data = await response.json() as { data?: { cache_id: string }; cache_id?: string };
      const cacheId = data?.data?.cache_id || data?.cache_id || null;

      if (cacheId) {
        this.logger.info(
          { cache_id: cacheId, content_length: systemContent.length, ttl: "7d" },
          `[zhipu-cache] Cache created: ${cacheId.slice(0, 16)}... (${systemContent.length} chars)`
        );
      }

      return cacheId;
    } catch (e: any) {
      this.logger.warn(`[zhipu-cache] Create cache error: ${e.message}`);
      return null;
    }
  }

  async transformRequestIn(
    request: UnifiedChatRequest,
    provider: LLMProvider,
    context?: any
  ): Promise<Record<string, unknown>> {
    const systemContent = this.getSystemContent(request);
    if (!systemContent) {
      return { body: request };
    }

    const hash = this.simpleHash(systemContent);
    const cached = this.cache.get(hash);
    const reqId = context?.req?.id || hash;

    let cacheId: string | null = null;
    let isHit = false;

    if (cached && cached.expiresAt > Date.now()) {
      cacheId = cached.cache_id;
      isHit = true;
      this.logger.info(
        { cache_id: cacheId, content_length: cached.systemLength, req_id: reqId },
        `[zhipu-cache] Cache HIT: ${cacheId.slice(0, 16)}... (${cached.systemLength} chars cached, ${Math.round((Date.now() - (cached.expiresAt - TTL_MS)) / 60000)}min ago)`
      );
    } else {
      cacheId = await this.createCache(systemContent, provider);
      if (cacheId) {
        this.cache.set(hash, { cache_id: cacheId, expiresAt: Date.now() + TTL_MS, systemLength: systemContent.length });
      }
    }

    this.removeSystemMessages(request);

    if (cacheId) {
      request.messages.unshift({
        role: "system",
        content: "",
        cache_id: cacheId,
      } as any);

      // Track this request for response injection
      requestCacheMap.set(reqId, { cache_id: cacheId, systemLength: cached?.systemLength || systemContent.length, hit: isHit });
    }

    return { body: request };
  }

  async transformResponseOut(response: Response, context?: any): Promise<Response> {
    // If this request used a cache, inject cached_tokens into the response
    // so that Claude Code can see the cache benefit
    const reqId = context?.req?.id;
    if (!reqId) return response;

    const cacheInfo = requestCacheMap.get(reqId);
    if (!cacheInfo) return response;

    requestCacheMap.delete(reqId);

    // Estimate cached tokens from system content length (rough: 1 Chinese char ≈ 1 token)
    const estimatedCachedTokens = Math.round(cacheInfo.systemLength * 0.7);

    if (!response.ok || !response.body) return response;

    // For streaming responses, we need to inject cache_read_input_tokens into message_delta events
    const contentType = response.headers.get("content-type") || "";
    if (contentType.includes("text/event-stream")) {
      const originalBody = response.body;
      const injected = estimatedCachedTokens;

      const transformStream = new TransformStream({
        transform(chunk, controller) {
          const text = new TextDecoder().decode(chunk);

          // Inject cache_read_input_tokens into message_delta usage
          if (text.includes('"usage"') && text.includes('"message_delta"')) {
            const modified = text.replace(
              /"input_tokens":(\d+),"output_tokens":(\d+)/g,
              `"input_tokens":$1,"output_tokens":$2,"cache_read_input_tokens":${injected}`
            );
            controller.enqueue(new TextEncoder().encode(modified));
            return;
          }

          controller.enqueue(chunk);
        }
      });

      const newBody = originalBody.pipeThrough(transformStream);

      return new Response(newBody, {
        status: response.status,
        statusText: response.statusText,
        headers: response.headers,
      });
    }

    return response;
  }

  /**
   * Get cache statistics (for monitoring/debugging)
   */
  getStats(): { entries: number; oldestAge: string } {
    let oldestAge = "N/A";
    let minExpiry = Infinity;
    for (const entry of this.cache.values()) {
      const age = Date.now() - (entry.expiresAt - TTL_MS);
      if (age < minExpiry) minExpiry = age;
    }
    if (minExpiry !== Infinity) {
      const minutes = Math.round(minExpiry / 60000);
      oldestAge = `${minutes}min`;
    }
    return { entries: this.cache.size, oldestAge };
  }

  /**
   * Clear all in-memory cache entries (does not delete server-side caches)
   */
  clear(): number {
    const count = this.cache.size;
    this.cache.clear();
    requestCacheMap.clear();
    return count;
  }
}
