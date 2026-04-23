import { ChatCompletion } from "openai/resources";
import {
  LLMProvider,
  UnifiedChatRequest,
  UnifiedMessage,
  UnifiedTool,
} from "@/types/llm";
import {
  Transformer,
  TransformerContext,
  TransformerOptions,
} from "@/types/transformer";
import { v4 as uuidv4 } from "uuid";
import { getThinkLevel } from "@/utils/thinking";
import { createApiError } from "@/api/middleware";
import { formatBase64 } from "@/utils/image";

export class AnthropicTransformer implements Transformer {
  name = "Anthropic";
  endPoint = "/v1/messages";
  private useBearer: boolean;
  logger?: any;

  constructor(private readonly options?: TransformerOptions) {
    this.useBearer = this.options?.UseBearer ?? false;
  }

  async auth(request: any, provider: LLMProvider): Promise<any> {
    const headers: Record<string, string | undefined> = {};

    if (this.useBearer) {
      headers["authorization"] = `Bearer ${provider.apiKey}`;
      headers["x-api-key"] = undefined;
    } else {
      headers["x-api-key"] = provider.apiKey;
      headers["authorization"] = undefined;
    }

    return {
      body: request,
      config: {
        headers,
      },
    };
  }

  /**
   * 将 UnifiedChatRequest 转回标准 Anthropic 格式
   * 解决 bypass=false 时（多个 transformer）Unified 格式直接发给 Anthropic 兼容端点的兼容性问题
   */
  async transformRequestIn(
    request: UnifiedChatRequest,
    provider: LLMProvider,
    context: TransformerContext
  ): Promise<Record<string, any>> {
    const result: Record<string, any> = {
      model: request.model,
      max_tokens: request.max_tokens,
      stream: request.stream,
    };

    // 1. 提取 system 消息 → 顶层 system 字段（Anthropic 标准格式）
    const systemMessages: any[] = [];
    const otherMessages: UnifiedMessage[] = [];

    for (const msg of request.messages || []) {
      if (msg.role === "system") {
        // system 消息放到顶层 system 字段
        if (typeof msg.content === "string") {
          systemMessages.push({ type: "text", text: msg.content });
        } else if (Array.isArray(msg.content)) {
          for (const part of msg.content) {
            if (part.type === "text" && part.text) {
              systemMessages.push({
                type: "text",
                text: part.text,
                ...(part.cache_control ? { cache_control: part.cache_control } : {}),
              });
            }
          }
        }
      } else {
        otherMessages.push(msg);
      }
    }

    // 设置顶层 system 字段
    if (systemMessages.length === 1 && !systemMessages[0].cache_control) {
      result.system = systemMessages[0].text;
    } else if (systemMessages.length > 0) {
      result.system = systemMessages;
    }

    // 2. 转换消息格式：Unified → Anthropic
    result.messages = otherMessages.map((msg) => {
      // user 消息
      if (msg.role === "user") {
        if (typeof msg.content === "string") {
          return { role: "user", content: msg.content };
        }
        if (Array.isArray(msg.content)) {
          const parts: any[] = [];
          for (const part of msg.content) {
            if (part.type === "text") {
              parts.push({ type: "text", text: part.text });
            } else if (part.type === "image_url" && part.image_url) {
              // base64 或 URL 图片
              const url = part.image_url.url || "";
              if (url.startsWith("data:")) {
                const match = url.match(/^data:([^;]+);base64,(.+)$/);
                if (match) {
                  parts.push({
                    type: "image",
                    source: {
                      type: "base64",
                      media_type: match[1],
                      data: match[2],
                    },
                  });
                }
              } else {
                parts.push({
                  type: "image",
                  source: { type: "url", url },
                });
              }
            }
          }
          return { role: "user", content: parts };
        }
        return { role: "user", content: msg.content || "" };
      }

      // assistant 消息
      if (msg.role === "assistant") {
        const contentParts: any[] = [];

        // thinking 部分
        if (msg.thinking?.content) {
          contentParts.push({
            type: "thinking",
            thinking: msg.thinking.content,
            ...(msg.thinking.signature
              ? { signature: msg.thinking.signature }
              : {}),
          });
        }

        // 文本内容
        if (typeof msg.content === "string" && msg.content) {
          contentParts.push({ type: "text", text: msg.content });
        }

        // tool_calls 转回 Anthropic tool_use 格式
        if (msg.tool_calls?.length) {
          for (const tc of msg.tool_calls) {
            let input: any = {};
            try {
              input = JSON.parse(tc.function.arguments || "{}");
            } catch {}
            contentParts.push({
              type: "tool_use",
              id: tc.id,
              name: tc.function.name,
              input,
            });
          }
        }

        // 简单文本
        if (contentParts.length === 0 && typeof msg.content === "string") {
          return { role: "assistant", content: msg.content };
        }
        return { role: "assistant", content: contentParts };
      }

      // tool 消息 → tool_result
      if (msg.role === "tool") {
        return {
          role: "user",
          content: [
            {
              type: "tool_result",
              tool_use_id: msg.tool_call_id,
              content: typeof msg.content === "string" ? msg.content : JSON.stringify(msg.content),
            },
          ],
        };
      }

      return msg;
    });

    // 3. 转换 tools：Unified（OpenAI 格式）→ Anthropic 格式
    if (request.tools?.length) {
      result.tools = request.tools.map((tool: any) => {
        // 已经是 Anthropic 格式（有 name + input_schema）
        if (tool.name && tool.input_schema) {
          return tool;
        }
        // Unified/OpenAI 格式（有 type + function）
        if (tool.type === "function" && tool.function) {
          return {
            name: tool.function.name,
            description: tool.function.description || "",
            input_schema: tool.function.parameters || { type: "object", properties: {} },
          };
        }
        return tool;
      });
    }

    // 4. 转换 tool_choice
    if (request.tool_choice) {
      if (typeof request.tool_choice === "string") {
        // "auto" | "any" | "none" 等直接映射
        result.tool_choice = { type: request.tool_choice };
      } else if (request.tool_choice?.type === "function" && request.tool_choice?.function?.name) {
        result.tool_choice = { type: "tool", name: request.tool_choice.function.name };
      } else if (request.tool_choice?.type) {
        result.tool_choice = request.tool_choice;
      }
    }

    // 5. 转换 reasoning → thinking
    if (request.reasoning) {
      result.thinking = {
        type: request.reasoning.enabled !== false ? "enabled" : "disabled",
        budget_tokens: request.reasoning.max_tokens || 10000,
      };
    }

    return result;
  }

  async transformRequestOut(
    request: Record<string, any>
  ): Promise<UnifiedChatRequest> {
    const messages: UnifiedMessage[] = [];

    if (request.system) {
      if (typeof request.system === "string") {
        messages.push({
          role: "system",
          content: request.system,
        });
      } else if (Array.isArray(request.system) && request.system.length) {
        const textParts = request.system
          .filter((item: any) => item.type === "text" && item.text)
          .map((item: any) => ({
            type: "text" as const,
            text: item.text,
            cache_control: item.cache_control,
          }));
        messages.push({
          role: "system",
          content: textParts,
        });
      }
    }

    const requestMessages = JSON.parse(JSON.stringify(request.messages || []));

    requestMessages?.forEach((msg: any) => {
      if (msg.role === "user" || msg.role === "assistant") {
        if (typeof msg.content === "string") {
          messages.push({
            role: msg.role,
            content: msg.content,
          });
          return;
        }

        if (Array.isArray(msg.content)) {
          if (msg.role === "user") {
            const toolParts = msg.content.filter(
              (c: any) => c.type === "tool_result" && c.tool_use_id
            );
            const textAndMediaParts = msg.content.filter(
              (c: any) =>
                (c.type === "text" && c.text) ||
                (c.type === "image" && c.source)
            );

            // 将 tool_result 转为结构化 tool 消息，保留工具调用上下文
            const contentParts: any[] = [];
            const toolResultMessages: any[] = [];
            for (const tool of toolParts) {
              const resultContent =
                typeof tool.content === "string"
                  ? tool.content
                  : JSON.stringify(tool.content);
              // 保留为独立的 tool role 消息（OpenAI 格式）
              toolResultMessages.push({
                role: "tool",
                content: resultContent,
                tool_call_id: tool.tool_use_id,
              });
            }
            for (const part of textAndMediaParts) {
              if (part?.type === "image") {
                contentParts.push({
                  type: "image_url",
                  image_url: {
                    url:
                      part.source?.type === "base64"
                        ? formatBase64(
                            part.source.data,
                            part.source.media_type
                          )
                        : part.source.url,
                  },
                  media_type: part.source.media_type,
                });
              } else {
                contentParts.push(part);
              }
            }
            if (contentParts.length) {
              messages.push({
                role: "user",
                content:
                  contentParts.length === 1 && contentParts[0].type === "text"
                    ? contentParts[0].text
                    : contentParts,
              });
            }
            // tool_result 作为独立的 tool 消息插入
            for (const trm of toolResultMessages) {
              messages.push(trm);
            }
          } else if (msg.role === "assistant") {
            const contentTexts: string[] = [];

            const textParts = msg.content.filter(
              (c: any) => c.type === "text" && c.text
            );
            for (const part of textParts) {
              contentTexts.push(part.text);
            }

            // 将 tool_use 转为结构化 tool_calls，保留工具调用能力
            const toolCallParts = msg.content.filter(
              (c: any) => c.type === "tool_use" && c.id
            );
            const toolCalls: any[] = [];
            for (const tool of toolCallParts) {
              toolCalls.push({
                id: tool.id,
                type: "function",
                function: {
                  name: tool.name,
                  arguments: JSON.stringify(tool.input || {}),
                },
              });
            }

            const assistantMessage: UnifiedMessage = {
              role: "assistant",
              content: contentTexts.join("\n") || null,
            };
            // 保留结构化 tool_calls，供下游 OpenAI 格式使用
            if (toolCalls.length) {
              assistantMessage.tool_calls = toolCalls;
            }

            const thinkingPart = msg.content.find(
              (c: any) => c.type === "thinking" && c.signature
            );
            if (thinkingPart) {
              assistantMessage.thinking = {
                content: thinkingPart.thinking,
                signature: thinkingPart.signature,
              };
            }

            messages.push(assistantMessage);
          }
          return;
        }
      }
    });

    const result: UnifiedChatRequest = {
      messages,
      model: request.model,
      max_tokens: request.max_tokens,
      temperature: request.temperature,
      stream: request.stream,
      tools: request.tools?.length
        ? this.convertAnthropicToolsToUnified(request.tools)
        : undefined,
      tool_choice: request.tool_choice,
    };
    // GLM 等 Anthropic 兼容端点不支持 reasoning 参数，直接删除
    // 避免上游 API 报错导致请求失败
    // if (request.thinking) {
    //   result.reasoning = {
    //     effort: getThinkLevel(request.thinking.budget_tokens),
    //     // max_tokens: request.thinking.budget_tokens,
    //     enabled: request.thinking.type === "enabled",
    //   };
    // }
    if (request.tool_choice) {
      if (request.tool_choice.type === "tool") {
        result.tool_choice = {
          type: "function",
          function: { name: request.tool_choice.name },
        };
      } else {
        result.tool_choice = request.tool_choice.type;
      }
    }
    return result;
  }

  async transformResponseIn(
    response: Response,
    context?: TransformerContext
  ): Promise<Response> {
    const isStream = response.headers
      .get("Content-Type")
      ?.includes("text/event-stream");

    if (isStream) {
      if (!response.body) {
        throw new Error("Stream response body is null");
      }

      // 读取第一个 chunk 检测格式，然后构建新流把第一个 chunk 放回去
      const reader = response.body.getReader();
      let firstChunk: Uint8Array | undefined;
      try {
        const { value } = await reader.read();
        firstChunk = value;
      } catch {}

      const headerChunk = firstChunk
        ? new TextDecoder().decode(firstChunk)
        : "";

      // Anthropic SSE 以 "event: message_start" 或 "event: ping" 开头
      // OpenAI SSE 以 "data: {" 开头，包含 choices 字段
      const isAnthropicSSE =
        headerChunk.includes("event: message_start") ||
        headerChunk.includes("event: ping") ||
        headerChunk.includes('"type":"message_start"') ||
        headerChunk.includes('"type": "message_start"');

      // 构建一个新流，先把 firstChunk 放回去，再继续读 reader
      const replayStream = new ReadableStream({
        start: (controller) => {
          if (firstChunk) {
            controller.enqueue(firstChunk);
          }
          const pump = async () => {
            try {
              while (true) {
                const { done, value } = await reader.read();
                if (done) {
                  controller.close();
                  break;
                }
                controller.enqueue(value);
              }
            } catch (e) {
              controller.error(e);
            }
          };
          pump();
        },
        cancel: () => {
          reader.cancel().catch(() => {});
        },
      });

      if (isAnthropicSSE) {
        this.logger.debug(
          { reqId: context?.req.id },
          `Stream response is already Anthropic SSE format, pass-through`
        );
        return new Response(replayStream, {
          headers: {
            "Content-Type": "text/event-stream",
            "Cache-Control": "no-cache",
            Connection: "keep-alive",
          },
        });
      }

      // OpenAI 格式，需要转换
      const convertedStream = await this.convertOpenAIStreamToAnthropic(
        replayStream,
        context!
      );
      return new Response(convertedStream, {
        headers: {
          "Content-Type": "text/event-stream",
          "Cache-Control": "no-cache",
          Connection: "keep-alive",
        },
      });
    } else {
      const data = (await response.json()) as any;
      // 非流式：检测是否已经是 Anthropic 格式
      if (data.type === "message" && Array.isArray(data.content)) {
        this.logger.debug(
          { reqId: context?.req.id },
          `Non-stream response is already Anthropic format, pass-through`
        );
        return new Response(JSON.stringify(data), {
          headers: { "Content-Type": "application/json" },
        });
      }
      const anthropicResponse = this.convertOpenAIResponseToAnthropic(
        data,
        context!
      );
      return new Response(JSON.stringify(anthropicResponse), {
        headers: { "Content-Type": "application/json" },
      });
    }
  }

  private convertAnthropicToolsToUnified(tools: any[]): UnifiedTool[] {
    return tools.map((tool) => ({
      type: "function",
      function: {
        name: tool.name,
        description: tool.description || "",
        parameters: tool.input_schema,
      },
    }));
  }

  private async convertOpenAIStreamToAnthropic(
    openaiStream: ReadableStream,
    context: TransformerContext
  ): Promise<ReadableStream> {
    const readable = new ReadableStream({
      start: async (controller) => {
        const encoder = new TextEncoder();
        const messageId = `msg_${Date.now()}`;
        let stopReasonMessageDelta: null | Record<string, any> = null;
        let model = "unknown";
        let hasStarted = false;
        let hasTextContentStarted = false;
        let hasFinished = false;
        const toolCalls = new Map<number, any>();
        const toolCallIndexToContentBlockIndex = new Map<number, number>();
        let totalChunks = 0;
        let contentChunks = 0;
        let toolCallChunks = 0;
        let isClosed = false;
        let isThinkingStarted = false;
        let contentIndex = 0;
        let currentContentBlockIndex = -1; // Track the current content block index

        // 原子性的content block index分配函数
        const assignContentBlockIndex = (): number => {
          const currentIndex = contentIndex;
          contentIndex++;
          return currentIndex;
        };

        const safeEnqueue = (data: Uint8Array) => {
          if (!isClosed) {
            try {
              controller.enqueue(data);
              const dataStr = new TextDecoder().decode(data);
              this.logger.debug({
                reqId: context.req.id,
                data: dataStr,
                type: "send data",
              });
            } catch (error) {
              if (
                error instanceof TypeError &&
                error.message.includes("Controller is already closed")
              ) {
                isClosed = true;
              } else {
                this.logger.debug({
                  reqId: context.req.id,
                  error: error instanceof Error ? error.message : String(error),
                  type: "send data error",
                });
                throw error;
              }
            }
          }
        };

        const safeClose = () => {
          if (!isClosed) {
            try {
              // Close any remaining open content block
              if (currentContentBlockIndex >= 0) {
                const contentBlockStop = {
                  type: "content_block_stop",
                  index: currentContentBlockIndex,
                };
                safeEnqueue(
                  encoder.encode(
                    `event: content_block_stop\ndata: ${JSON.stringify(
                      contentBlockStop
                    )}\n\n`
                  )
                );
                currentContentBlockIndex = -1;
              }

              if (stopReasonMessageDelta) {
                safeEnqueue(
                  encoder.encode(
                    `event: message_delta\ndata: ${JSON.stringify(
                      stopReasonMessageDelta
                    )}\n\n`
                  )
                );
                stopReasonMessageDelta = null;
              } else {
                safeEnqueue(
                  encoder.encode(
                    `event: message_delta\ndata: ${JSON.stringify({
                      type: "message_delta",
                      delta: {
                        stop_reason: "end_turn",
                        stop_sequence: null,
                      },
                      usage: {
                        input_tokens: 0,
                        output_tokens: 0,
                        cache_read_input_tokens: 0,
                      },
                    })}\n\n`
                  )
                );
              }
              const messageStop = {
                type: "message_stop",
              };
              safeEnqueue(
                encoder.encode(
                  `event: message_stop\ndata: ${JSON.stringify(
                    messageStop
                  )}\n\n`
                )
              );
              controller.close();
              isClosed = true;
            } catch (error) {
              if (
                error instanceof TypeError &&
                error.message.includes("Controller is already closed")
              ) {
                isClosed = true;
              } else {
                throw error;
              }
            }
          }
        };

        let reader: ReadableStreamDefaultReader<Uint8Array> | null = null;

        try {
          reader = openaiStream.getReader();
          const decoder = new TextDecoder();
          let buffer = "";

          while (true) {
            if (isClosed) {
              break;
            }

            const { done, value } = await reader.read();
            if (done) break;

            buffer += decoder.decode(value, { stream: true });
            const lines = buffer.split("\n");
            buffer = lines.pop() || "";

            for (const line of lines) {
              if (isClosed || hasFinished) break;

              if (!line.startsWith("data:")) continue;
              const data = line.slice(5).trim();
              this.logger.debug({
                reqId: context.req.id,
                type: "recieved data",
                data,
              });

              if (data === "[DONE]") {
                continue;
              }

              try {
                const chunk = JSON.parse(data);
                totalChunks++;
                this.logger.debug({
                  reqId: context.req.id,
                  response: chunk,
                  tppe: "Original Response",
                });
                if (chunk.error) {
                  const errorMessage = {
                    type: "error",
                    message: {
                      type: "api_error",
                      message: JSON.stringify(chunk.error),
                    },
                  };

                  safeEnqueue(
                    encoder.encode(
                      `event: error\ndata: ${JSON.stringify(errorMessage)}\n\n`
                    )
                  );
                  continue;
                }

                model = chunk.model || model;

                if (!hasStarted && !isClosed && !hasFinished) {
                  hasStarted = true;

                  const messageStart = {
                    type: "message_start",
                    message: {
                      id: messageId,
                      type: "message",
                      role: "assistant",
                      content: [],
                      model: model,
                      stop_reason: null,
                      stop_sequence: null,
                      usage: {
                        input_tokens: 0,
                        output_tokens: 0,
                      },
                    },
                  };

                  safeEnqueue(
                    encoder.encode(
                      `event: message_start\ndata: ${JSON.stringify(
                        messageStart
                      )}\n\n`
                    )
                  );
                }

                const choice = chunk.choices?.[0];
                if (chunk.usage) {
                  if (!stopReasonMessageDelta) {
                    stopReasonMessageDelta = {
                      type: "message_delta",
                      delta: {
                        stop_reason: "end_turn",
                        stop_sequence: null,
                      },
                      usage: {
                        input_tokens:
                          (chunk.usage?.prompt_tokens || 0) -
                          (chunk.usage?.prompt_tokens_details?.cached_tokens ||
                            0),
                        output_tokens: chunk.usage?.completion_tokens || 0,
                        cache_read_input_tokens:
                          chunk.usage?.prompt_tokens_details?.cached_tokens ||
                          0,
                      },
                    };
                  } else {
                    stopReasonMessageDelta.usage = {
                      input_tokens:
                        (chunk.usage?.prompt_tokens || 0) -
                        (chunk.usage?.prompt_tokens_details?.cached_tokens ||
                          0),
                      output_tokens: chunk.usage?.completion_tokens || 0,
                      cache_read_input_tokens:
                        chunk.usage?.prompt_tokens_details?.cached_tokens || 0,
                    };
                  }
                }
                if (!choice) {
                  continue;
                }

                if (choice?.delta?.thinking && !isClosed && !hasFinished) {
                  // Close any previous content block if open
                  // if (currentContentBlockIndex >= 0) {
                  //   const contentBlockStop = {
                  //     type: "content_block_stop",
                  //     index: currentContentBlockIndex,
                  //   };
                  //   safeEnqueue(
                  //     encoder.encode(
                  //       `event: content_block_stop\ndata: ${JSON.stringify(
                  //         contentBlockStop
                  //       )}\n\n`
                  //     )
                  //   );
                  //   currentContentBlockIndex = -1;
                  // }

                  if (!isThinkingStarted) {
                    const thinkingBlockIndex = assignContentBlockIndex();
                    const contentBlockStart = {
                      type: "content_block_start",
                      index: thinkingBlockIndex,
                      content_block: { type: "thinking", thinking: "" },
                    };
                    safeEnqueue(
                      encoder.encode(
                        `event: content_block_start\ndata: ${JSON.stringify(
                          contentBlockStart
                        )}\n\n`
                      )
                    );
                    currentContentBlockIndex = thinkingBlockIndex;
                    isThinkingStarted = true;
                  }
                  if (choice.delta.thinking.signature) {
                    const thinkingSignature = {
                      type: "content_block_delta",
                      index: currentContentBlockIndex,
                      delta: {
                        type: "signature_delta",
                        signature: choice.delta.thinking.signature,
                      },
                    };
                    safeEnqueue(
                      encoder.encode(
                        `event: content_block_delta\ndata: ${JSON.stringify(
                          thinkingSignature
                        )}\n\n`
                      )
                    );
                    const contentBlockStop = {
                      type: "content_block_stop",
                      index: currentContentBlockIndex,
                    };
                    safeEnqueue(
                      encoder.encode(
                        `event: content_block_stop\ndata: ${JSON.stringify(
                          contentBlockStop
                        )}\n\n`
                      )
                    );
                    currentContentBlockIndex = -1;
                  } else if (choice.delta.thinking.content) {
                    const thinkingChunk = {
                      type: "content_block_delta",
                      index: currentContentBlockIndex,
                      delta: {
                        type: "thinking_delta",
                        thinking: choice.delta.thinking.content || "",
                      },
                    };
                    safeEnqueue(
                      encoder.encode(
                        `event: content_block_delta\ndata: ${JSON.stringify(
                          thinkingChunk
                        )}\n\n`
                      )
                    );
                  }
                }

                if (choice?.delta?.content && !isClosed && !hasFinished) {
                  contentChunks++;

                  // Close any previous content block if open and it's not a text content block
                  if (currentContentBlockIndex >= 0) {
                    // Check if current content block is text type
                    const isCurrentTextBlock = hasTextContentStarted;
                    if (!isCurrentTextBlock) {
                      const contentBlockStop = {
                        type: "content_block_stop",
                        index: currentContentBlockIndex,
                      };
                      safeEnqueue(
                        encoder.encode(
                          `event: content_block_stop\ndata: ${JSON.stringify(
                            contentBlockStop
                          )}\n\n`
                        )
                      );
                      currentContentBlockIndex = -1;
                    }
                  }

                  if (!hasTextContentStarted && !hasFinished) {
                    hasTextContentStarted = true;
                    const textBlockIndex = assignContentBlockIndex();
                    const contentBlockStart = {
                      type: "content_block_start",
                      index: textBlockIndex,
                      content_block: {
                        type: "text",
                        text: "",
                      },
                    };
                    safeEnqueue(
                      encoder.encode(
                        `event: content_block_start\ndata: ${JSON.stringify(
                          contentBlockStart
                        )}\n\n`
                      )
                    );
                    currentContentBlockIndex = textBlockIndex;
                  }

                  if (!isClosed && !hasFinished) {
                    const anthropicChunk = {
                      type: "content_block_delta",
                      index: currentContentBlockIndex, // Use current content block index
                      delta: {
                        type: "text_delta",
                        text: choice.delta.content,
                      },
                    };
                    safeEnqueue(
                      encoder.encode(
                        `event: content_block_delta\ndata: ${JSON.stringify(
                          anthropicChunk
                        )}\n\n`
                      )
                    );
                  }
                }

                if (
                  choice?.delta?.annotations?.length &&
                  !isClosed &&
                  !hasFinished
                ) {
                  // Close text content block if open
                  if (currentContentBlockIndex >= 0 && hasTextContentStarted) {
                    const contentBlockStop = {
                      type: "content_block_stop",
                      index: currentContentBlockIndex,
                    };
                    safeEnqueue(
                      encoder.encode(
                        `event: content_block_stop\ndata: ${JSON.stringify(
                          contentBlockStop
                        )}\n\n`
                      )
                    );
                    currentContentBlockIndex = -1;
                    hasTextContentStarted = false;
                  }

                  choice?.delta?.annotations.forEach((annotation: any) => {
                    const annotationBlockIndex = assignContentBlockIndex();
                    const contentBlockStart = {
                      type: "content_block_start",
                      index: annotationBlockIndex,
                      content_block: {
                        type: "web_search_tool_result",
                        tool_use_id: `srvtoolu_${uuidv4()}`,
                        content: [
                          {
                            type: "web_search_result",
                            title: annotation.url_citation.title,
                            url: annotation.url_citation.url,
                          },
                        ],
                      },
                    };
                    safeEnqueue(
                      encoder.encode(
                        `event: content_block_start\ndata: ${JSON.stringify(
                          contentBlockStart
                        )}\n\n`
                      )
                    );

                    const contentBlockStop = {
                      type: "content_block_stop",
                      index: annotationBlockIndex,
                    };
                    safeEnqueue(
                      encoder.encode(
                        `event: content_block_stop\ndata: ${JSON.stringify(
                          contentBlockStop
                        )}\n\n`
                      )
                    );
                    currentContentBlockIndex = -1;
                  });
                }

                if (choice?.delta?.tool_calls && !isClosed && !hasFinished) {
                  toolCallChunks++;
                  const processedInThisChunk = new Set<number>();

                  for (const toolCall of choice.delta.tool_calls) {
                    if (isClosed) break;
                    const toolCallIndex = toolCall.index ?? 0;
                    if (processedInThisChunk.has(toolCallIndex)) {
                      continue;
                    }
                    processedInThisChunk.add(toolCallIndex);
                    const isUnknownIndex =
                      !toolCallIndexToContentBlockIndex.has(toolCallIndex);

                    if (isUnknownIndex) {
                      // Close any previous content block if open
                      if (currentContentBlockIndex >= 0) {
                        const contentBlockStop = {
                          type: "content_block_stop",
                          index: currentContentBlockIndex,
                        };
                        safeEnqueue(
                          encoder.encode(
                            `event: content_block_stop\ndata: ${JSON.stringify(
                              contentBlockStop
                            )}\n\n`
                          )
                        );
                        currentContentBlockIndex = -1;
                      }

                      const newContentBlockIndex = assignContentBlockIndex();
                      toolCallIndexToContentBlockIndex.set(
                        toolCallIndex,
                        newContentBlockIndex
                      );
                      const toolCallId =
                        toolCall.id || `call_${Date.now()}_${toolCallIndex}`;
                      const toolCallName =
                        toolCall.function?.name || `tool_${toolCallIndex}`;
                      const contentBlockStart = {
                        type: "content_block_start",
                        index: newContentBlockIndex,
                        content_block: {
                          type: "tool_use",
                          id: toolCallId,
                          name: toolCallName,
                          input: {},
                        },
                      };

                      safeEnqueue(
                        encoder.encode(
                          `event: content_block_start\ndata: ${JSON.stringify(
                            contentBlockStart
                          )}\n\n`
                        )
                      );
                      currentContentBlockIndex = newContentBlockIndex;

                      const toolCallInfo = {
                        id: toolCallId,
                        name: toolCallName,
                        arguments: "",
                        contentBlockIndex: newContentBlockIndex,
                      };
                      toolCalls.set(toolCallIndex, toolCallInfo);
                    } else if (toolCall.id && toolCall.function?.name) {
                      const existingToolCall = toolCalls.get(toolCallIndex)!;
                      const wasTemporary =
                        existingToolCall.id.startsWith("call_") &&
                        existingToolCall.name.startsWith("tool_");

                      if (wasTemporary) {
                        existingToolCall.id = toolCall.id;
                        existingToolCall.name = toolCall.function.name;
                      }
                    }

                    if (
                      toolCall.function?.arguments &&
                      !isClosed &&
                      !hasFinished
                    ) {
                      const blockIndex =
                        toolCallIndexToContentBlockIndex.get(toolCallIndex);
                      if (blockIndex === undefined) {
                        continue;
                      }
                      const currentToolCall = toolCalls.get(toolCallIndex);
                      if (currentToolCall) {
                        currentToolCall.arguments +=
                          toolCall.function.arguments;
                      }

                      try {
                        const anthropicChunk = {
                          type: "content_block_delta",
                          index: blockIndex,
                          delta: {
                            type: "input_json_delta",
                            partial_json: toolCall.function.arguments,
                          },
                        };
                        safeEnqueue(
                          encoder.encode(
                            `event: content_block_delta\ndata: ${JSON.stringify(
                              anthropicChunk
                            )}\n\n`
                          )
                        );
                      } catch {
                        try {
                          const fixedArgument = toolCall.function.arguments
                            .replace(/[\x00-\x1F\x7F-\x9F]/g, "")
                            .replace(/\\/g, "\\\\")
                            .replace(/"/g, '\\"');

                          const fixedChunk = {
                            type: "content_block_delta",
                            index: blockIndex, // Use the correct content block index
                            delta: {
                              type: "input_json_delta",
                              partial_json: fixedArgument,
                            },
                          };
                          safeEnqueue(
                            encoder.encode(
                              `event: content_block_delta\ndata: ${JSON.stringify(
                                fixedChunk
                              )}\n\n`
                            )
                          );
                        } catch (fixError) {
                          console.error(fixError);
                        }
                      }
                    }
                  }
                }

                if (choice?.finish_reason && !isClosed && !hasFinished) {
                  if (contentChunks === 0 && toolCallChunks === 0) {
                    console.error(
                      "Warning: No content in the stream response!"
                    );
                  }

                  // Close any remaining open content block
                  if (currentContentBlockIndex >= 0) {
                    const contentBlockStop = {
                      type: "content_block_stop",
                      index: currentContentBlockIndex,
                    };
                    safeEnqueue(
                      encoder.encode(
                        `event: content_block_stop\ndata: ${JSON.stringify(
                          contentBlockStop
                        )}\n\n`
                      )
                    );
                    currentContentBlockIndex = -1;
                  }

                  if (!isClosed) {
                    const stopReasonMapping: Record<string, string> = {
                      stop: "end_turn",
                      length: "max_tokens",
                      tool_calls: "tool_use",
                      content_filter: "stop_sequence",
                    };

                    const anthropicStopReason =
                      stopReasonMapping[choice.finish_reason] || "end_turn";

                    stopReasonMessageDelta = {
                      type: "message_delta",
                      delta: {
                        stop_reason: anthropicStopReason,
                        stop_sequence: null,
                      },
                      usage: {
                        input_tokens:
                          (chunk.usage?.prompt_tokens || 0) -
                          (chunk.usage?.prompt_tokens_details?.cached_tokens ||
                            0),
                        output_tokens: chunk.usage?.completion_tokens || 0,
                        cache_read_input_tokens:
                          chunk.usage?.prompt_tokens_details?.cached_tokens ||
                          0,
                      },
                    };
                  }

                  break;
                }
              } catch (parseError: any) {
                this.logger?.error(
                  `parseError: ${parseError.name} message: ${parseError.message} stack: ${parseError.stack} data: ${data}`
                );
              }
            }
          }
          safeClose();
        } catch (error) {
          if (!isClosed) {
            try {
              controller.error(error);
            } catch (controllerError) {
              console.error(controllerError);
            }
          }
        } finally {
          if (reader) {
            try {
              reader.releaseLock();
            } catch (releaseError) {
              console.error(releaseError);
            }
          }
        }
      },
      cancel: (reason) => {
        this.logger.debug(
          {
            reqId: context.req.id,
          },
          `cancle stream: ${reason}`
        );
      },
    });

    return readable;
  }

  private convertOpenAIResponseToAnthropic(
    openaiResponse: ChatCompletion,
    context: TransformerContext
  ): any {
    this.logger.debug(
      {
        reqId: context.req.id,
        response: openaiResponse,
      },
      `Original OpenAI response`
    );
    // 如果响应已经是 Anthropic 格式（如 GLM 的 Anthropic 兼容端点），直接透传
    if (
      (openaiResponse as any).type === "message" &&
      Array.isArray((openaiResponse as any).content)
    ) {
      this.logger.debug(
        { reqId: context.req.id },
        `Response is already Anthropic format, pass-through`
      );
      return openaiResponse;
    }
    try {
      const choice = openaiResponse.choices[0];
      if (!choice) {
        throw new Error("No choices found in OpenAI response");
      }
      const content: any[] = [];
      if (choice.message.annotations) {
        const id = `srvtoolu_${uuidv4()}`;
        content.push({
          type: "server_tool_use",
          id,
          name: "web_search",
          input: {
            query: "",
          },
        });
        content.push({
          type: "web_search_tool_result",
          tool_use_id: id,
          content: choice.message.annotations.map((item) => {
            return {
              type: "web_search_result",
              url: item.url_citation.url,
              title: item.url_citation.title,
            };
          }),
        });
      }
      if (choice.message.content) {
        content.push({
          type: "text",
          text: choice.message.content,
        });
      }
      if (choice.message.tool_calls && choice.message.tool_calls.length > 0) {
        choice.message.tool_calls.forEach((toolCall) => {
          let parsedInput = {};
          try {
            const argumentsStr = toolCall.function.arguments || "{}";

            if (typeof argumentsStr === "object") {
              parsedInput = argumentsStr;
            } else if (typeof argumentsStr === "string") {
              parsedInput = JSON.parse(argumentsStr);
            }
          } catch {
            parsedInput = { text: toolCall.function.arguments || "" };
          }

          content.push({
            type: "tool_use",
            id: toolCall.id,
            name: toolCall.function.name,
            input: parsedInput,
          });
        });
      }
      if ((choice.message as any)?.thinking?.content) {
        content.push({
          type: "thinking",
          thinking: (choice.message as any).thinking.content,
          signature: (choice.message as any).thinking.signature,
        });
      }
      const result = {
        id: openaiResponse.id,
        type: "message",
        role: "assistant",
        model: openaiResponse.model,
        content: content,
        stop_reason:
          choice.finish_reason === "stop"
            ? "end_turn"
            : choice.finish_reason === "length"
            ? "max_tokens"
            : choice.finish_reason === "tool_calls"
            ? "tool_use"
            : choice.finish_reason === "content_filter"
            ? "stop_sequence"
            : "end_turn",
        stop_sequence: null,
        usage: {
          input_tokens:
            (openaiResponse.usage?.prompt_tokens || 0) -
            (openaiResponse.usage?.prompt_tokens_details?.cached_tokens || 0),
          output_tokens: openaiResponse.usage?.completion_tokens || 0,
          cache_read_input_tokens:
            openaiResponse.usage?.prompt_tokens_details?.cached_tokens || 0,
        },
      };
      this.logger.debug(
        {
          reqId: context.req.id,
          result,
        },
        `Conversion complete, final Anthropic response`
      );
      return result;
    } catch {
      throw createApiError(
        `Provider error: ${JSON.stringify(openaiResponse)}`,
        500,
        "provider_error"
      );
    }
  }
}
