import { existsSync, readFileSync, writeFileSync } from 'fs';
import { PID_FILE, REFERENCE_COUNT_FILE } from '@CCR/shared';
import { readConfigFile } from '.';
import find from 'find-process';
import { execSync } from 'child_process'; // 引入 execSync 来执行命令行

// Windows execSync 超时阈值（毫秒），防止 netstat/tasklist 阻塞
const EXEC_SYNC_TIMEOUT_MS = 5000;

/**
 * 带超时的安全 execSync 封装，防止 Windows 系统调用卡死
 */
function safeExecSync(cmd: string, options: { stdio: 'pipe' } & Record<string, unknown>): string {
    try {
        // encoding: 'utf-8' 确保 execSync 返回 string 而非 Buffer，否则 .split() 等字符串方法不可用
        return execSync(cmd, { ...options, timeout: EXEC_SYNC_TIMEOUT_MS, encoding: 'utf-8' });
    } catch (e: unknown) {
        // 超时或异常：返回空字符串，由调用方处理
        return '';
    }
}

export async function isProcessRunning(pid: number): Promise<boolean> {
    try {
        const processes = await find('pid', pid);
        return processes.length > 0;
    } catch (error) {
        return false;
    }
}

/**
 * 通过端口号查找正在监听该端口的进程 PID
 * 解决 PID 文件丢失时无法定位服务进程的问题
 * @param port 目标端口号
 * @returns 占用该端口的进程 PID，未找到返回 null
 */
export function findPidByPort(port: number): number | null {
    try {
        if (process.platform === 'win32') {
            // Windows: 使用 netstat -ano 查找监听指定端口的进程（带超时防止卡死）
            const output = safeExecSync(`netstat -ano`, { stdio: 'pipe' });
            if (!output) return null; // 超时或异常
            const lines = output.split('\n');
            for (const line of lines) {
                // 匹配 LISTENING 状态且绑定到目标端口的行
                // 格式: TCP    127.0.0.1:3456    0.0.0.0:0    LISTENING    12345
                if (line.includes('LISTENING')) {
                    const parts = line.trim().split(/\s+/);
                    const localAddr = parts[1]; // 如 127.0.0.1:3456 或 0.0.0.0:3456
                    if (localAddr && localAddr.endsWith(`:${port}`)) {
                        const pid = parseInt(parts[parts.length - 1], 10);
                        if (!isNaN(pid) && pid > 0) {
                            return pid;
                        }
                    }
                }
            }
        } else {
            // Linux/macOS: 使用 lsof 查找监听指定端口的进程（带超时防止卡死）
            const output = safeExecSync(`lsof -i :${port} -t -sTCP:LISTEN`, { stdio: 'pipe' });
            if (!output) return null;
            const pid = parseInt(output.trim(), 10);
            if (!isNaN(pid) && pid > 0) {
                return pid;
            }
        }
    } catch (e) {
        // netstat/lsof 执行失败或没有找到结果
    }
    return null;
}

/**
 * 获取配置中的服务端口
 * 内部使用，避免异步调用
 */
function getConfigPort(): number {
    try {
        if (existsSync(require('@CCR/shared').CONFIG_FILE)) {
            const config = JSON.parse(readFileSync(require('@CCR/shared').CONFIG_FILE, 'utf-8'));
            return config.PORT || 3456;
        }
    } catch (e) {
        // 配置读取失败，使用默认端口
    }
    return 3456;
}

/**
 * 综合查找服务进程 PID：优先 PID 文件，fallback 到端口检测
 * 找到后自动修复 PID 文件
 * @returns 服务进程 PID，未找到返回 null
 */
export function findServicePid(): number | null {
    // 第一优先级：从 PID 文件读取
    if (existsSync(PID_FILE)) {
        try {
            const pid = parseInt(readFileSync(PID_FILE, 'utf-8'), 10);
            if (!isNaN(pid) && isProcessAlive(pid)) {
                return pid;
            }
        } catch (e) {
            // PID 文件读取失败
        }
        // PID 文件中的进程已死，清理
        cleanupPidFile();
    }

    // 第二优先级：通过端口查找
    const port = getConfigPort();
    const pidByPort = findPidByPort(port);
    if (pidByPort !== null) {
        // 找到了占用端口的进程，修复 PID 文件
        savePid(pidByPort);
        return pidByPort;
    }

    return null;
}

/**
 * 检查指定 PID 的进程是否存活（同步版本，用于内部调用）
 * 使用 Node.js 原生 process.kill(pid, 0)，不依赖任何系统命令，避免 Windows tasklist 卡死
 */
function isProcessAlive(pid: number): boolean {
    try {
        // process.kill(pid, 0) 不会真正杀进程，仅检查进程是否存在
        process.kill(pid, 0);
        return true;
    } catch (e) {
        return false;
    }
}

export function incrementReferenceCount() {
    let count = 0;
    if (existsSync(REFERENCE_COUNT_FILE)) {
        count = parseInt(readFileSync(REFERENCE_COUNT_FILE, 'utf-8')) || 0;
    }
    count++;
    writeFileSync(REFERENCE_COUNT_FILE, count.toString());
}

export function decrementReferenceCount() {
    let count = 0;
    if (existsSync(REFERENCE_COUNT_FILE)) {
        count = parseInt(readFileSync(REFERENCE_COUNT_FILE, 'utf-8')) || 0;
    }
    count = Math.max(0, count - 1);
    writeFileSync(REFERENCE_COUNT_FILE, count.toString());
}

export function getReferenceCount(): number {
    if (!existsSync(REFERENCE_COUNT_FILE)) {
        return 0;
    }
    return parseInt(readFileSync(REFERENCE_COUNT_FILE, 'utf-8')) || 0;
}

export function isServiceRunning(): boolean {
    // 使用 findServicePid 统一检测：PID 文件优先，端口 fallback
    // findServicePid 内部会自动修复 PID 文件
    return findServicePid() !== null;
}

export function savePid(pid: number) {
    writeFileSync(PID_FILE, pid.toString());
}

export function cleanupPidFile() {
    if (existsSync(PID_FILE)) {
        try {
            const fs = require('fs');
            fs.unlinkSync(PID_FILE);
        } catch (e) {
            // Ignore cleanup errors
        }
    }
}

/**
 * 跨平台强制杀死进程
 * Windows 上 process.kill 对 detached 进程可能无效，使用 taskkill /F 替代
 * @param pid 目标进程 PID
 */
export function killProcess(pid: number): void {
    if (process.platform === 'win32') {
        // Windows: 使用 taskkill /F 强制终止，带超时防止卡死
        try {
            execSync(`taskkill /PID ${pid} /F`, { stdio: 'pipe', timeout: EXEC_SYNC_TIMEOUT_MS, encoding: 'utf-8' });
        } catch (e) {
            // taskkill 超时或失败，尝试用 Node.js 原生方式杀进程
            try { process.kill(pid, 'SIGKILL'); } catch (_) { /* 忽略 */ }
        }
    } else {
        // Linux/macOS: 先 SIGTERM，进程不退则 SIGKILL
        try {
            process.kill(pid, 'SIGTERM');
        } catch (e) {
            // 进程可能已退出
        }
    }
}

export function getServicePid(): number | null {
    // 使用统一检测逻辑，支持端口 fallback
    return findServicePid();
}

export async function getServiceInfo() {
    const pid = findServicePid();
    const config = await readConfigFile();
    const port = config.PORT || 3456;

    return {
        running: pid !== null,
        pid,
        port,
        endpoint: `http://127.0.0.1:${port}`,
        pidFile: PID_FILE,
        referenceCount: getReferenceCount()
    };
}

export async function closeService() {
    // Check reference count
    const referenceCount = getReferenceCount();

    // Only stop the service if reference count is 0
    if (referenceCount === 0) {
        const pid = findServicePid();
        if (pid !== null) {
            try {
                killProcess(pid);
            } catch (e) {
                // Ignore kill errors
            }
        }
    }
}
