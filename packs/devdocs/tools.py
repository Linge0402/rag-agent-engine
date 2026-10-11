"""技术文档领域工具：版本兼容查询 + 代码沙箱执行。

- check_version_compat：回答"X 库 N 版本依赖什么/兼容什么"——
  数据来自 compat.yaml（真实包元数据），不靠模型记忆（版本号是
  模型最容易幻觉的内容之一，必须走数据）；
- run_python：执行用户给的 Python 片段并返回输出——"文档里的示例
  能不能跑通"当场验证。防护为演示级：超时 + 输出上限 + 简单危险
  模式拦截；生产环境应替换为容器/gVisor/nsjail 级隔离（见 docstring）。
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import yaml
from pydantic import BaseModel, Field

from core.agent.tools import QueryContext, ToolResult

COMPAT_PATH = Path(__file__).resolve().parent / "compat.yaml"

# ---------------------------------------------------------------------------
# 版本兼容查询
# ---------------------------------------------------------------------------


class VersionCompatArgs(BaseModel):
    library: str = Field(description="库名，如 langgraph")
    version: str = Field(default="", description="版本号，如 1.2.14；留空则列出已知版本")


class VersionCompatTool:
    """查询库版本的依赖与兼容要求。"""

    name = "check_version_compat"
    description = (
        "查询某个库指定版本的依赖约束与 Python 版本要求。"
        "当用户问「langgraph 1.x 需要什么版本的 langchain/pydantic」"
        "「这个版本兼容吗」类问题时使用——版本信息以数据为准，不要凭记忆回答。"
    )
    args_schema = VersionCompatArgs

    def __init__(self, compat_path: Path = COMPAT_PATH) -> None:
        self._data: dict = yaml.safe_load(compat_path.read_text(encoding="utf-8")) or {}

    async def run(self, args: BaseModel, ctx: QueryContext) -> ToolResult:
        assert isinstance(args, VersionCompatArgs)
        lib = self._data.get(args.library)
        if lib is None:
            return ToolResult(error=f"暂无库「{args.library}」的兼容数据")
        if not args.version:
            return ToolResult(
                data={"library": args.library, "known_versions": sorted(lib.keys())}
            )
        info = lib.get(args.version)
        if info is None:
            return ToolResult(
                error=f"暂无 {args.library} {args.version} 的数据；已知版本：{sorted(lib.keys())}"
            )
        return ToolResult(data={"library": args.library, "version": args.version, **info})


# ---------------------------------------------------------------------------
# 代码沙箱执行
# ---------------------------------------------------------------------------

_TIMEOUT_SECONDS = 8
_MAX_OUTPUT_CHARS = 4000

# 演示级危险模式拦截：只能挡住"明显"的危险，挡不住有意的绕过。
# 生产环境必须换成真正的隔离（容器/gVisor/nsjail），本字段仅为纵深防御的一层。
_BLOCKED_PATTERNS = re.compile(
    r"\b(os\.system|subprocess|shutil\.rmtree|__import__\s*\(\s*['\"]os|open\s*\(.+['\"]w)"
)


class RunPythonArgs(BaseModel):
    code: str = Field(description="要执行的 Python 代码片段")


class RunPythonTool:
    """执行 Python 片段，返回标准输出/错误。

    防护边界（如实说明）：超时 8 秒、输出 4000 字符上限、危险调用模式
    拦截；与宿主机同 Python 环境、无文件系统与网络隔离。
    仅供本机演示"示例代码当场验证"，不可暴露给不可信输入。
    """

    name = "run_python"
    description = (
        "执行一段 Python 代码并返回输出。当用户想验证文档中的示例代码、"
        "或问「这段代码运行结果是什么」时使用。代码会在超时后被强制终止。"
    )
    args_schema = RunPythonArgs

    async def run(self, args: BaseModel, ctx: QueryContext) -> ToolResult:
        assert isinstance(args, RunPythonArgs)
        if _BLOCKED_PATTERNS.search(args.code):
            return ToolResult(error="代码包含被拦截的危险调用模式（演示级防护）")
        try:
            proc = subprocess.run(
                [sys.executable, "-c", args.code],
                capture_output=True,
                text=True,
                timeout=_TIMEOUT_SECONDS,
            )
        except subprocess.TimeoutExpired:
            return ToolResult(error=f"执行超时（{_TIMEOUT_SECONDS} 秒上限），已强制终止")
        output = (proc.stdout + proc.stderr).strip()
        truncated = len(output) > _MAX_OUTPUT_CHARS
        if truncated:
            output = output[:_MAX_OUTPUT_CHARS]
        return ToolResult(
            data={
                "returncode": proc.returncode,
                "output": output or "（无输出）",
                "truncated": truncated,
            }
        )
