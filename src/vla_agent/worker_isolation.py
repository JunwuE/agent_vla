"""Linux Landlock + seccomp confinement, installed before generated code is loaded.

No host writes, process creation, networking, execution, signal delivery, or metadata
mutation. Only runtime libraries and the current attempt's exported inputs are readable.
Landlock ABI 1 lacks truncate protection; the syscall allowlist denies it explicitly.
"""

import ctypes
import errno
import os
import platform
import sys
from pathlib import Path


def confine(directory):
    if platform.machine() not in {"x86_64", "aarch64"}:
        raise RuntimeError("Unsupported Landlock syscall architecture")
    libc = ctypes.CDLL(None, use_errno=True)
    sec = ctypes.CDLL("libseccomp.so.2", use_errno=True)
    sec.seccomp_init.argtypes = [ctypes.c_uint32]
    sec.seccomp_init.restype = ctypes.c_void_p
    sec.seccomp_syscall_resolve_name.argtypes = [ctypes.c_char_p]
    sec.seccomp_rule_add.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_int, ctypes.c_uint]
    sec.seccomp_load.argtypes = [ctypes.c_void_p]
    sec.seccomp_release.argtypes = [ctypes.c_void_p]
    abi = libc.syscall(444, 0, 0, 1)
    if abi < 1:
        raise OSError(ctypes.get_errno(), "Landlock is required; isolation failed closed")

    class Ruleset(ctypes.Structure):
        _fields_ = [("handled_access_fs", ctypes.c_uint64)]

    class Beneath(ctypes.Structure):
        _pack_ = 1
        _fields_ = [("allowed_access", ctypes.c_uint64), ("parent_fd", ctypes.c_int32)]

    rules = Ruleset((1 << 13) - 1)  # Every ABI-1 filesystem permission is handled.
    fd = libc.syscall(444, ctypes.byref(rules), ctypes.sizeof(rules), 0)
    if fd < 0:
        raise OSError(ctypes.get_errno(), "create Landlock ruleset")
    try:
        roots = [
            Path("/usr"),
            Path("/lib"),
            Path("/lib64"),
            Path(sys.base_prefix),
            Path(sys.prefix),
            Path(directory),
            Path("/etc/ld.so.cache"),
            Path("/dev/urandom"),
            Path("/dev/null"),
        ]
        for path in set(p.resolve() for p in roots if p.exists()):
            child = os.open(path, os.O_PATH | os.O_CLOEXEC)
            try:
                rule = Beneath((1 << 2) | ((1 << 3) if path.is_dir() else 0), child)
                if libc.syscall(445, fd, 1, ctypes.byref(rule), 0):
                    raise OSError(ctypes.get_errno(), f"Landlock rule: {path}")
            finally:
                os.close(child)
        if libc.prctl(38, 1, 0, 0, 0) or libc.syscall(446, fd, 0):
            raise OSError(ctypes.get_errno(), "enforce Landlock")
    finally:
        os.close(fd)
    # Allow ordinary single-process Python/numpy computation and pipe RPC only.
    # No clone/fork/exec, socket, ptrace, kill, mount, io_uring, chmod or truncate.
    names = """read write readv writev pread64 pwrite64 close close_range open openat
        fstat newfstatat stat lstat statx lseek access faccessat faccessat2 readlink readlinkat
        getdents getdents64 fcntl dup dup2 dup3 mmap mprotect munmap mremap brk madvise
        rt_sigaction rt_sigprocmask rt_sigreturn sigaltstack futex futex_waitv set_robust_list
        getpid getppid gettid getuid geteuid getgid getegid getresuid getresgid getgroups
        clock_gettime clock_getres clock_nanosleep nanosleep gettimeofday time getrandom
        sched_getaffinity sched_yield getcpu uname sysinfo getcwd chdir fchdir
        poll ppoll select pselect6 epoll_create epoll_create1 epoll_ctl epoll_wait epoll_pwait
        getrlimit getrusage restart_syscall exit exit_group""".split()
    ctx = sec.seccomp_init(0x00050000 | errno.EPERM)
    if not ctx:
        raise RuntimeError("Cannot create seccomp filter")
    try:
        for name in names:
            number = sec.seccomp_syscall_resolve_name(name.encode())
            if number >= 0 and sec.seccomp_rule_add(ctx, 0x7FFF0000, number, 0):
                raise RuntimeError(f"Cannot allow syscall {name}")
        if sec.seccomp_load(ctx):
            raise RuntimeError("Cannot enforce seccomp filter")
    finally:
        sec.seccomp_release(ctx)
