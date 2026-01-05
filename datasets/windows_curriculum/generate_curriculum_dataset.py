#!/usr/bin/env python3
"""
Windows Systems Programming Curriculum Dataset
Tiered dataset for progressive SFT and RLVR training
"""

import json
from pathlib import Path
from typing import List, Dict, Optional

BASE_DIR = Path(__file__).parent
PROBLEMS = []

def add(tier: int, cat: str, subcat: str, api: str, diff: str, prompt: str, solution: str,
        test_cases: Optional[List[Dict]] = None, tags: Optional[List[str]] = None,
        verification: str = "stdout_contains"):
    """Add a problem to the dataset with tier metadata"""
    PROBLEMS.append({
        "tier": tier,
        "category": cat,
        "subcategory": subcat,
        "api": api,
        "difficulty": diff,
        "prompt": prompt,
        "solution": solution,
        "test_cases": test_cases or [{"type": "compiles", "expected": True}],
        "tags": tags or [api],
        "verification_strategy": verification
    })

# =============================================================================
# TIER 1: FOUNDATIONS (Easy) - Verifiable via stdout comparison
# =============================================================================

# --- Process Info ---
add(1, "process", "info", "GetCurrentProcessId", "beginner",
"Write a C++ program that displays the current process ID, thread ID, and executable path.",
r'''#include <windows.h>
#include <stdio.h>

int main() {
    DWORD pid = GetCurrentProcessId();
    DWORD tid = GetCurrentThreadId();
    
    char path[MAX_PATH];
    GetModuleFileNameA(NULL, path, MAX_PATH);
    
    printf("Process ID: %lu\n", pid);
    printf("Thread ID: %lu\n", tid);
    printf("Executable: %s\n", path);
    printf("Command Line: %s\n", GetCommandLineA());
    
    return 0;
}''',
[{"type": "output_contains", "value": "Process ID:"},
 {"type": "output_contains", "value": "Executable:"}])

add(1, "process", "info", "GetParentProcessId", "beginner",
"Write a C++ program that gets the parent process ID using NtQueryInformationProcess.",
r'''#include <windows.h>
#include <stdio.h>

typedef LONG NTSTATUS;
typedef struct _PROCESS_BASIC_INFORMATION {
    PVOID Reserved1;
    PVOID PebBaseAddress;
    PVOID Reserved2[2];
    ULONG_PTR UniqueProcessId;
    ULONG_PTR InheritedFromUniqueProcessId;
} PROCESS_BASIC_INFORMATION;

typedef NTSTATUS (NTAPI *pNtQueryInformationProcess)(HANDLE, ULONG, PVOID, ULONG, PULONG);

int main() {
    HMODULE ntdll = GetModuleHandleA("ntdll.dll");
    pNtQueryInformationProcess NtQIP = (pNtQueryInformationProcess)
        GetProcAddress(ntdll, "NtQueryInformationProcess");
    
    PROCESS_BASIC_INFORMATION pbi;
    ULONG len;
    NtQIP(GetCurrentProcess(), 0, &pbi, sizeof(pbi), &len);
    
    printf("Current PID: %llu\n", (ULONGLONG)pbi.UniqueProcessId);
    printf("Parent PID: %llu\n", (ULONGLONG)pbi.InheritedFromUniqueProcessId);
    printf("Verification: GetCurrentProcessId = %lu\n", GetCurrentProcessId());
    
    return 0;
}''',
[{"type": "output_contains", "value": "Parent PID:"}])

# --- Environment Variables ---
add(1, "environment", "get", "GetEnvironmentVariable", "beginner",
"Write a C++ program that reads and displays the PATH environment variable.",
r'''#include <windows.h>
#include <stdio.h>

int main() {
    char buffer[32768];
    DWORD len = GetEnvironmentVariableA("PATH", buffer, sizeof(buffer));
    
    if (len > 0) {
        printf("PATH length: %lu chars\n", len);
        printf("PATH (first 200 chars): %.200s...\n", buffer);
    } else {
        printf("GetEnvironmentVariable failed: %lu\n", GetLastError());
    }
    
    // Also get COMPUTERNAME
    len = GetEnvironmentVariableA("COMPUTERNAME", buffer, sizeof(buffer));
    if (len > 0) {
        printf("COMPUTERNAME: %s\n", buffer);
    }
    
    // Get USERNAME
    len = GetEnvironmentVariableA("USERNAME", buffer, sizeof(buffer));
    if (len > 0) {
        printf("USERNAME: %s\n", buffer);
    }
    
    return 0;
}''',
[{"type": "output_contains", "value": "PATH length:"},
 {"type": "output_contains", "value": "COMPUTERNAME:"}])

add(1, "environment", "set", "SetEnvironmentVariable", "beginner",
"Write a C++ program that sets an environment variable and reads it back.",
r'''#include <windows.h>
#include <stdio.h>

int main() {
    const char* varName = "MY_TEST_VAR";
    const char* varValue = "HelloFromSetEnvironmentVariable";
    
    printf("Setting %s = %s\n", varName, varValue);
    
    if (!SetEnvironmentVariableA(varName, varValue)) {
        printf("SetEnvironmentVariable failed: %lu\n", GetLastError());
        return 1;
    }
    
    char buffer[256];
    DWORD len = GetEnvironmentVariableA(varName, buffer, sizeof(buffer));
    
    if (len > 0) {
        printf("Read back: %s\n", buffer);
        printf("Round-trip: %s\n", strcmp(buffer, varValue) == 0 ? "SUCCESS" : "FAILED");
    }
    
    // Clean up
    SetEnvironmentVariableA(varName, NULL);
    printf("Variable cleared\n");
    
    return 0;
}''',
[{"type": "output_contains", "value": "Round-trip: SUCCESS"}])

add(1, "environment", "expand", "ExpandEnvironmentStrings", "beginner",
"Write a C++ program that expands environment variable references in a string.",
r'''#include <windows.h>
#include <stdio.h>

int main() {
    const char* input = "%USERPROFILE%\\Documents\\test.txt";
    char output[MAX_PATH];
    
    printf("Input:  %s\n", input);
    
    DWORD len = ExpandEnvironmentStringsA(input, output, sizeof(output));
    
    if (len > 0) {
        printf("Output: %s\n", output);
        printf("Expanded %lu chars\n", len);
    }
    
    // Try another
    const char* input2 = "%SYSTEMROOT%\\System32";
    ExpandEnvironmentStringsA(input2, output, sizeof(output));
    printf("System32: %s\n", output);
    
    return 0;
}''',
[{"type": "output_contains", "value": "Output:"},
 {"type": "output_contains", "value": "System32:"}])

# --- System Info ---
add(1, "sysinfo", "cpu", "GetSystemInfo", "beginner",
"Write a C++ program that displays CPU and system architecture information.",
r'''#include <windows.h>
#include <stdio.h>

int main() {
    SYSTEM_INFO si;
    GetSystemInfo(&si);
    
    printf("System Information\n\n");
    printf("Processor Architecture: ");
    switch (si.wProcessorArchitecture) {
        case PROCESSOR_ARCHITECTURE_AMD64: printf("x64 (AMD64)\n"); break;
        case PROCESSOR_ARCHITECTURE_INTEL: printf("x86\n"); break;
        case PROCESSOR_ARCHITECTURE_ARM64: printf("ARM64\n"); break;
        default: printf("Unknown (%u)\n", si.wProcessorArchitecture);
    }
    
    printf("Number of Processors: %lu\n", si.dwNumberOfProcessors);
    printf("Page Size: %lu bytes\n", si.dwPageSize);
    printf("Processor Type: %lu\n", si.dwProcessorType);
    printf("Processor Level: %u\n", si.wProcessorLevel);
    printf("Processor Revision: 0x%04X\n", si.wProcessorRevision);
    printf("Allocation Granularity: %lu\n", si.dwAllocationGranularity);
    printf("Min App Address: 0x%p\n", si.lpMinimumApplicationAddress);
    printf("Max App Address: 0x%p\n", si.lpMaximumApplicationAddress);
    
    return 0;
}''',
[{"type": "output_contains", "value": "Number of Processors:"},
 {"type": "output_contains", "value": "Page Size:"}])

add(1, "sysinfo", "memory", "GlobalMemoryStatusEx", "beginner",
"Write a C++ program that displays memory usage statistics.",
r'''#include <windows.h>
#include <stdio.h>

int main() {
    MEMORYSTATUSEX mem;
    mem.dwLength = sizeof(mem);
    
    if (!GlobalMemoryStatusEx(&mem)) {
        printf("GlobalMemoryStatusEx failed\n");
        return 1;
    }
    
    printf("Memory Status\n\n");
    printf("Memory Load: %lu%%\n", mem.dwMemoryLoad);
    printf("Total Physical: %.2f GB\n", mem.ullTotalPhys / (1024.0*1024*1024));
    printf("Available Physical: %.2f GB\n", mem.ullAvailPhys / (1024.0*1024*1024));
    printf("Total Page File: %.2f GB\n", mem.ullTotalPageFile / (1024.0*1024*1024));
    printf("Available Page File: %.2f GB\n", mem.ullAvailPageFile / (1024.0*1024*1024));
    printf("Total Virtual: %.2f TB\n", mem.ullTotalVirtual / (1024.0*1024*1024*1024));
    printf("Available Virtual: %.2f TB\n", mem.ullAvailVirtual / (1024.0*1024*1024*1024));
    
    return 0;
}''',
[{"type": "output_contains", "value": "Memory Load:"},
 {"type": "output_contains", "value": "Total Physical:"}])

add(1, "sysinfo", "version", "RtlGetVersion", "beginner",
"Write a C++ program that gets the true Windows version using RtlGetVersion.",
r'''#include <windows.h>
#include <stdio.h>

typedef LONG NTSTATUS;
typedef NTSTATUS (NTAPI *pRtlGetVersion)(OSVERSIONINFOEXW*);

int main() {
    HMODULE ntdll = GetModuleHandleA("ntdll.dll");
    pRtlGetVersion RtlGetVersion = (pRtlGetVersion)GetProcAddress(ntdll, "RtlGetVersion");
    
    OSVERSIONINFOEXW ver = {sizeof(ver)};
    RtlGetVersion(&ver);
    
    printf("Windows Version (RtlGetVersion)\n\n");
    printf("Version: %lu.%lu.%lu\n", ver.dwMajorVersion, ver.dwMinorVersion, ver.dwBuildNumber);
    printf("Platform ID: %lu\n", ver.dwPlatformId);
    printf("Product Type: %s\n", 
        ver.wProductType == 1 ? "Workstation" :
        ver.wProductType == 2 ? "Domain Controller" :
        ver.wProductType == 3 ? "Server" : "Unknown");
    
    // Determine Windows name
    printf("Windows: ");
    if (ver.dwMajorVersion == 10 && ver.dwBuildNumber >= 22000) printf("Windows 11\n");
    else if (ver.dwMajorVersion == 10) printf("Windows 10\n");
    else if (ver.dwMajorVersion == 6 && ver.dwMinorVersion == 3) printf("Windows 8.1\n");
    else if (ver.dwMajorVersion == 6 && ver.dwMinorVersion == 2) printf("Windows 8\n");
    else if (ver.dwMajorVersion == 6 && ver.dwMinorVersion == 1) printf("Windows 7\n");
    else printf("Other\n");
    
    return 0;
}''',
[{"type": "output_contains", "value": "Version:"},
 {"type": "output_contains", "value": "Windows:"}])

add(1, "sysinfo", "computer", "GetComputerName", "beginner",
"Write a C++ program that displays computer name and workgroup/domain info.",
r'''#include <windows.h>
#include <stdio.h>
#pragma comment(lib, "netapi32.lib")
#pragma comment(lib, "advapi32.lib")

int main() {
    char compName[MAX_COMPUTERNAME_LENGTH + 1];
    DWORD size = sizeof(compName);
    GetComputerNameA(compName, &size);
    printf("Computer Name: %s\n", compName);
    
    char userName[256];
    size = sizeof(userName);
    GetUserNameA(userName, &size);
    printf("User Name: %s\n", userName);
    
    // Get Windows directory
    char winDir[MAX_PATH];
    GetWindowsDirectoryA(winDir, MAX_PATH);
    printf("Windows Directory: %s\n", winDir);
    
    // Get System directory
    char sysDir[MAX_PATH];
    GetSystemDirectoryA(sysDir, MAX_PATH);
    printf("System Directory: %s\n", sysDir);
    
    // Get temp path
    char tempPath[MAX_PATH];
    GetTempPathA(MAX_PATH, tempPath);
    printf("Temp Path: %s\n", tempPath);
    
    return 0;
}''',
[{"type": "output_contains", "value": "Computer Name:"},
 {"type": "output_contains", "value": "Windows Directory:"}])

# --- Time/Date ---
add(1, "time", "system", "GetSystemTime", "beginner",
"Write a C++ program that displays system time in UTC and local time.",
r'''#include <windows.h>
#include <stdio.h>

int main() {
    SYSTEMTIME utc, local;
    
    GetSystemTime(&utc);
    GetLocalTime(&local);
    
    printf("Time Information\n\n");
    printf("UTC Time:   %04d-%02d-%02d %02d:%02d:%02d.%03d\n",
        utc.wYear, utc.wMonth, utc.wDay,
        utc.wHour, utc.wMinute, utc.wSecond, utc.wMilliseconds);
    
    printf("Local Time: %04d-%02d-%02d %02d:%02d:%02d.%03d\n",
        local.wYear, local.wMonth, local.wDay,
        local.wHour, local.wMinute, local.wSecond, local.wMilliseconds);
    
    // Day of week
    const char* days[] = {"Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"};
    printf("Day of Week: %s\n", days[local.wDayOfWeek]);
    
    return 0;
}''',
[{"type": "output_contains", "value": "UTC Time:"},
 {"type": "output_contains", "value": "Local Time:"}])

add(1, "time", "filetime", "FileTimeToSystemTime", "beginner",
"Write a C++ program that converts between FILETIME and SYSTEMTIME formats.",
r'''#include <windows.h>
#include <stdio.h>

int main() {
    printf("FILETIME Conversion Demo\n\n");
    
    // Get current time as FILETIME
    FILETIME ft;
    GetSystemTimeAsFileTime(&ft);
    
    printf("Current FILETIME: 0x%08lX%08lX\n", ft.dwHighDateTime, ft.dwLowDateTime);
    
    // Convert to SYSTEMTIME
    SYSTEMTIME st;
    FileTimeToSystemTime(&ft, &st);
    printf("As SYSTEMTIME: %04d-%02d-%02d %02d:%02d:%02d\n",
        st.wYear, st.wMonth, st.wDay, st.wHour, st.wMinute, st.wSecond);
    
    // Convert back
    FILETIME ft2;
    SystemTimeToFileTime(&st, &ft2);
    printf("Round-trip: %s\n", 
        (ft.dwHighDateTime == ft2.dwHighDateTime && ft.dwLowDateTime == ft2.dwLowDateTime) 
        ? "SUCCESS" : "FAILED");
    
    // FILETIME to ULARGE_INTEGER for math
    ULARGE_INTEGER uli;
    uli.LowPart = ft.dwLowDateTime;
    uli.HighPart = ft.dwHighDateTime;
    printf("As 64-bit: %llu (100ns intervals since 1601)\n", uli.QuadPart);
    
    return 0;
}''',
[{"type": "output_contains", "value": "As SYSTEMTIME:"},
 {"type": "output_contains", "value": "Round-trip: SUCCESS"}])

add(1, "time", "ticks", "GetTickCount64", "beginner",
"Write a C++ program that measures elapsed time using tick counts.",
r'''#include <windows.h>
#include <stdio.h>

int main() {
    printf("Tick Count Demo\n\n");
    
    ULONGLONG tick64 = GetTickCount64();
    DWORD tick32 = GetTickCount();
    
    printf("GetTickCount64: %llu ms\n", tick64);
    printf("GetTickCount:   %lu ms\n", tick32);
    printf("System uptime: %.2f hours\n", tick64 / (1000.0 * 60 * 60));
    
    // Measure a Sleep
    printf("\nMeasuring Sleep(100)...\n");
    ULONGLONG start = GetTickCount64();
    Sleep(100);
    ULONGLONG elapsed = GetTickCount64() - start;
    printf("Elapsed: %llu ms\n", elapsed);
    
    // QueryPerformanceCounter for high precision
    LARGE_INTEGER freq, pc1, pc2;
    QueryPerformanceFrequency(&freq);
    printf("\nQPC Frequency: %lld Hz\n", freq.QuadPart);
    
    QueryPerformanceCounter(&pc1);
    Sleep(50);
    QueryPerformanceCounter(&pc2);
    
    double ms = (double)(pc2.QuadPart - pc1.QuadPart) / freq.QuadPart * 1000;
    printf("High-res elapsed: %.3f ms\n", ms);
    
    return 0;
}''',
[{"type": "output_contains", "value": "System uptime:"},
 {"type": "output_contains", "value": "High-res elapsed:"}])

# --- String Handling ---
add(1, "string", "wide", "MultiByteToWideChar", "beginner",
"Write a C++ program that converts between ANSI and Unicode strings.",
r'''#include <windows.h>
#include <stdio.h>

int main() {
    printf("String Conversion Demo\n\n");
    
    // ANSI to Unicode
    const char* ansiStr = "Hello, Windows!";
    wchar_t wideStr[256];
    
    int wideLen = MultiByteToWideChar(CP_ACP, 0, ansiStr, -1, wideStr, 256);
    printf("ANSI: %s (%zu bytes)\n", ansiStr, strlen(ansiStr));
    wprintf(L"Wide: %s (%d chars)\n", wideStr, wideLen - 1);
    
    // Unicode to ANSI
    char ansiBack[256];
    int ansiLen = WideCharToMultiByte(CP_ACP, 0, wideStr, -1, ansiBack, 256, NULL, NULL);
    printf("Back: %s (%d bytes)\n", ansiBack, ansiLen - 1);
    
    printf("Round-trip: %s\n", strcmp(ansiStr, ansiBack) == 0 ? "SUCCESS" : "FAILED");
    
    // UTF-8 conversion
    const char* utf8Str = "Caf\xC3\xA9";  // "Café" in UTF-8
    wchar_t utf8Wide[256];
    MultiByteToWideChar(CP_UTF8, 0, utf8Str, -1, utf8Wide, 256);
    printf("\nUTF-8 test: input bytes=%zu, output chars=%zu\n", 
        strlen(utf8Str), wcslen(utf8Wide));
    
    return 0;
}''',
[{"type": "output_contains", "value": "Round-trip: SUCCESS"}])

add(1, "string", "format", "wsprintfA", "beginner",
"Write a C++ program demonstrating Windows string formatting functions.",
r'''#include <windows.h>
#include <stdio.h>
#include <strsafe.h>
#pragma comment(lib, "user32.lib")

int main() {
    printf("String Formatting Demo\n\n");
    
    char buffer[256];
    
    // wsprintf (legacy)
    wsprintfA(buffer, "PID=%lu, Name=%s", GetCurrentProcessId(), "TestApp");
    printf("wsprintf: %s\n", buffer);
    
    // StringCchPrintf (safe)
    HRESULT hr = StringCchPrintfA(buffer, sizeof(buffer), 
        "Safe format: %d + %d = %d", 10, 20, 30);
    printf("StringCchPrintf: %s (hr=0x%lX)\n", buffer, hr);
    
    // StringCchCopy
    StringCchCopyA(buffer, sizeof(buffer), "Source string");
    printf("StringCchCopy: %s\n", buffer);
    
    // StringCchCat
    StringCchCatA(buffer, sizeof(buffer), " + appended");
    printf("StringCchCat: %s\n", buffer);
    
    // StringCchLength
    size_t len;
    StringCchLengthA(buffer, sizeof(buffer), &len);
    printf("StringCchLength: %zu\n", len);
    
    return 0;
}''',
[{"type": "output_contains", "value": "wsprintf:"},
 {"type": "output_contains", "value": "StringCchPrintf:"}])

# --- Error Handling ---
add(1, "error", "format", "FormatMessage", "beginner",
"Write a C++ program that formats Windows error codes into readable messages.",
r'''#include <windows.h>
#include <stdio.h>

void PrintError(DWORD code) {
    char* msg = NULL;
    DWORD len = FormatMessageA(
        FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS,
        NULL, code, MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT),
        (LPSTR)&msg, 0, NULL);
    
    if (len > 0) {
        // Remove trailing newline
        while (len > 0 && (msg[len-1] == '\n' || msg[len-1] == '\r')) msg[--len] = 0;
        printf("Error %lu: %s\n", code, msg);
        LocalFree(msg);
    } else {
        printf("Error %lu: (no message)\n", code);
    }
}

int main() {
    printf("Error Message Formatting\n\n");
    
    // Common error codes
    PrintError(ERROR_SUCCESS);
    PrintError(ERROR_FILE_NOT_FOUND);
    PrintError(ERROR_ACCESS_DENIED);
    PrintError(ERROR_INVALID_HANDLE);
    PrintError(ERROR_NOT_ENOUGH_MEMORY);
    PrintError(ERROR_INVALID_PARAMETER);
    
    // Trigger an error
    printf("\nTriggering ERROR_FILE_NOT_FOUND:\n");
    HANDLE h = CreateFileA("nonexistent_file_12345.xyz", GENERIC_READ, 0, NULL,
        OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
    PrintError(GetLastError());
    
    return 0;
}''',
[{"type": "output_contains", "value": "Error 0: The operation completed successfully"},
 {"type": "output_contains", "value": "Error 2:"}])

# --- File Operations ---
add(1, "file", "basic", "CreateFileA", "beginner",
"Write a C++ program that creates, writes, reads, and deletes a file.",
r'''#include <windows.h>
#include <stdio.h>

int main() {
    const char* filename = "test_output.txt";
    const char* content = "Hello from Windows API!";
    
    printf("File Operations Demo\n\n");
    
    // Create and write
    HANDLE hFile = CreateFileA(filename, GENERIC_WRITE, 0, NULL,
        CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, NULL);
    
    if (hFile == INVALID_HANDLE_VALUE) {
        printf("CreateFile failed: %lu\n", GetLastError());
        return 1;
    }
    
    DWORD written;
    WriteFile(hFile, content, strlen(content), &written, NULL);
    printf("Wrote %lu bytes to %s\n", written, filename);
    CloseHandle(hFile);
    
    // Read back
    hFile = CreateFileA(filename, GENERIC_READ, 0, NULL,
        OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
    
    char buffer[256] = {0};
    DWORD bytesRead;
    ReadFile(hFile, buffer, sizeof(buffer)-1, &bytesRead, NULL);
    printf("Read %lu bytes: %s\n", bytesRead, buffer);
    CloseHandle(hFile);
    
    // Get file size
    WIN32_FILE_ATTRIBUTE_DATA fad;
    GetFileAttributesExA(filename, GetFileExInfoStandard, &fad);
    printf("File size: %lu bytes\n", fad.nFileSizeLow);
    
    // Delete
    DeleteFileA(filename);
    printf("File deleted\n");
    
    return 0;
}''',
[{"type": "output_contains", "value": "Hello from Windows API"},
 {"type": "output_contains", "value": "File deleted"}])

add(1, "file", "attributes", "GetFileAttributes", "beginner",
"Write a C++ program that displays file attributes for system files.",
r'''#include <windows.h>
#include <stdio.h>

void ShowAttributes(const char* path) {
    DWORD attrs = GetFileAttributesA(path);
    
    if (attrs == INVALID_FILE_ATTRIBUTES) {
        printf("%-40s ERROR (%lu)\n", path, GetLastError());
        return;
    }
    
    char flags[64] = "";
    if (attrs & FILE_ATTRIBUTE_DIRECTORY) strcat(flags, "D");
    if (attrs & FILE_ATTRIBUTE_READONLY) strcat(flags, "R");
    if (attrs & FILE_ATTRIBUTE_HIDDEN) strcat(flags, "H");
    if (attrs & FILE_ATTRIBUTE_SYSTEM) strcat(flags, "S");
    if (attrs & FILE_ATTRIBUTE_ARCHIVE) strcat(flags, "A");
    if (attrs & FILE_ATTRIBUTE_REPARSE_POINT) strcat(flags, "L");
    
    printf("%-40s [%s] 0x%08lX\n", path, flags, attrs);
}

int main() {
    printf("File Attributes Demo\n\n");
    printf("%-40s %-8s %s\n", "Path", "Flags", "Value");
    printf("---------------------------------------- -------- ----------\n");
    
    ShowAttributes("C:\\Windows");
    ShowAttributes("C:\\Windows\\System32");
    ShowAttributes("C:\\Windows\\System32\\ntdll.dll");
    ShowAttributes("C:\\Windows\\System32\\kernel32.dll");
    ShowAttributes("C:\\pagefile.sys");
    ShowAttributes("C:\\hiberfil.sys");
    
    return 0;
}''',
[{"type": "output_contains", "value": "ntdll.dll"},
 {"type": "output_contains", "value": "["}])

add(1, "file", "directory", "FindFirstFile", "beginner",
"Write a C++ program that lists files in the Windows directory.",
r'''#include <windows.h>
#include <stdio.h>

int main() {
    printf("Directory Listing Demo\n\n");
    
    WIN32_FIND_DATAA fd;
    HANDLE hFind = FindFirstFileA("C:\\Windows\\*.exe", &fd);
    
    if (hFind == INVALID_HANDLE_VALUE) {
        printf("FindFirstFile failed: %lu\n", GetLastError());
        return 1;
    }
    
    printf("EXE files in C:\\Windows:\n\n");
    printf("%-30s %12s %s\n", "Name", "Size", "Attrs");
    printf("------------------------------ ------------ -----\n");
    
    int count = 0;
    do {
        if (!(fd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY)) {
            char attrs[8] = "";
            if (fd.dwFileAttributes & FILE_ATTRIBUTE_HIDDEN) strcat(attrs, "H");
            if (fd.dwFileAttributes & FILE_ATTRIBUTE_SYSTEM) strcat(attrs, "S");
            if (fd.dwFileAttributes & FILE_ATTRIBUTE_READONLY) strcat(attrs, "R");
            
            printf("%-30s %12lu %s\n", fd.cFileName, fd.nFileSizeLow, attrs);
            count++;
        }
    } while (FindNextFileA(hFind, &fd) && count < 10);
    
    FindClose(hFind);
    printf("\nFound %d files (limited to 10)\n", count);
    
    return 0;
}''',
[{"type": "output_contains", "value": "EXE files"},
 {"type": "output_contains", "value": "Found"}])

# --- Registry ---
add(1, "registry", "read", "RegQueryValueEx", "beginner",
"Write a C++ program that reads Windows version from the registry.",
r'''#include <windows.h>
#include <stdio.h>
#pragma comment(lib, "advapi32.lib")

int main() {
    printf("Registry Reader\n\n");
    
    HKEY hKey;
    LONG result = RegOpenKeyExA(HKEY_LOCAL_MACHINE,
        "SOFTWARE\\Microsoft\\Windows NT\\CurrentVersion",
        0, KEY_READ, &hKey);
    
    if (result != ERROR_SUCCESS) {
        printf("RegOpenKeyEx failed: %ld\n", result);
        return 1;
    }
    
    char strVal[256];
    DWORD strSize, valType;
    
    const char* names[] = {"ProductName", "CurrentBuild", "DisplayVersion", "EditionID"};
    
    for (int i = 0; i < 4; i++) {
        strSize = sizeof(strVal);
        if (RegQueryValueExA(hKey, names[i], NULL, &valType,
                (LPBYTE)strVal, &strSize) == ERROR_SUCCESS) {
            printf("%-15s: %s\n", names[i], strVal);
        }
    }
    
    DWORD dwordVal, dwordSize = sizeof(dwordVal);
    if (RegQueryValueExA(hKey, "CurrentMajorVersionNumber", NULL, &valType,
            (LPBYTE)&dwordVal, &dwordSize) == ERROR_SUCCESS) {
        printf("%-15s: %lu\n", "MajorVersion", dwordVal);
    }
    
    RegCloseKey(hKey);
    return 0;
}''',
[{"type": "output_contains", "value": "ProductName:"},
 {"type": "output_contains", "value": "CurrentBuild:"}])

add(1, "registry", "enum", "RegEnumKeyEx", "beginner",
"Write a C++ program that enumerates registry subkeys.",
r'''#include <windows.h>
#include <stdio.h>
#pragma comment(lib, "advapi32.lib")

int main() {
    printf("Registry Subkey Enumeration\n\n");
    
    HKEY hKey;
    if (RegOpenKeyExA(HKEY_LOCAL_MACHINE, "SOFTWARE\\Microsoft", 
            0, KEY_READ, &hKey) != ERROR_SUCCESS) {
        printf("Failed to open key\n");
        return 1;
    }
    
    printf("Subkeys of HKLM\\SOFTWARE\\Microsoft:\n\n");
    
    char keyName[256];
    DWORD keyLen;
    FILETIME ft;
    
    for (DWORD i = 0; i < 20; i++) {
        keyLen = sizeof(keyName);
        LONG result = RegEnumKeyExA(hKey, i, keyName, &keyLen, NULL, NULL, NULL, &ft);
        
        if (result == ERROR_NO_MORE_ITEMS) break;
        if (result == ERROR_SUCCESS) {
            printf("  [%lu] %s\n", i, keyName);
        }
    }
    
    RegCloseKey(hKey);
    printf("\n(Limited to 20 entries)\n");
    
    return 0;
}''',
[{"type": "output_contains", "value": "Subkeys of"},
 {"type": "output_contains", "value": "[0]"}])

# --- Memory ---
add(1, "memory", "virtual", "VirtualAlloc", "beginner",
"Write a C++ program that allocates memory, writes to it, and frees it.",
r'''#include <windows.h>
#include <stdio.h>

int main() {
    SIZE_T size = 4096;
    LPVOID pMem = VirtualAlloc(NULL, size, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
    
    if (!pMem) {
        printf("VirtualAlloc failed: %lu\n", GetLastError());
        return 1;
    }
    
    printf("Allocated at: 0x%p\n", pMem);
    
    const char* msg = "Hello from VirtualAlloc";
    memcpy(pMem, msg, strlen(msg) + 1);
    printf("Data: %s\n", (char*)pMem);
    
    VirtualFree(pMem, 0, MEM_RELEASE);
    printf("Memory freed\n");
    
    return 0;
}''',
[{"type": "output_contains", "value": "Allocated at:"},
 {"type": "output_contains", "value": "Hello from VirtualAlloc"}])

# =============================================================================
# TIER 2: CORE APIS (Medium) - Verifiable via output + side effects
# =============================================================================

# --- Named Pipes ---
add(2, "ipc", "pipes", "CreateNamedPipe", "intermediate",
"Write a C++ program that creates a named pipe server and client in the same process.",
r'''#include <windows.h>
#include <stdio.h>

int main() {
    printf("Named Pipe Demo\n\n");
    
    const char* pipeName = "\\\\.\\pipe\\TestPipe";
    
    // Create server
    HANDLE hServer = CreateNamedPipeA(pipeName,
        PIPE_ACCESS_DUPLEX,
        PIPE_TYPE_MESSAGE | PIPE_READMODE_MESSAGE | PIPE_WAIT,
        1, 4096, 4096, 0, NULL);
    
    if (hServer == INVALID_HANDLE_VALUE) {
        printf("CreateNamedPipe failed: %lu\n", GetLastError());
        return 1;
    }
    printf("Server pipe created\n");
    
    // Connect client (in same process for demo)
    HANDLE hClient = CreateFileA(pipeName, GENERIC_READ | GENERIC_WRITE,
        0, NULL, OPEN_EXISTING, 0, NULL);
    
    if (hClient == INVALID_HANDLE_VALUE) {
        printf("Client connect failed: %lu\n", GetLastError());
        CloseHandle(hServer);
        return 1;
    }
    printf("Client connected\n");
    
    // Write from client
    const char* msg = "Hello via Named Pipe!";
    DWORD written;
    WriteFile(hClient, msg, strlen(msg)+1, &written, NULL);
    printf("Client sent: %s\n", msg);
    
    // Read on server
    char buffer[256];
    DWORD bytesRead;
    ReadFile(hServer, buffer, sizeof(buffer), &bytesRead, NULL);
    printf("Server received: %s\n", buffer);
    
    printf("Round-trip: %s\n", strcmp(msg, buffer) == 0 ? "SUCCESS" : "FAILED");
    
    CloseHandle(hClient);
    CloseHandle(hServer);
    
    return 0;
}''',
[{"type": "output_contains", "value": "Round-trip: SUCCESS"}])

# --- Events ---
add(2, "sync", "event", "CreateEvent", "intermediate",
"Write a C++ program demonstrating event synchronization between threads.",
r'''#include <windows.h>
#include <stdio.h>

HANDLE hEvent;

DWORD WINAPI Worker(LPVOID arg) {
    int id = (int)(INT_PTR)arg;
    printf("[Thread %d] Waiting for event...\n", id);
    WaitForSingleObject(hEvent, INFINITE);
    printf("[Thread %d] Event signaled!\n", id);
    return 0;
}

int main() {
    printf("Event Synchronization Demo\n\n");
    
    // Create auto-reset event (initially non-signaled)
    hEvent = CreateEventA(NULL, FALSE, FALSE, "TestEvent");
    printf("Event created: 0x%p\n", hEvent);
    
    // Start threads
    HANDLE threads[3];
    for (int i = 0; i < 3; i++) {
        threads[i] = CreateThread(NULL, 0, Worker, (LPVOID)(INT_PTR)(i+1), 0, NULL);
    }
    
    Sleep(100);  // Let threads start waiting
    
    // Signal event 3 times (one for each thread, auto-reset)
    for (int i = 0; i < 3; i++) {
        printf("[Main] Signaling event (%d)\n", i+1);
        SetEvent(hEvent);
        Sleep(50);
    }
    
    WaitForMultipleObjects(3, threads, TRUE, 5000);
    
    for (int i = 0; i < 3; i++) CloseHandle(threads[i]);
    CloseHandle(hEvent);
    
    printf("\nAll threads completed\n");
    return 0;
}''',
[{"type": "output_contains", "value": "Event signaled!"},
 {"type": "output_contains", "value": "All threads completed"}])

# --- Mutexes ---
add(2, "sync", "mutex", "CreateMutex", "intermediate",
"Write a C++ program demonstrating mutex synchronization.",
r'''#include <windows.h>
#include <stdio.h>

HANDLE hMutex;
int sharedCounter = 0;

DWORD WINAPI Worker(LPVOID arg) {
    int id = (int)(INT_PTR)arg;
    
    for (int i = 0; i < 1000; i++) {
        WaitForSingleObject(hMutex, INFINITE);
        sharedCounter++;
        ReleaseMutex(hMutex);
    }
    
    printf("[Thread %d] Done (1000 increments)\n", id);
    return 0;
}

int main() {
    printf("Mutex Synchronization Demo\n\n");
    
    hMutex = CreateMutexA(NULL, FALSE, NULL);
    printf("Mutex created: 0x%p\n\n", hMutex);
    
    const int N = 4;
    HANDLE threads[N];
    
    for (int i = 0; i < N; i++) {
        threads[i] = CreateThread(NULL, 0, Worker, (LPVOID)(INT_PTR)(i+1), 0, NULL);
    }
    
    WaitForMultipleObjects(N, threads, TRUE, INFINITE);
    
    printf("\nExpected: %d\n", N * 1000);
    printf("Actual:   %d\n", sharedCounter);
    printf("Result:   %s\n", sharedCounter == N * 1000 ? "CORRECT" : "RACE!");
    
    for (int i = 0; i < N; i++) CloseHandle(threads[i]);
    CloseHandle(hMutex);
    
    return 0;
}''',
[{"type": "output_contains", "value": "Result: CORRECT"}])

# --- Semaphores ---
add(2, "sync", "semaphore", "CreateSemaphore", "intermediate",
"Write a C++ program demonstrating semaphore-based resource limiting.",
r'''#include <windows.h>
#include <stdio.h>

HANDLE hSem;
volatile LONG activeCount = 0;

DWORD WINAPI Worker(LPVOID arg) {
    int id = (int)(INT_PTR)arg;
    
    WaitForSingleObject(hSem, INFINITE);
    
    LONG count = InterlockedIncrement(&activeCount);
    printf("[Thread %d] Acquired (active: %ld)\n", id, count);
    
    Sleep(100);  // Simulate work
    
    InterlockedDecrement(&activeCount);
    ReleaseSemaphore(hSem, 1, NULL);
    printf("[Thread %d] Released\n", id);
    
    return 0;
}

int main() {
    printf("Semaphore Demo (max 2 concurrent)\n\n");
    
    // Create semaphore: initial=2, max=2
    hSem = CreateSemaphoreA(NULL, 2, 2, NULL);
    printf("Semaphore created (max 2)\n\n");
    
    const int N = 6;
    HANDLE threads[N];
    
    for (int i = 0; i < N; i++) {
        threads[i] = CreateThread(NULL, 0, Worker, (LPVOID)(INT_PTR)(i+1), 0, NULL);
        Sleep(20);  // Stagger starts
    }
    
    WaitForMultipleObjects(N, threads, TRUE, INFINITE);
    
    for (int i = 0; i < N; i++) CloseHandle(threads[i]);
    CloseHandle(hSem);
    
    printf("\nAll threads completed\n");
    return 0;
}''',
[{"type": "output_contains", "value": "Acquired"},
 {"type": "output_contains", "value": "All threads completed"}])

# --- Console ---
add(2, "console", "title", "SetConsoleTitle", "intermediate",
"Write a C++ program that manipulates console properties.",
r'''#include <windows.h>
#include <stdio.h>

int main() {
    printf("Console Properties Demo\n\n");
    
    HANDLE hCon = GetStdHandle(STD_OUTPUT_HANDLE);
    
    // Get console title
    char title[256];
    GetConsoleTitleA(title, sizeof(title));
    printf("Original title: %s\n", title);
    
    // Set new title
    SetConsoleTitleA("Windows Systems Programming Test");
    printf("Title changed\n");
    
    // Get console info
    CONSOLE_SCREEN_BUFFER_INFO csbi;
    GetConsoleScreenBufferInfo(hCon, &csbi);
    
    printf("\nScreen Buffer Info:\n");
    printf("  Buffer size: %dx%d\n", csbi.dwSize.X, csbi.dwSize.Y);
    printf("  Window size: %dx%d\n", 
        csbi.srWindow.Right - csbi.srWindow.Left + 1,
        csbi.srWindow.Bottom - csbi.srWindow.Top + 1);
    printf("  Cursor pos: (%d, %d)\n", csbi.dwCursorPosition.X, csbi.dwCursorPosition.Y);
    printf("  Attributes: 0x%04X\n", csbi.wAttributes);
    
    // Restore title
    SetConsoleTitleA(title);
    printf("\nTitle restored\n");
    
    return 0;
}''',
[{"type": "output_contains", "value": "Buffer size:"},
 {"type": "output_contains", "value": "Title restored"}])

# --- Heap Operations ---
add(2, "memory", "heap", "HeapCreate", "intermediate",
"Write a C++ program that creates a private heap and manages allocations.",
r'''#include <windows.h>
#include <stdio.h>

int main() {
    printf("Private Heap Demo\n\n");
    
    HANDLE hHeap = HeapCreate(0, 0x10000, 0);
    if (!hHeap) {
        printf("HeapCreate failed\n");
        return 1;
    }
    printf("Heap created: 0x%p\n", hHeap);
    
    // Allocate blocks
    LPVOID p1 = HeapAlloc(hHeap, HEAP_ZERO_MEMORY, 100);
    LPVOID p2 = HeapAlloc(hHeap, HEAP_ZERO_MEMORY, 200);
    LPVOID p3 = HeapAlloc(hHeap, HEAP_ZERO_MEMORY, 300);
    
    printf("Allocated: 0x%p (100), 0x%p (200), 0x%p (300)\n", p1, p2, p3);
    
    // Get sizes
    SIZE_T s1 = HeapSize(hHeap, 0, p1);
    SIZE_T s2 = HeapSize(hHeap, 0, p2);
    SIZE_T s3 = HeapSize(hHeap, 0, p3);
    printf("Sizes: %zu, %zu, %zu\n", s1, s2, s3);
    
    // Realloc
    p2 = HeapReAlloc(hHeap, HEAP_ZERO_MEMORY, p2, 500);
    SIZE_T s2new = HeapSize(hHeap, 0, p2);
    printf("After realloc p2: 0x%p (size %zu)\n", p2, s2new);
    
    // Free and destroy
    HeapFree(hHeap, 0, p1);
    HeapFree(hHeap, 0, p2);
    HeapFree(hHeap, 0, p3);
    HeapDestroy(hHeap);
    
    printf("Heap destroyed\n");
    return 0;
}''',
[{"type": "output_contains", "value": "Heap created:"},
 {"type": "output_contains", "value": "Heap destroyed"}])

# --- Process Enumeration ---
add(2, "process", "enum", "CreateToolhelp32Snapshot", "intermediate",
"Write a C++ program that enumerates processes and finds explorer.exe.",
r'''#include <windows.h>
#include <tlhelp32.h>
#include <stdio.h>

int main() {
    printf("Process Enumeration Demo\n\n");
    
    HANDLE hSnap = CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0);
    if (hSnap == INVALID_HANDLE_VALUE) {
        printf("Snapshot failed: %lu\n", GetLastError());
        return 1;
    }
    
    PROCESSENTRY32 pe = {sizeof(pe)};
    DWORD explorerPid = 0;
    int count = 0;
    
    printf("%-8s %-8s %s\n", "PID", "PPID", "Name");
    printf("-------- -------- --------\n");
    
    if (Process32First(hSnap, &pe)) {
        do {
            if (count < 15) {
                printf("%-8lu %-8lu %s\n", pe.th32ProcessID, 
                    pe.th32ParentProcessID, pe.szExeFile);
            }
            
            if (_stricmp(pe.szExeFile, "explorer.exe") == 0) {
                explorerPid = pe.th32ProcessID;
            }
            count++;
        } while (Process32Next(hSnap, &pe));
    }
    
    CloseHandle(hSnap);
    
    printf("\nTotal processes: %d\n", count);
    if (explorerPid) {
        printf("explorer.exe PID: %lu\n", explorerPid);
    }
    
    return 0;
}''',
[{"type": "output_contains", "value": "Total processes:"},
 {"type": "output_contains", "value": "explorer.exe PID:"}])

# --- Module Enumeration ---
add(2, "process", "modules", "EnumProcessModules", "intermediate",
"Write a C++ program that enumerates loaded modules in the current process.",
r'''#include <windows.h>
#include <psapi.h>
#include <stdio.h>
#pragma comment(lib, "psapi.lib")

int main() {
    printf("Module Enumeration Demo\n\n");
    
    HANDLE hProc = GetCurrentProcess();
    HMODULE modules[256];
    DWORD needed;
    
    if (!EnumProcessModules(hProc, modules, sizeof(modules), &needed)) {
        printf("EnumProcessModules failed: %lu\n", GetLastError());
        return 1;
    }
    
    int count = needed / sizeof(HMODULE);
    printf("Loaded modules: %d\n\n", count);
    
    printf("%-18s %-10s %s\n", "Base", "Size", "Name");
    printf("------------------ ---------- ----\n");
    
    for (int i = 0; i < count && i < 15; i++) {
        char name[MAX_PATH];
        MODULEINFO mi;
        
        GetModuleFileNameExA(hProc, modules[i], name, MAX_PATH);
        GetModuleInformation(hProc, modules[i], &mi, sizeof(mi));
        
        char* fname = strrchr(name, '\\');
        fname = fname ? fname + 1 : name;
        
        printf("0x%p 0x%08lX %s\n", mi.lpBaseOfDll, mi.SizeOfImage, fname);
    }
    
    return 0;
}''',
[{"type": "output_contains", "value": "Loaded modules:"},
 {"type": "output_contains", "value": "ntdll.dll"}])

# --- Job Objects ---
add(2, "process", "job", "CreateJobObject", "intermediate",
"Write a C++ program that creates a job object and queries its information.",
r'''#include <windows.h>
#include <stdio.h>

int main() {
    printf("Job Object Demo\n\n");
    
    HANDLE hJob = CreateJobObjectA(NULL, "TestJob");
    if (!hJob) {
        printf("CreateJobObject failed: %lu\n", GetLastError());
        return 1;
    }
    printf("Job created: 0x%p\n", hJob);
    
    // Assign current process to job
    if (!AssignProcessToJobObject(hJob, GetCurrentProcess())) {
        DWORD err = GetLastError();
        if (err == ERROR_ACCESS_DENIED) {
            printf("Already in a job (Windows 8+)\n");
        } else {
            printf("AssignProcess failed: %lu\n", err);
        }
    } else {
        printf("Process assigned to job\n");
    }
    
    // Query job info
    JOBOBJECT_BASIC_ACCOUNTING_INFORMATION info;
    if (QueryInformationJobObject(hJob, JobObjectBasicAccountingInformation,
            &info, sizeof(info), NULL)) {
        printf("\nJob Accounting:\n");
        printf("  Total Processes: %lu\n", info.TotalProcesses);
        printf("  Active Processes: %lu\n", info.ActiveProcesses);
        printf("  Total User Time: %llu\n", info.TotalUserTime.QuadPart);
        printf("  Total Kernel Time: %llu\n", info.TotalKernelTime.QuadPart);
    }
    
    CloseHandle(hJob);
    printf("\nJob closed\n");
    
    return 0;
}''',
[{"type": "output_contains", "value": "Job created:"},
 {"type": "output_contains", "value": "Job Accounting:"}])

# =============================================================================
# TIER 3: INTERMEDIATE (Medium-Hard) - Verifiable via structured output
# =============================================================================

# --- Tokens and Security ---
add(3, "security", "token", "OpenProcessToken", "intermediate",
"Write a C++ program that queries the process token for user SID and privileges.",
r'''#include <windows.h>
#include <sddl.h>
#include <stdio.h>
#pragma comment(lib, "advapi32.lib")

int main() {
    printf("Token Query Demo\n\n");
    
    HANDLE hToken;
    if (!OpenProcessToken(GetCurrentProcess(), TOKEN_QUERY, &hToken)) {
        printf("OpenProcessToken failed\n");
        return 1;
    }
    
    DWORD len;
    
    // User SID
    BYTE userBuf[256];
    TOKEN_USER* pUser = (TOKEN_USER*)userBuf;
    if (GetTokenInformation(hToken, TokenUser, pUser, sizeof(userBuf), &len)) {
        LPSTR sidStr;
        ConvertSidToStringSidA(pUser->User.Sid, &sidStr);
        printf("User SID: %s\n", sidStr);
        LocalFree(sidStr);
    }
    
    // Integrity level
    BYTE intBuf[256];
    TOKEN_MANDATORY_LABEL* pLabel = (TOKEN_MANDATORY_LABEL*)intBuf;
    if (GetTokenInformation(hToken, TokenIntegrityLevel, pLabel, sizeof(intBuf), &len)) {
        DWORD level = *GetSidSubAuthority(pLabel->Label.Sid, 
            *GetSidSubAuthorityCount(pLabel->Label.Sid) - 1);
        printf("Integrity Level: %s (0x%lX)\n",
            level >= 0x4000 ? "System" : level >= 0x3000 ? "High" :
            level >= 0x2000 ? "Medium" : "Low", level);
    }
    
    // Privileges
    BYTE privBuf[4096];
    TOKEN_PRIVILEGES* pPrivs = (TOKEN_PRIVILEGES*)privBuf;
    if (GetTokenInformation(hToken, TokenPrivileges, pPrivs, sizeof(privBuf), &len)) {
        printf("Privileges: %lu\n", pPrivs->PrivilegeCount);
        printf("  SeChangeNotifyPrivilege: ");
        for (DWORD i = 0; i < pPrivs->PrivilegeCount; i++) {
            char name[64];
            DWORD nameLen = 64;
            LookupPrivilegeNameA(NULL, &pPrivs->Privileges[i].Luid, name, &nameLen);
            if (strcmp(name, "SeChangeNotifyPrivilege") == 0) {
                printf("FOUND\n");
                break;
            }
        }
    }
    
    CloseHandle(hToken);
    return 0;
}''',
[{"type": "output_contains", "value": "User SID:"},
 {"type": "output_contains", "value": "SeChangeNotifyPrivilege: FOUND"}])

# --- Service Enumeration ---
add(3, "services", "enum", "EnumServicesStatusEx", "intermediate",
"Write a C++ program that enumerates running Windows services.",
r'''#include <windows.h>
#include <stdio.h>
#pragma comment(lib, "advapi32.lib")

int main() {
    printf("Service Enumeration Demo\n\n");
    
    SC_HANDLE hSCM = OpenSCManagerA(NULL, NULL, SC_MANAGER_ENUMERATE_SERVICE);
    if (!hSCM) {
        printf("OpenSCManager failed: %lu\n", GetLastError());
        return 1;
    }
    
    DWORD needed, count, resume = 0;
    EnumServicesStatusExA(hSCM, SC_ENUM_PROCESS_INFO, SERVICE_WIN32,
        SERVICE_ACTIVE, NULL, 0, &needed, &count, &resume, NULL);
    
    BYTE* buf = (BYTE*)malloc(needed);
    ENUM_SERVICE_STATUS_PROCESSA* services = (ENUM_SERVICE_STATUS_PROCESSA*)buf;
    
    if (!EnumServicesStatusExA(hSCM, SC_ENUM_PROCESS_INFO, SERVICE_WIN32,
            SERVICE_ACTIVE, buf, needed, &needed, &count, &resume, NULL)) {
        printf("EnumServices failed: %lu\n", GetLastError());
        free(buf);
        CloseServiceHandle(hSCM);
        return 1;
    }
    
    printf("Running services: %lu\n\n", count);
    printf("%-30s %-8s %s\n", "Name", "PID", "Display");
    printf("------------------------------ -------- --------\n");
    
    for (DWORD i = 0; i < count && i < 15; i++) {
        printf("%-30s %-8lu %.30s\n",
            services[i].lpServiceName,
            services[i].ServiceStatusProcess.dwProcessId,
            services[i].lpDisplayName);
    }
    
    free(buf);
    CloseServiceHandle(hSCM);
    
    printf("\n(Limited to 15 entries)\n");
    return 0;
}''',
[{"type": "output_contains", "value": "Running services:"},
 {"type": "output_contains", "value": "PID"}])

# --- Thread Pool ---
add(3, "threading", "pool", "QueueUserWorkItem", "intermediate",
"Write a C++ program demonstrating the legacy thread pool API.",
r'''#include <windows.h>
#include <stdio.h>

volatile LONG completedCount = 0;
HANDLE hEvent;

DWORD CALLBACK WorkCallback(PVOID context) {
    int id = (int)(INT_PTR)context;
    printf("[Work %d] Executing on thread %lu\n", id, GetCurrentThreadId());
    Sleep(50);
    
    if (InterlockedIncrement(&completedCount) == 5) {
        SetEvent(hEvent);
    }
    return 0;
}

int main() {
    printf("Thread Pool Demo\n\n");
    
    hEvent = CreateEventA(NULL, TRUE, FALSE, NULL);
    
    printf("Queuing 5 work items...\n\n");
    
    for (int i = 1; i <= 5; i++) {
        if (!QueueUserWorkItem(WorkCallback, (PVOID)(INT_PTR)i, WT_EXECUTEDEFAULT)) {
            printf("QueueUserWorkItem failed: %lu\n", GetLastError());
        }
    }
    
    WaitForSingleObject(hEvent, 5000);
    
    printf("\nAll %ld items completed\n", completedCount);
    CloseHandle(hEvent);
    
    return 0;
}''',
[{"type": "output_contains", "value": "Queuing 5 work items"},
 {"type": "output_contains", "value": "All 5 items completed"}])

# --- Fibers ---
add(3, "threading", "fiber", "ConvertThreadToFiber", "intermediate",
"Write a C++ program demonstrating fiber creation and switching.",
r'''#include <windows.h>
#include <stdio.h>

LPVOID mainFiber;
LPVOID workerFiber;
int counter = 0;

VOID CALLBACK FiberProc(LPVOID param) {
    int id = (int)(INT_PTR)param;
    
    for (int i = 0; i < 3; i++) {
        printf("[Fiber %d] Iteration %d (counter=%d)\n", id, i, counter++);
        SwitchToFiber(mainFiber);
    }
    
    printf("[Fiber %d] Done\n", id);
    SwitchToFiber(mainFiber);
}

int main() {
    printf("Fiber Demo\n\n");
    
    mainFiber = ConvertThreadToFiber(NULL);
    if (!mainFiber) {
        printf("ConvertThreadToFiber failed: %lu\n", GetLastError());
        return 1;
    }
    printf("Main fiber: 0x%p\n", mainFiber);
    
    workerFiber = CreateFiber(0, FiberProc, (LPVOID)1);
    printf("Worker fiber: 0x%p\n\n", workerFiber);
    
    for (int i = 0; i < 4; i++) {
        printf("[Main] Switching to worker (counter=%d)\n", counter);
        SwitchToFiber(workerFiber);
    }
    
    DeleteFiber(workerFiber);
    ConvertFiberToThread();
    
    printf("\nFinal counter: %d\n", counter);
    return 0;
}''',
[{"type": "output_contains", "value": "Fiber Demo"},
 {"type": "output_contains", "value": "Final counter: 3"}])

# --- Vectored Exception Handler ---
add(3, "exception", "veh", "AddVectoredExceptionHandler", "intermediate",
"Write a C++ program demonstrating vectored exception handling.",
r'''#include <windows.h>
#include <stdio.h>

volatile LONG handlerCalled = 0;

LONG CALLBACK VehHandler(PEXCEPTION_POINTERS ep) {
    printf("[VEH] Exception caught: 0x%08lX\n", ep->ExceptionRecord->ExceptionCode);
    printf("[VEH] At address: 0x%p\n", ep->ExceptionRecord->ExceptionAddress);
    
    InterlockedIncrement(&handlerCalled);
    
    // For access violation, we can't continue
    if (ep->ExceptionRecord->ExceptionCode == EXCEPTION_ACCESS_VIOLATION) {
        return EXCEPTION_CONTINUE_SEARCH;
    }
    
    return EXCEPTION_CONTINUE_SEARCH;
}

int main() {
    printf("Vectored Exception Handler Demo\n\n");
    
    PVOID handler = AddVectoredExceptionHandler(1, VehHandler);
    printf("VEH installed: 0x%p\n\n", handler);
    
    // Trigger an exception using SEH to catch it
    __try {
        printf("Triggering divide by zero...\n");
        volatile int x = 0;
        volatile int y = 1 / x;
        (void)y;
    } __except(EXCEPTION_EXECUTE_HANDLER) {
        printf("[SEH] Exception handled\n");
    }
    
    printf("\nVEH was called: %s\n", handlerCalled > 0 ? "YES" : "NO");
    
    RemoveVectoredExceptionHandler(handler);
    printf("VEH removed\n");
    
    return 0;
}''',
[{"type": "output_contains", "value": "VEH installed:"},
 {"type": "output_contains", "value": "VEH was called: YES"}])

# --- PE Parsing ---
add(3, "pe", "headers", "IMAGE_DOS_HEADER", "intermediate",
"Write a C++ program that parses PE headers of the current executable.",
r'''#include <windows.h>
#include <stdio.h>

int main() {
    printf("PE Header Parser\n\n");
    
    BYTE* base = (BYTE*)GetModuleHandleA(NULL);
    PIMAGE_DOS_HEADER dos = (PIMAGE_DOS_HEADER)base;
    
    printf("DOS Header @ 0x%p\n", dos);
    printf("  e_magic:  0x%04X (%s)\n", dos->e_magic,
        dos->e_magic == IMAGE_DOS_SIGNATURE ? "MZ" : "?");
    printf("  e_lfanew: 0x%08lX\n", dos->e_lfanew);
    
    PIMAGE_NT_HEADERS nt = (PIMAGE_NT_HEADERS)(base + dos->e_lfanew);
    printf("\nNT Headers @ 0x%p\n", nt);
    printf("  Signature: 0x%08lX (%s)\n", nt->Signature,
        nt->Signature == IMAGE_NT_SIGNATURE ? "PE" : "?");
    
    printf("\nFile Header:\n");
    printf("  Machine: 0x%04X (%s)\n", nt->FileHeader.Machine,
        nt->FileHeader.Machine == IMAGE_FILE_MACHINE_AMD64 ? "x64" : "x86");
    printf("  Sections: %u\n", nt->FileHeader.NumberOfSections);
    
    printf("\nOptional Header:\n");
    printf("  Magic: 0x%04X (%s)\n", nt->OptionalHeader.Magic,
        nt->OptionalHeader.Magic == IMAGE_NT_OPTIONAL_HDR64_MAGIC ? "PE32+" : "PE32");
    printf("  ImageBase: 0x%p\n", (void*)nt->OptionalHeader.ImageBase);
    printf("  SizeOfImage: 0x%lX\n", nt->OptionalHeader.SizeOfImage);
    
    return 0;
}''',
[{"type": "output_contains", "value": "DOS Header @"},
 {"type": "output_contains", "value": "NT Headers @"}])

# --- PE Sections ---
add(3, "pe", "sections", "IMAGE_SECTION_HEADER", "intermediate",
"Write a C++ program that enumerates PE sections.",
r'''#include <windows.h>
#include <stdio.h>

int main() {
    printf("PE Section Enumeration\n\n");
    
    BYTE* base = (BYTE*)GetModuleHandleA(NULL);
    PIMAGE_DOS_HEADER dos = (PIMAGE_DOS_HEADER)base;
    PIMAGE_NT_HEADERS nt = (PIMAGE_NT_HEADERS)(base + dos->e_lfanew);
    
    PIMAGE_SECTION_HEADER sec = IMAGE_FIRST_SECTION(nt);
    int count = nt->FileHeader.NumberOfSections;
    
    printf("%-8s %-10s %-10s %s\n", "Name", "VirtAddr", "Size", "Flags");
    printf("-------- ---------- ---------- -----\n");
    
    for (int i = 0; i < count; i++) {
        char name[9] = {0};
        memcpy(name, sec[i].Name, 8);
        
        char flags[16] = "";
        DWORD c = sec[i].Characteristics;
        if (c & IMAGE_SCN_MEM_EXECUTE) strcat(flags, "X");
        if (c & IMAGE_SCN_MEM_READ) strcat(flags, "R");
        if (c & IMAGE_SCN_MEM_WRITE) strcat(flags, "W");
        
        printf("%-8s 0x%08lX 0x%08lX %s\n",
            name, sec[i].VirtualAddress, sec[i].Misc.VirtualSize, flags);
    }
    
    printf("\nTotal sections: %d\n", count);
    return 0;
}''',
[{"type": "output_contains", "value": ".text"},
 {"type": "output_contains", "value": "Total sections:"}])

# =============================================================================
# TIER 4: ADVANCED (Hard) - Verifiable via behavioral checks
# =============================================================================

# --- Native API ---
add(4, "native", "ntdll", "NtQuerySystemInformation", "advanced",
"Write a C++ program that uses NtQuerySystemInformation to enumerate processes.",
r'''#include <windows.h>
#include <stdio.h>

typedef LONG NTSTATUS;
#define NT_SUCCESS(s) ((NTSTATUS)(s) >= 0)

typedef struct _UNICODE_STRING {
    USHORT Length, MaxLength;
    PWSTR Buffer;
} UNICODE_STRING;

typedef struct _SYSTEM_PROCESS_INFO {
    ULONG NextOffset;
    ULONG ThreadCount;
    LARGE_INTEGER Reserved[3];
    LARGE_INTEGER CreateTime, UserTime, KernelTime;
    UNICODE_STRING ImageName;
    LONG BasePriority;
    HANDLE UniqueProcessId;
    HANDLE ParentProcessId;
    ULONG HandleCount;
} SYSTEM_PROCESS_INFO;

typedef NTSTATUS (NTAPI *pNtQuerySystemInformation)(ULONG, PVOID, ULONG, PULONG);

int main() {
    printf("NtQuerySystemInformation Demo\n\n");
    
    HMODULE ntdll = GetModuleHandleA("ntdll.dll");
    pNtQuerySystemInformation NtQSI = (pNtQuerySystemInformation)
        GetProcAddress(ntdll, "NtQuerySystemInformation");
    
    printf("NtQuerySystemInformation @ 0x%p\n\n", NtQSI);
    
    ULONG size = 0;
    NtQSI(5, NULL, 0, &size);
    
    PVOID buf = malloc(size * 2);
    NTSTATUS status = NtQSI(5, buf, size * 2, &size);
    
    if (!NT_SUCCESS(status)) {
        printf("Failed: 0x%lX\n", status);
        free(buf);
        return 1;
    }
    
    printf("%-8s %-6s %s\n", "PID", "Thds", "Name");
    printf("-------- ------ ----\n");
    
    SYSTEM_PROCESS_INFO* p = (SYSTEM_PROCESS_INFO*)buf;
    int count = 0;
    
    while (count < 15) {
        wchar_t name[256] = L"[System]";
        if (p->ImageName.Buffer) wcsncpy(name, p->ImageName.Buffer, 255);
        
        printf("%-8llu %-6lu %ws\n",
            (ULONGLONG)p->UniqueProcessId, p->ThreadCount, name);
        
        if (!p->NextOffset) break;
        p = (SYSTEM_PROCESS_INFO*)((BYTE*)p + p->NextOffset);
        count++;
    }
    
    free(buf);
    return 0;
}''',
[{"type": "output_contains", "value": "NtQuerySystemInformation @"},
 {"type": "output_contains", "value": "PID"}])

# --- Syscall Resolution ---
add(4, "native", "syscall", "syscall_resolve", "advanced",
"Write a C++ program that resolves syscall numbers from ntdll.dll.",
r'''#include <windows.h>
#include <stdio.h>

DWORD GetSyscallNumber(const char* name) {
    HMODULE ntdll = GetModuleHandleA("ntdll.dll");
    BYTE* func = (BYTE*)GetProcAddress(ntdll, name);
    if (!func) return 0;
    
    // x64 pattern: 4C 8B D1 B8 XX XX 00 00
    if (func[0] == 0x4C && func[1] == 0x8B && func[2] == 0xD1 && func[3] == 0xB8) {
        return *(DWORD*)(func + 4);
    }
    
    if (func[0] == 0xE9 || func[0] == 0xFF) {
        return 0xFFFFFFFF;  // Hooked
    }
    
    return 0;
}

int main() {
    printf("Syscall Number Resolution\n\n");
    printf("%-35s %s\n", "Function", "Syscall #");
    printf("----------------------------------- ---------\n");
    
    const char* funcs[] = {
        "NtAllocateVirtualMemory", "NtProtectVirtualMemory",
        "NtWriteVirtualMemory", "NtReadVirtualMemory",
        "NtOpenProcess", "NtClose",
        "NtQuerySystemInformation", "NtCreateFile"
    };
    
    for (int i = 0; i < sizeof(funcs)/sizeof(funcs[0]); i++) {
        DWORD num = GetSyscallNumber(funcs[i]);
        if (num == 0xFFFFFFFF) {
            printf("%-35s HOOKED\n", funcs[i]);
        } else if (num > 0) {
            printf("%-35s 0x%04lX\n", funcs[i], num);
        } else {
            printf("%-35s N/A\n", funcs[i]);
        }
    }
    
    return 0;
}''',
[{"type": "output_contains", "value": "NtAllocateVirtualMemory"},
 {"type": "output_contains", "value": "0x"}])

# --- PEB Walking ---
add(4, "internals", "peb", "PEB_walk", "advanced",
"Write a C++ program that walks the PEB to enumerate modules.",
r'''#include <windows.h>
#include <stdio.h>
#include <intrin.h>

typedef struct _UNICODE_STRING { USHORT Length, MaxLength; PWSTR Buffer; } UNICODE_STRING;

typedef struct _LDR_DATA_TABLE_ENTRY {
    LIST_ENTRY InLoadOrderLinks;
    LIST_ENTRY InMemoryOrderLinks;
    LIST_ENTRY InInitOrderLinks;
    PVOID DllBase;
    PVOID EntryPoint;
    ULONG SizeOfImage;
    UNICODE_STRING FullDllName;
    UNICODE_STRING BaseDllName;
} LDR_DATA_TABLE_ENTRY;

typedef struct _PEB_LDR_DATA {
    ULONG Length;
    BOOLEAN Initialized;
    HANDLE SsHandle;
    LIST_ENTRY InLoadOrderModuleList;
} PEB_LDR_DATA;

typedef struct _PEB {
    BYTE Reserved1[2];
    BYTE BeingDebugged;
    BYTE Reserved2[1];
    PVOID Reserved3[2];
    PEB_LDR_DATA* Ldr;
} PEB;

int main() {
    printf("PEB Module Walk\n\n");
    
    PEB* peb = (PEB*)__readgsqword(0x60);
    printf("PEB @ 0x%p\n", peb);
    printf("BeingDebugged: %d\n\n", peb->BeingDebugged);
    
    LIST_ENTRY* head = &peb->Ldr->InLoadOrderModuleList;
    LIST_ENTRY* curr = head->Flink;
    
    printf("%-18s %-10s %s\n", "Base", "Size", "Name");
    printf("------------------ ---------- ----\n");
    
    int count = 0;
    while (curr != head && count < 15) {
        LDR_DATA_TABLE_ENTRY* entry = CONTAINING_RECORD(
            curr, LDR_DATA_TABLE_ENTRY, InLoadOrderLinks);
        
        if (entry->DllBase) {
            printf("0x%p 0x%08lX %.*ws\n",
                entry->DllBase, entry->SizeOfImage,
                entry->BaseDllName.Length / 2, entry->BaseDllName.Buffer);
            count++;
        }
        curr = curr->Flink;
    }
    
    return 0;
}''',
[{"type": "output_contains", "value": "PEB @"},
 {"type": "output_contains", "value": "ntdll.dll"}])

# --- TEB Access ---
add(4, "internals", "teb", "TEB_access", "advanced",
"Write a C++ program that accesses TEB fields directly.",
r'''#include <windows.h>
#include <stdio.h>
#include <intrin.h>

typedef struct _MY_CLIENT_ID { HANDLE Process; HANDLE Thread; } MY_CLIENT_ID;

typedef struct _MY_TEB {
    NT_TIB NtTib;  // Use the SDK's NT_TIB
    PVOID EnvironmentPointer;
    MY_CLIENT_ID ClientId;
    PVOID ActiveRpcHandle;
    PVOID ThreadLocalStoragePointer;
    PVOID ProcessEnvironmentBlock;
    DWORD LastErrorValue;
} MY_TEB;

int main() {
    printf("TEB Access Demo\n\n");
    
    MY_TEB* teb = (MY_TEB*)__readgsqword(0x30);
    
    printf("TEB @ 0x%p\n\n", teb);
    
    printf("[NT_TIB]\n");
    printf("  StackBase:  0x%p\n", teb->NtTib.StackBase);
    printf("  StackLimit: 0x%p\n", teb->NtTib.StackLimit);
    printf("  Stack Size: %zu KB\n\n",
        ((SIZE_T)teb->NtTib.StackBase - (SIZE_T)teb->NtTib.StackLimit) / 1024);
    
    printf("[TEB]\n");
    printf("  PID: %llu\n", (ULONGLONG)teb->ClientId.Process);
    printf("  TID: %llu\n", (ULONGLONG)teb->ClientId.Thread);
    printf("  PEB: 0x%p\n", teb->ProcessEnvironmentBlock);
    
    // Verify
    printf("\n[Verification]\n");
    printf("  GetCurrentProcessId: %lu\n", GetCurrentProcessId());
    printf("  GetCurrentThreadId:  %lu\n", GetCurrentThreadId());
    
    return 0;
}''',
[{"type": "output_contains", "value": "TEB @"},
 {"type": "output_contains", "value": "PID:"}])

# --- APC Queuing ---
add(4, "threading", "apc", "QueueUserAPC", "advanced",
"Write a C++ program demonstrating user-mode APC queuing.",
r'''#include <windows.h>
#include <stdio.h>

volatile LONG apcExecuted = 0;

VOID CALLBACK ApcCallback(ULONG_PTR param) {
    printf("[APC] Executed with param: %llu\n", (ULONGLONG)param);
    InterlockedIncrement(&apcExecuted);
}

DWORD WINAPI AlertableThread(LPVOID arg) {
    printf("[Thread] Entering alertable wait...\n");
    SleepEx(INFINITE, TRUE);  // Alertable wait
    printf("[Thread] Woke up from alertable wait\n");
    return 0;
}

int main() {
    printf("User APC Demo\n\n");
    
    // Create alertable thread
    HANDLE hThread = CreateThread(NULL, 0, AlertableThread, NULL, 0, NULL);
    Sleep(100);  // Let thread enter wait
    
    // Queue APC
    printf("[Main] Queuing APC...\n");
    if (QueueUserAPC(ApcCallback, hThread, 0x12345678)) {
        printf("[Main] APC queued successfully\n");
    }
    
    WaitForSingleObject(hThread, 2000);
    CloseHandle(hThread);
    
    printf("\nAPC executed: %s\n", apcExecuted > 0 ? "YES" : "NO");
    
    return 0;
}''',
[{"type": "output_contains", "value": "APC queued successfully"},
 {"type": "output_contains", "value": "APC executed: YES"}])

# --- Anti-Debug ---
add(4, "evasion", "antidebug", "IsDebuggerPresent", "advanced",
"Write a C++ program demonstrating debugger detection techniques.",
r'''#include <windows.h>
#include <stdio.h>
#include <intrin.h>

int main() {
    printf("Debugger Detection Demo\n\n");
    
    // 1. IsDebuggerPresent
    printf("[1] IsDebuggerPresent: %s\n",
        IsDebuggerPresent() ? "DETECTED" : "Clean");
    
    // 2. CheckRemoteDebuggerPresent
    BOOL remote = FALSE;
    CheckRemoteDebuggerPresent(GetCurrentProcess(), &remote);
    printf("[2] Remote debugger: %s\n", remote ? "DETECTED" : "Clean");
    
    // 3. PEB.BeingDebugged
    BYTE* peb = (BYTE*)__readgsqword(0x60);
    printf("[3] PEB.BeingDebugged: %s\n", peb[2] ? "DETECTED" : "Clean");
    
    // 4. NtGlobalFlag
    DWORD ntGlobalFlag = *(DWORD*)(peb + 0xBC);
    printf("[4] NtGlobalFlag: %s (0x%lX)\n",
        (ntGlobalFlag & 0x70) ? "DETECTED" : "Clean", ntGlobalFlag);
    
    // 5. Timing check
    LARGE_INTEGER freq, start, end;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&start);
    volatile int x = 0;
    for (int i = 0; i < 1000000; i++) x++;
    QueryPerformanceCounter(&end);
    double ms = (double)(end.QuadPart - start.QuadPart) / freq.QuadPart * 1000;
    printf("[5] Timing: %.2f ms %s\n", ms, ms > 50 ? "(Suspicious)" : "(Normal)");
    
    return 0;
}''',
[{"type": "output_contains", "value": "IsDebuggerPresent:"},
 {"type": "output_contains", "value": "PEB.BeingDebugged:"}])

# =============================================================================
# ADDITIONAL TIER 1 PROBLEMS
# =============================================================================

add(1, "file", "path", "GetFullPathName", "beginner",
"Write a C++ program that converts relative paths to absolute paths.",
r'''#include <windows.h>
#include <stdio.h>

int main() {
    printf("Path Resolution Demo\n\n");
    
    const char* paths[] = {".", "..", "test.txt", ".\\sub\\file.txt"};
    char fullPath[MAX_PATH];
    
    for (int i = 0; i < 4; i++) {
        DWORD len = GetFullPathNameA(paths[i], MAX_PATH, fullPath, NULL);
        if (len > 0) {
            printf("%-20s -> %s\n", paths[i], fullPath);
        }
    }
    
    // Current directory
    char cwd[MAX_PATH];
    GetCurrentDirectoryA(MAX_PATH, cwd);
    printf("\nCurrent directory: %s\n", cwd);
    
    return 0;
}''',
[{"type": "output_contains", "value": "Current directory:"}])

add(1, "file", "temp", "GetTempFileName", "beginner",
"Write a C++ program that creates a unique temporary file.",
r'''#include <windows.h>
#include <stdio.h>

int main() {
    printf("Temp File Demo\n\n");
    
    char tempPath[MAX_PATH];
    char tempFile[MAX_PATH];
    
    GetTempPathA(MAX_PATH, tempPath);
    printf("Temp path: %s\n", tempPath);
    
    if (GetTempFileNameA(tempPath, "TMP", 0, tempFile)) {
        printf("Temp file: %s\n", tempFile);
        
        // Write to it
        HANDLE h = CreateFileA(tempFile, GENERIC_WRITE, 0, NULL,
            CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, NULL);
        if (h != INVALID_HANDLE_VALUE) {
            const char* data = "Test data";
            DWORD written;
            WriteFile(h, data, strlen(data), &written, NULL);
            CloseHandle(h);
            printf("Wrote %lu bytes\n", written);
        }
        
        DeleteFileA(tempFile);
        printf("Temp file deleted\n");
    }
    
    return 0;
}''',
[{"type": "output_contains", "value": "Temp file:"},
 {"type": "output_contains", "value": "Temp file deleted"}])

add(1, "sysinfo", "drives", "GetLogicalDrives", "beginner",
"Write a C++ program that lists all logical drives.",
r'''#include <windows.h>
#include <stdio.h>

int main() {
    printf("Logical Drives\n\n");
    
    DWORD drives = GetLogicalDrives();
    printf("Drive mask: 0x%08lX\n\n", drives);
    
    for (char c = 'A'; c <= 'Z'; c++) {
        if (drives & (1 << (c - 'A'))) {
            char root[4] = {c, ':', '\\', 0};
            UINT type = GetDriveTypeA(root);
            
            const char* typeStr;
            switch (type) {
                case DRIVE_FIXED: typeStr = "Fixed"; break;
                case DRIVE_REMOVABLE: typeStr = "Removable"; break;
                case DRIVE_REMOTE: typeStr = "Network"; break;
                case DRIVE_CDROM: typeStr = "CD-ROM"; break;
                case DRIVE_RAMDISK: typeStr = "RAM Disk"; break;
                default: typeStr = "Unknown";
            }
            
            printf("%s  %s\n", root, typeStr);
        }
    }
    
    return 0;
}''',
[{"type": "output_contains", "value": "C:\\"},
 {"type": "output_contains", "value": "Fixed"}])

add(1, "sysinfo", "diskspace", "GetDiskFreeSpaceEx", "beginner",
"Write a C++ program that displays disk space information.",
r'''#include <windows.h>
#include <stdio.h>

int main() {
    printf("Disk Space Info\n\n");
    
    ULARGE_INTEGER freeBytesAvail, totalBytes, totalFreeBytes;
    
    if (GetDiskFreeSpaceExA("C:\\", &freeBytesAvail, &totalBytes, &totalFreeBytes)) {
        printf("C: Drive:\n");
        printf("  Total:     %.2f GB\n", totalBytes.QuadPart / (1024.0*1024*1024));
        printf("  Free:      %.2f GB\n", totalFreeBytes.QuadPart / (1024.0*1024*1024));
        printf("  Available: %.2f GB\n", freeBytesAvail.QuadPart / (1024.0*1024*1024));
        printf("  Used:      %.2f GB\n", 
            (totalBytes.QuadPart - totalFreeBytes.QuadPart) / (1024.0*1024*1024));
        printf("  Usage:     %.1f%%\n", 
            100.0 * (totalBytes.QuadPart - totalFreeBytes.QuadPart) / totalBytes.QuadPart);
    }
    
    return 0;
}''',
[{"type": "output_contains", "value": "Total:"},
 {"type": "output_contains", "value": "Free:"}])

add(1, "process", "exit", "ExitProcess", "beginner",
"Write a C++ program demonstrating process exit codes.",
r'''#include <windows.h>
#include <stdio.h>

int main(int argc, char* argv[]) {
    printf("Exit Code Demo\n\n");
    
    int exitCode = argc > 1 ? atoi(argv[1]) : 42;
    
    printf("Process ID: %lu\n", GetCurrentProcessId());
    printf("Will exit with code: %d\n", exitCode);
    printf("ExitProcess will be called after main()\n");
    
    // In real code, you'd call ExitProcess(exitCode)
    // Here we just return to demonstrate
    return exitCode;
}''',
[{"type": "output_contains", "value": "Will exit with code:"}],
verification="exit_code")

add(1, "environment", "block", "GetEnvironmentStrings", "beginner",
"Write a C++ program that dumps all environment variables.",
r'''#include <windows.h>
#include <stdio.h>

int main() {
    printf("Environment Variables\n\n");
    
    LPCH envBlock = GetEnvironmentStringsA();
    if (!envBlock) {
        printf("GetEnvironmentStrings failed\n");
        return 1;
    }
    
    LPCH p = envBlock;
    int count = 0;
    
    while (*p && count < 15) {
        printf("%s\n", p);
        p += strlen(p) + 1;
        count++;
    }
    
    printf("\n... (showing first 15)\n");
    
    FreeEnvironmentStringsA(envBlock);
    return 0;
}''',
[{"type": "output_contains", "value": "="},
 {"type": "output_contains", "value": "showing first 15"}])

add(1, "string", "compare", "CompareString", "beginner",
"Write a C++ program demonstrating locale-aware string comparison.",
r'''#include <windows.h>
#include <stdio.h>

int main() {
    printf("String Comparison Demo\n\n");
    
    const char* str1 = "Hello";
    const char* str2 = "hello";
    const char* str3 = "World";
    
    int result = CompareStringA(LOCALE_USER_DEFAULT, 0, str1, -1, str2, -1);
    printf("\"%s\" vs \"%s\": %s\n", str1, str2,
        result == CSTR_LESS_THAN ? "<" :
        result == CSTR_EQUAL ? "=" : ">");
    
    result = CompareStringA(LOCALE_USER_DEFAULT, NORM_IGNORECASE, str1, -1, str2, -1);
    printf("\"%s\" vs \"%s\" (ignore case): %s\n", str1, str2,
        result == CSTR_LESS_THAN ? "<" :
        result == CSTR_EQUAL ? "=" : ">");
    
    result = CompareStringA(LOCALE_USER_DEFAULT, 0, str1, -1, str3, -1);
    printf("\"%s\" vs \"%s\": %s\n", str1, str3,
        result == CSTR_LESS_THAN ? "<" :
        result == CSTR_EQUAL ? "=" : ">");
    
    return 0;
}''',
[{"type": "output_contains", "value": "ignore case): ="}])

add(1, "time", "zone", "GetTimeZoneInformation", "beginner",
"Write a C++ program that displays time zone information.",
r'''#include <windows.h>
#include <stdio.h>

int main() {
    printf("Time Zone Information\n\n");
    
    TIME_ZONE_INFORMATION tzi;
    DWORD result = GetTimeZoneInformation(&tzi);
    
    printf("Status: %s\n", 
        result == TIME_ZONE_ID_STANDARD ? "Standard" :
        result == TIME_ZONE_ID_DAYLIGHT ? "Daylight" : "Unknown");
    
    printf("Bias: %ld minutes (UTC%+.1f)\n", tzi.Bias, -tzi.Bias / 60.0);
    wprintf(L"Standard Name: %s\n", tzi.StandardName);
    wprintf(L"Daylight Name: %s\n", tzi.DaylightName);
    
    // Dynamic time zone
    DYNAMIC_TIME_ZONE_INFORMATION dtzi;
    GetDynamicTimeZoneInformation(&dtzi);
    wprintf(L"Time Zone Key: %s\n", dtzi.TimeZoneKeyName);
    
    return 0;
}''',
[{"type": "output_contains", "value": "Bias:"},
 {"type": "output_contains", "value": "Time Zone Key:"}])

add(1, "registry", "write", "RegSetValueEx", "beginner",
"Write a C++ program that writes and reads a registry value.",
r'''#include <windows.h>
#include <stdio.h>
#pragma comment(lib, "advapi32.lib")

int main() {
    printf("Registry Write Demo\n\n");
    
    HKEY hKey;
    const char* subkey = "SOFTWARE\\HaloForgeTest";
    
    // Create/open key
    LONG result = RegCreateKeyExA(HKEY_CURRENT_USER, subkey, 0, NULL,
        REG_OPTION_NON_VOLATILE, KEY_ALL_ACCESS, NULL, &hKey, NULL);
    
    if (result != ERROR_SUCCESS) {
        printf("RegCreateKeyEx failed: %ld\n", result);
        return 1;
    }
    printf("Key created/opened\n");
    
    // Write string value
    const char* strVal = "TestValue123";
    RegSetValueExA(hKey, "TestString", 0, REG_SZ, (BYTE*)strVal, strlen(strVal)+1);
    printf("Wrote string: %s\n", strVal);
    
    // Write DWORD value
    DWORD dwordVal = 42;
    RegSetValueExA(hKey, "TestDword", 0, REG_DWORD, (BYTE*)&dwordVal, sizeof(dwordVal));
    printf("Wrote DWORD: %lu\n", dwordVal);
    
    // Read back
    char readStr[256];
    DWORD readSize = sizeof(readStr);
    RegQueryValueExA(hKey, "TestString", NULL, NULL, (BYTE*)readStr, &readSize);
    printf("Read back: %s\n", readStr);
    
    // Cleanup
    RegCloseKey(hKey);
    RegDeleteKeyA(HKEY_CURRENT_USER, subkey);
    printf("Key deleted\n");
    
    return 0;
}''',
[{"type": "output_contains", "value": "Wrote string:"},
 {"type": "output_contains", "value": "Key deleted"}])

# =============================================================================
# ADDITIONAL TIER 2 PROBLEMS
# =============================================================================

add(2, "file", "mapping", "CreateFileMapping", "intermediate",
"Write a C++ program demonstrating memory-mapped file I/O.",
r'''#include <windows.h>
#include <stdio.h>

int main() {
    printf("Memory-Mapped File Demo\n\n");
    
    HANDLE hFile = CreateFileA("mapped_test.dat", 
        GENERIC_READ | GENERIC_WRITE, 0, NULL,
        CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, NULL);
    
    if (hFile == INVALID_HANDLE_VALUE) {
        printf("CreateFile failed\n");
        return 1;
    }
    
    DWORD mapSize = 4096;
    HANDLE hMap = CreateFileMappingA(hFile, NULL, PAGE_READWRITE, 0, mapSize, NULL);
    
    if (!hMap) {
        printf("CreateFileMapping failed\n");
        CloseHandle(hFile);
        return 1;
    }
    printf("File mapping created\n");
    
    LPVOID pView = MapViewOfFile(hMap, FILE_MAP_ALL_ACCESS, 0, 0, mapSize);
    printf("Mapped at: 0x%p\n", pView);
    
    const char* data = "Data via memory mapping!";
    memcpy(pView, data, strlen(data) + 1);
    printf("Wrote: %s\n", (char*)pView);
    
    FlushViewOfFile(pView, mapSize);
    UnmapViewOfFile(pView);
    CloseHandle(hMap);
    CloseHandle(hFile);
    DeleteFileA("mapped_test.dat");
    
    printf("File mapping closed\n");
    return 0;
}''',
[{"type": "output_contains", "value": "Data via memory mapping!"}])

add(2, "sync", "srwlock", "InitializeSRWLock", "intermediate",
"Write a C++ program demonstrating slim reader/writer locks.",
r'''#include <windows.h>
#include <stdio.h>

SRWLOCK srwLock = SRWLOCK_INIT;
int sharedData = 0;

DWORD WINAPI Reader(LPVOID arg) {
    int id = (int)(INT_PTR)arg;
    for (int i = 0; i < 3; i++) {
        AcquireSRWLockShared(&srwLock);
        printf("[Reader %d] Read value: %d\n", id, sharedData);
        ReleaseSRWLockShared(&srwLock);
        Sleep(50);
    }
    return 0;
}

DWORD WINAPI Writer(LPVOID arg) {
    int id = (int)(INT_PTR)arg;
    for (int i = 0; i < 3; i++) {
        AcquireSRWLockExclusive(&srwLock);
        sharedData++;
        printf("[Writer %d] Wrote value: %d\n", id, sharedData);
        ReleaseSRWLockExclusive(&srwLock);
        Sleep(100);
    }
    return 0;
}

int main() {
    printf("SRW Lock Demo\n\n");
    
    HANDLE threads[4];
    threads[0] = CreateThread(NULL, 0, Writer, (LPVOID)1, 0, NULL);
    threads[1] = CreateThread(NULL, 0, Reader, (LPVOID)1, 0, NULL);
    threads[2] = CreateThread(NULL, 0, Reader, (LPVOID)2, 0, NULL);
    threads[3] = CreateThread(NULL, 0, Writer, (LPVOID)2, 0, NULL);
    
    WaitForMultipleObjects(4, threads, TRUE, INFINITE);
    
    for (int i = 0; i < 4; i++) CloseHandle(threads[i]);
    printf("\nFinal value: %d\n", sharedData);
    
    return 0;
}''',
[{"type": "output_contains", "value": "SRW Lock Demo"},
 {"type": "output_contains", "value": "Final value: 6"}])

add(2, "sync", "condvar", "InitializeConditionVariable", "intermediate",
"Write a C++ program demonstrating condition variables.",
r'''#include <windows.h>
#include <stdio.h>

CRITICAL_SECTION cs;
CONDITION_VARIABLE cv;
BOOL dataReady = FALSE;
int sharedValue = 0;

DWORD WINAPI Producer(LPVOID arg) {
    Sleep(100);
    EnterCriticalSection(&cs);
    sharedValue = 42;
    dataReady = TRUE;
    printf("[Producer] Data ready: %d\n", sharedValue);
    WakeConditionVariable(&cv);
    LeaveCriticalSection(&cs);
    return 0;
}

DWORD WINAPI Consumer(LPVOID arg) {
    int id = (int)(INT_PTR)arg;
    EnterCriticalSection(&cs);
    while (!dataReady) {
        printf("[Consumer %d] Waiting...\n", id);
        SleepConditionVariableCS(&cv, &cs, INFINITE);
    }
    printf("[Consumer %d] Got data: %d\n", id, sharedValue);
    LeaveCriticalSection(&cs);
    return 0;
}

int main() {
    printf("Condition Variable Demo\n\n");
    
    InitializeCriticalSection(&cs);
    InitializeConditionVariable(&cv);
    
    HANDLE t1 = CreateThread(NULL, 0, Consumer, (LPVOID)1, 0, NULL);
    HANDLE t2 = CreateThread(NULL, 0, Producer, NULL, 0, NULL);
    
    WaitForSingleObject(t1, 5000);
    WaitForSingleObject(t2, 5000);
    
    CloseHandle(t1);
    CloseHandle(t2);
    DeleteCriticalSection(&cs);
    
    printf("\nDemo complete\n");
    return 0;
}''',
[{"type": "output_contains", "value": "Got data: 42"}])

add(2, "ipc", "mailslot", "CreateMailslot", "intermediate",
"Write a C++ program demonstrating mailslot communication.",
r'''#include <windows.h>
#include <stdio.h>

int main() {
    printf("Mailslot Demo\n\n");
    
    const char* slotName = "\\\\.\\mailslot\\TestSlot";
    
    // Create mailslot (server)
    HANDLE hSlot = CreateMailslotA(slotName, 0, 1000, NULL);
    if (hSlot == INVALID_HANDLE_VALUE) {
        printf("CreateMailslot failed: %lu\n", GetLastError());
        return 1;
    }
    printf("Mailslot created\n");
    
    // Open for writing (client)
    HANDLE hClient = CreateFileA(slotName, GENERIC_WRITE, FILE_SHARE_READ,
        NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
    
    if (hClient == INVALID_HANDLE_VALUE) {
        printf("Client connect failed: %lu\n", GetLastError());
        CloseHandle(hSlot);
        return 1;
    }
    printf("Client connected\n");
    
    // Write message
    const char* msg = "Hello via Mailslot!";
    DWORD written;
    WriteFile(hClient, msg, strlen(msg)+1, &written, NULL);
    printf("Sent: %s\n", msg);
    CloseHandle(hClient);
    
    // Read message
    char buffer[256];
    DWORD bytesRead;
    if (ReadFile(hSlot, buffer, sizeof(buffer), &bytesRead, NULL)) {
        printf("Received: %s\n", buffer);
    }
    
    CloseHandle(hSlot);
    printf("Mailslot closed\n");
    
    return 0;
}''',
[{"type": "output_contains", "value": "Hello via Mailslot!"}])

add(2, "threading", "timer", "CreateWaitableTimer", "intermediate",
"Write a C++ program demonstrating waitable timers.",
r'''#include <windows.h>
#include <stdio.h>

int main() {
    printf("Waitable Timer Demo\n\n");
    
    HANDLE hTimer = CreateWaitableTimerA(NULL, TRUE, NULL);
    if (!hTimer) {
        printf("CreateWaitableTimer failed\n");
        return 1;
    }
    
    // Set timer for 100ms from now
    LARGE_INTEGER due;
    due.QuadPart = -1000000LL;  // 100ms in 100ns units, negative = relative
    
    printf("Setting timer for 100ms...\n");
    ULONGLONG start = GetTickCount64();
    
    if (!SetWaitableTimer(hTimer, &due, 0, NULL, NULL, FALSE)) {
        printf("SetWaitableTimer failed\n");
        CloseHandle(hTimer);
        return 1;
    }
    
    WaitForSingleObject(hTimer, INFINITE);
    
    ULONGLONG elapsed = GetTickCount64() - start;
    printf("Timer fired after %llu ms\n", elapsed);
    
    CloseHandle(hTimer);
    return 0;
}''',
[{"type": "output_contains", "value": "Timer fired after"}])

add(2, "memory", "virtual", "VirtualProtect", "intermediate",
"Write a C++ program that changes memory protection.",
r'''#include <windows.h>
#include <stdio.h>

int main() {
    printf("VirtualProtect Demo\n\n");
    
    // Allocate RW memory
    LPVOID pMem = VirtualAlloc(NULL, 4096, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
    printf("Allocated at: 0x%p (PAGE_READWRITE)\n", pMem);
    
    // Write data
    const char* msg = "Hello";
    memcpy(pMem, msg, strlen(msg)+1);
    printf("Wrote: %s\n", (char*)pMem);
    
    // Query protection
    MEMORY_BASIC_INFORMATION mbi;
    VirtualQuery(pMem, &mbi, sizeof(mbi));
    printf("Protection: 0x%lX\n", mbi.Protect);
    
    // Change to read-only
    DWORD oldProtect;
    VirtualProtect(pMem, 4096, PAGE_READONLY, &oldProtect);
    printf("Changed to PAGE_READONLY (old: 0x%lX)\n", oldProtect);
    
    // Read still works
    printf("Read: %s\n", (char*)pMem);
    
    VirtualFree(pMem, 0, MEM_RELEASE);
    printf("Memory freed\n");
    
    return 0;
}''',
[{"type": "output_contains", "value": "Changed to PAGE_READONLY"}])

add(2, "memory", "query", "VirtualQuery", "intermediate",
"Write a C++ program that walks the virtual address space.",
r'''#include <windows.h>
#include <stdio.h>

const char* StateStr(DWORD state) {
    if (state == MEM_COMMIT) return "COMMIT";
    if (state == MEM_RESERVE) return "RESERVE";
    if (state == MEM_FREE) return "FREE";
    return "?";
}

int main() {
    printf("Virtual Memory Walk\n\n");
    
    MEMORY_BASIC_INFORMATION mbi;
    LPVOID addr = NULL;
    int count = 0;
    SIZE_T totalCommit = 0;
    
    printf("%-18s %-10s %-10s %s\n", "Address", "Size", "State", "Type");
    printf("------------------ ---------- ---------- ----\n");
    
    while (VirtualQuery(addr, &mbi, sizeof(mbi)) && count < 15) {
        if (mbi.State != MEM_FREE) {
            printf("0x%p 0x%08zX %-10s %s\n",
                mbi.BaseAddress, mbi.RegionSize, StateStr(mbi.State),
                mbi.Type == MEM_IMAGE ? "IMAGE" :
                mbi.Type == MEM_MAPPED ? "MAPPED" :
                mbi.Type == MEM_PRIVATE ? "PRIVATE" : "?");
            
            if (mbi.State == MEM_COMMIT) totalCommit += mbi.RegionSize;
            count++;
        }
        addr = (BYTE*)mbi.BaseAddress + mbi.RegionSize;
    }
    
    printf("\nTotal committed: %.2f MB\n", totalCommit / (1024.0*1024));
    
    return 0;
}''',
[{"type": "output_contains", "value": "Total committed:"}])

add(2, "process", "thread", "Thread32First", "intermediate",
"Write a C++ program that enumerates threads in a process.",
r'''#include <windows.h>
#include <tlhelp32.h>
#include <stdio.h>

int main() {
    printf("Thread Enumeration Demo\n\n");
    
    DWORD pid = GetCurrentProcessId();
    printf("Current PID: %lu\n\n", pid);
    
    HANDLE hSnap = CreateToolhelp32Snapshot(TH32CS_SNAPTHREAD, 0);
    if (hSnap == INVALID_HANDLE_VALUE) {
        printf("Snapshot failed\n");
        return 1;
    }
    
    THREADENTRY32 te = {sizeof(te)};
    int count = 0;
    
    printf("%-10s %-10s %s\n", "TID", "Owner", "Priority");
    printf("---------- ---------- --------\n");
    
    if (Thread32First(hSnap, &te)) {
        do {
            if (te.th32OwnerProcessID == pid) {
                printf("%-10lu %-10lu %ld\n",
                    te.th32ThreadID, te.th32OwnerProcessID, te.tpBasePri);
                count++;
            }
        } while (Thread32Next(hSnap, &te));
    }
    
    CloseHandle(hSnap);
    printf("\nTotal threads: %d\n", count);
    
    return 0;
}''',
[{"type": "output_contains", "value": "Total threads:"}])

# =============================================================================
# ADDITIONAL TIER 3 PROBLEMS
# =============================================================================

add(3, "pe", "imports", "IMAGE_IMPORT_DESCRIPTOR", "intermediate",
"Write a C++ program that parses the PE import table.",
r'''#include <windows.h>
#include <stdio.h>

int main() {
    printf("PE Import Parser\n\n");
    
    BYTE* base = (BYTE*)GetModuleHandleA(NULL);
    PIMAGE_DOS_HEADER dos = (PIMAGE_DOS_HEADER)base;
    PIMAGE_NT_HEADERS nt = (PIMAGE_NT_HEADERS)(base + dos->e_lfanew);
    
    DWORD impRVA = nt->OptionalHeader.DataDirectory[IMAGE_DIRECTORY_ENTRY_IMPORT].VirtualAddress;
    if (!impRVA) {
        printf("No imports\n");
        return 0;
    }
    
    PIMAGE_IMPORT_DESCRIPTOR imp = (PIMAGE_IMPORT_DESCRIPTOR)(base + impRVA);
    
    while (imp->Name) {
        printf("[%s]\n", (char*)(base + imp->Name));
        
        PIMAGE_THUNK_DATA thunk = (PIMAGE_THUNK_DATA)(base + 
            (imp->OriginalFirstThunk ? imp->OriginalFirstThunk : imp->FirstThunk));
        
        int count = 0;
        while (thunk->u1.AddressOfData && count < 5) {
            if (!IMAGE_SNAP_BY_ORDINAL(thunk->u1.Ordinal)) {
                PIMAGE_IMPORT_BY_NAME name = 
                    (PIMAGE_IMPORT_BY_NAME)(base + thunk->u1.AddressOfData);
                printf("  %s\n", name->Name);
            }
            thunk++;
            count++;
        }
        printf("\n");
        imp++;
    }
    
    return 0;
}''',
[{"type": "output_contains", "value": "KERNEL32.dll"}])

add(3, "pe", "exports", "IMAGE_EXPORT_DIRECTORY", "intermediate",
"Write a C++ program that parses kernel32.dll exports.",
r'''#include <windows.h>
#include <stdio.h>

int main() {
    printf("PE Export Parser\n\n");
    
    BYTE* base = (BYTE*)GetModuleHandleA("kernel32.dll");
    PIMAGE_DOS_HEADER dos = (PIMAGE_DOS_HEADER)base;
    PIMAGE_NT_HEADERS nt = (PIMAGE_NT_HEADERS)(base + dos->e_lfanew);
    
    DWORD expRVA = nt->OptionalHeader.DataDirectory[IMAGE_DIRECTORY_ENTRY_EXPORT].VirtualAddress;
    PIMAGE_EXPORT_DIRECTORY exp = (PIMAGE_EXPORT_DIRECTORY)(base + expRVA);
    
    printf("Module: %s\n", (char*)(base + exp->Name));
    printf("Functions: %lu\n", exp->NumberOfFunctions);
    printf("Names: %lu\n\n", exp->NumberOfNames);
    
    DWORD* names = (DWORD*)(base + exp->AddressOfNames);
    WORD* ords = (WORD*)(base + exp->AddressOfNameOrdinals);
    DWORD* funcs = (DWORD*)(base + exp->AddressOfFunctions);
    
    printf("%-6s %-10s %s\n", "Ord", "RVA", "Name");
    printf("------ ---------- ----\n");
    
    for (DWORD i = 0; i < exp->NumberOfNames && i < 10; i++) {
        printf("%-6u 0x%08lX %s\n",
            ords[i] + exp->Base, funcs[ords[i]], (char*)(base + names[i]));
    }
    
    return 0;
}''',
[{"type": "output_contains", "value": "KERNEL32.dll"},
 {"type": "output_contains", "value": "Functions:"}])

add(3, "security", "acl", "GetSecurityInfo", "intermediate",
"Write a C++ program that queries file security information.",
r'''#include <windows.h>
#include <aclapi.h>
#include <sddl.h>
#include <stdio.h>
#pragma comment(lib, "advapi32.lib")

int main() {
    printf("Security Info Demo\n\n");
    
    const char* filename = "C:\\Windows\\System32\\ntdll.dll";
    
    PSECURITY_DESCRIPTOR pSD = NULL;
    PSID pOwner = NULL;
    
    DWORD result = GetNamedSecurityInfoA(filename, SE_FILE_OBJECT,
        OWNER_SECURITY_INFORMATION | DACL_SECURITY_INFORMATION,
        &pOwner, NULL, NULL, NULL, &pSD);
    
    if (result != ERROR_SUCCESS) {
        printf("GetNamedSecurityInfo failed: %lu\n", result);
        return 1;
    }
    
    // Owner SID
    LPSTR sidStr;
    ConvertSidToStringSidA(pOwner, &sidStr);
    printf("File: %s\n", filename);
    printf("Owner SID: %s\n", sidStr);
    
    // Lookup owner name
    char name[256], domain[256];
    DWORD nameLen = 256, domLen = 256;
    SID_NAME_USE use;
    LookupAccountSidA(NULL, pOwner, name, &nameLen, domain, &domLen, &use);
    printf("Owner: %s\\%s\n", domain, name);
    
    LocalFree(sidStr);
    LocalFree(pSD);
    
    return 0;
}''',
[{"type": "output_contains", "value": "Owner SID:"},
 {"type": "output_contains", "value": "Owner:"}])

add(3, "dll", "load", "LoadLibrary", "intermediate",
"Write a C++ program that dynamically loads and uses a DLL.",
r'''#include <windows.h>
#include <stdio.h>

int main() {
    printf("Dynamic DLL Loading Demo\n\n");
    
    HMODULE hMod = LoadLibraryA("user32.dll");
    if (!hMod) {
        printf("LoadLibrary failed\n");
        return 1;
    }
    printf("Loaded user32.dll @ 0x%p\n\n", hMod);
    
    // Get function pointers
    typedef BOOL (WINAPI *pGetCursorPos)(LPPOINT);
    pGetCursorPos fnGetCursorPos = (pGetCursorPos)GetProcAddress(hMod, "GetCursorPos");
    
    typedef int (WINAPI *pGetSystemMetrics)(int);
    pGetSystemMetrics fnGetMetrics = (pGetSystemMetrics)GetProcAddress(hMod, "GetSystemMetrics");
    
    printf("GetCursorPos @ 0x%p\n", fnGetCursorPos);
    printf("GetSystemMetrics @ 0x%p\n\n", fnGetMetrics);
    
    if (fnGetCursorPos) {
        POINT pt;
        fnGetCursorPos(&pt);
        printf("Cursor: (%ld, %ld)\n", pt.x, pt.y);
    }
    
    if (fnGetMetrics) {
        printf("Screen: %d x %d\n", fnGetMetrics(0), fnGetMetrics(1));
    }
    
    FreeLibrary(hMod);
    printf("\nDLL unloaded\n");
    
    return 0;
}''',
[{"type": "output_contains", "value": "Loaded user32.dll"},
 {"type": "output_contains", "value": "DLL unloaded"}])

add(3, "dll", "manual", "ManualGetProcAddress", "intermediate",
"Write a C++ program that manually resolves exports by parsing PE.",
r'''#include <windows.h>
#include <stdio.h>

FARPROC ManualGetProcAddress(HMODULE hMod, const char* name) {
    BYTE* base = (BYTE*)hMod;
    PIMAGE_DOS_HEADER dos = (PIMAGE_DOS_HEADER)base;
    PIMAGE_NT_HEADERS nt = (PIMAGE_NT_HEADERS)(base + dos->e_lfanew);
    
    DWORD expRVA = nt->OptionalHeader.DataDirectory[0].VirtualAddress;
    if (!expRVA) return NULL;
    
    PIMAGE_EXPORT_DIRECTORY exp = (PIMAGE_EXPORT_DIRECTORY)(base + expRVA);
    DWORD* names = (DWORD*)(base + exp->AddressOfNames);
    WORD* ords = (WORD*)(base + exp->AddressOfNameOrdinals);
    DWORD* funcs = (DWORD*)(base + exp->AddressOfFunctions);
    
    for (DWORD i = 0; i < exp->NumberOfNames; i++) {
        if (strcmp((char*)(base + names[i]), name) == 0) {
            return (FARPROC)(base + funcs[ords[i]]);
        }
    }
    return NULL;
}

int main() {
    printf("Manual GetProcAddress\n\n");
    
    HMODULE ntdll = GetModuleHandleA("ntdll.dll");
    const char* funcs[] = {"NtClose", "RtlGetVersion", "NtQueryInformationProcess"};
    
    printf("%-30s %-18s %-18s\n", "Function", "Manual", "API");
    printf("------------------------------ ------------------ ------------------\n");
    
    for (int i = 0; i < 3; i++) {
        FARPROC manual = ManualGetProcAddress(ntdll, funcs[i]);
        FARPROC api = GetProcAddress(ntdll, funcs[i]);
        printf("%-30s 0x%p 0x%p %s\n", funcs[i], manual, api,
            manual == api ? "[OK]" : "[FAIL]");
    }
    
    return 0;
}''',
[{"type": "output_contains", "value": "NtClose"},
 {"type": "output_contains", "value": "[OK]"}])

add(3, "threading", "suspend", "SuspendThread", "intermediate",
"Write a C++ program demonstrating thread suspension.",
r'''#include <windows.h>
#include <stdio.h>

volatile BOOL running = TRUE;

DWORD WINAPI Counter(LPVOID arg) {
    int count = 0;
    while (running) {
        count++;
        Sleep(100);
    }
    printf("[Thread] Exiting with count: %d\n", count);
    return count;
}

int main() {
    printf("Thread Suspend Demo\n\n");
    
    HANDLE hThread = CreateThread(NULL, 0, Counter, NULL, 0, NULL);
    printf("Thread started\n");
    
    Sleep(300);
    
    DWORD suspendCount = SuspendThread(hThread);
    printf("Thread suspended (count: %lu)\n", suspendCount);
    
    Sleep(500);
    
    suspendCount = ResumeThread(hThread);
    printf("Thread resumed (count: %lu)\n", suspendCount);
    
    Sleep(300);
    running = FALSE;
    
    WaitForSingleObject(hThread, 1000);
    
    DWORD exitCode;
    GetExitCodeThread(hThread, &exitCode);
    printf("Exit code: %lu\n", exitCode);
    
    CloseHandle(hThread);
    return 0;
}''',
[{"type": "output_contains", "value": "Thread suspended"},
 {"type": "output_contains", "value": "Thread resumed"}])

add(3, "threading", "context", "GetThreadContext", "intermediate",
"Write a C++ program that reads thread context.",
r'''#include <windows.h>
#include <stdio.h>

DWORD WINAPI Worker(LPVOID arg) {
    volatile int x = 0;
    while (*(BOOL*)arg) {
        x++;
    }
    return x;
}

int main() {
    printf("Thread Context Demo\n\n");
    
    BOOL running = TRUE;
    HANDLE hThread = CreateThread(NULL, 0, Worker, &running, 0, NULL);
    
    Sleep(50);
    SuspendThread(hThread);
    
    CONTEXT ctx = {0};
    ctx.ContextFlags = CONTEXT_FULL;
    
    if (GetThreadContext(hThread, &ctx)) {
        printf("Thread Context (x64):\n");
        printf("  RIP: 0x%016llX\n", ctx.Rip);
        printf("  RSP: 0x%016llX\n", ctx.Rsp);
        printf("  RBP: 0x%016llX\n", ctx.Rbp);
        printf("  RAX: 0x%016llX\n", ctx.Rax);
        printf("  RBX: 0x%016llX\n", ctx.Rbx);
        printf("  RCX: 0x%016llX\n", ctx.Rcx);
        printf("  RDX: 0x%016llX\n", ctx.Rdx);
        printf("  EFlags: 0x%08lX\n", ctx.EFlags);
    }
    
    running = FALSE;
    ResumeThread(hThread);
    WaitForSingleObject(hThread, 1000);
    CloseHandle(hThread);
    
    return 0;
}''',
[{"type": "output_contains", "value": "RIP:"},
 {"type": "output_contains", "value": "RSP:"}])

# =============================================================================
# ADDITIONAL TIER 4 PROBLEMS
# =============================================================================

add(4, "native", "pbi", "NtQueryInformationProcess", "advanced",
"Write a C++ program that queries process basic information via native API.",
r'''#include <windows.h>
#include <stdio.h>

typedef LONG NTSTATUS;
typedef struct _PROCESS_BASIC_INFORMATION {
    PVOID Reserved1;
    PVOID PebBaseAddress;
    PVOID Reserved2[2];
    ULONG_PTR UniqueProcessId;
    ULONG_PTR InheritedFromUniqueProcessId;
} PROCESS_BASIC_INFORMATION;

typedef NTSTATUS (NTAPI *pNtQueryInformationProcess)(HANDLE, ULONG, PVOID, ULONG, PULONG);

int main() {
    printf("NtQueryInformationProcess Demo\n\n");
    
    HMODULE ntdll = GetModuleHandleA("ntdll.dll");
    pNtQueryInformationProcess NtQIP = (pNtQueryInformationProcess)
        GetProcAddress(ntdll, "NtQueryInformationProcess");
    
    PROCESS_BASIC_INFORMATION pbi;
    ULONG len;
    
    NTSTATUS status = NtQIP(GetCurrentProcess(), 0, &pbi, sizeof(pbi), &len);
    
    if (status == 0) {
        printf("PEB Address: 0x%p\n", pbi.PebBaseAddress);
        printf("Process ID: %llu\n", (ULONGLONG)pbi.UniqueProcessId);
        printf("Parent PID: %llu\n", (ULONGLONG)pbi.InheritedFromUniqueProcessId);
        
        printf("\nVerification:\n");
        printf("GetCurrentProcessId: %lu\n", GetCurrentProcessId());
    }
    
    return 0;
}''',
[{"type": "output_contains", "value": "PEB Address:"},
 {"type": "output_contains", "value": "Parent PID:"}])

add(4, "internals", "ldr", "LDR_DATA_TABLE_ENTRY", "advanced",
"Write a C++ program that manually implements GetModuleHandle via PEB.",
r'''#include <windows.h>
#include <stdio.h>
#include <intrin.h>

typedef struct _UNICODE_STRING { USHORT Length, MaxLength; PWSTR Buffer; } UNICODE_STRING;

typedef struct _LDR_DATA_TABLE_ENTRY {
    LIST_ENTRY InLoadOrderLinks;
    LIST_ENTRY InMemoryOrderLinks;
    LIST_ENTRY InInitOrderLinks;
    PVOID DllBase;
    PVOID EntryPoint;
    ULONG SizeOfImage;
    UNICODE_STRING FullDllName;
    UNICODE_STRING BaseDllName;
} LDR_DATA_TABLE_ENTRY;

typedef struct _PEB_LDR_DATA {
    ULONG Length;
    BOOLEAN Initialized;
    HANDLE SsHandle;
    LIST_ENTRY InLoadOrderModuleList;
} PEB_LDR_DATA;

typedef struct _PEB {
    BYTE Reserved[2];
    BYTE BeingDebugged;
    BYTE Reserved2;
    PVOID Reserved3[2];
    PEB_LDR_DATA* Ldr;
} PEB;

HMODULE ManualGetModuleHandle(const wchar_t* name) {
    PEB* peb = (PEB*)__readgsqword(0x60);
    LIST_ENTRY* head = &peb->Ldr->InLoadOrderModuleList;
    LIST_ENTRY* curr = head->Flink;
    
    while (curr != head) {
        LDR_DATA_TABLE_ENTRY* entry = CONTAINING_RECORD(curr, LDR_DATA_TABLE_ENTRY, InLoadOrderLinks);
        if (entry->BaseDllName.Buffer && _wcsicmp(entry->BaseDllName.Buffer, name) == 0) {
            return (HMODULE)entry->DllBase;
        }
        curr = curr->Flink;
    }
    return NULL;
}

int main() {
    printf("Manual GetModuleHandle\n\n");
    
    const wchar_t* modules[] = {L"ntdll.dll", L"kernel32.dll", L"kernelbase.dll"};
    
    for (int i = 0; i < 3; i++) {
        HMODULE manual = ManualGetModuleHandle(modules[i]);
        HMODULE api = GetModuleHandleW(modules[i]);
        printf("%-15ws Manual=0x%p API=0x%p %s\n", modules[i], manual, api,
            manual == api ? "[OK]" : "[FAIL]");
    }
    
    return 0;
}''',
[{"type": "output_contains", "value": "ntdll.dll"},
 {"type": "output_contains", "value": "[OK]"}])

add(4, "evasion", "sandbox", "SandboxChecks", "advanced",
"Write a C++ program that performs sandbox detection checks.",
r'''#include <windows.h>
#include <stdio.h>
#include <intrin.h>
#pragma comment(lib, "advapi32.lib")

int main() {
    printf("Sandbox Detection Demo\n\n");
    
    // Computer name
    char compName[256];
    DWORD size = sizeof(compName);
    GetComputerNameA(compName, &size);
    printf("[1] Computer: %s\n", compName);
    
    // Username
    char userName[256];
    size = sizeof(userName);
    GetUserNameA(userName, &size);
    printf("[2] User: %s\n", userName);
    
    // RAM
    MEMORYSTATUSEX mem = {sizeof(mem)};
    GlobalMemoryStatusEx(&mem);
    printf("[3] RAM: %.2f GB %s\n", mem.ullTotalPhys / (1024.0*1024*1024),
        mem.ullTotalPhys < 2ULL*1024*1024*1024 ? "(Low)" : "");
    
    // CPUs
    SYSTEM_INFO si;
    GetSystemInfo(&si);
    printf("[4] CPUs: %lu %s\n", si.dwNumberOfProcessors,
        si.dwNumberOfProcessors < 2 ? "(Low)" : "");
    
    // Disk size
    ULARGE_INTEGER total;
    GetDiskFreeSpaceExA("C:\\", NULL, &total, NULL);
    printf("[5] Disk: %.2f GB %s\n", total.QuadPart / (1024.0*1024*1024),
        total.QuadPart < 60ULL*1024*1024*1024 ? "(Small)" : "");
    
    // Hypervisor bit
    int cpuInfo[4];
    __cpuid(cpuInfo, 1);
    BOOL hypervisor = (cpuInfo[2] >> 31) & 1;
    printf("[6] Hypervisor: %s\n", hypervisor ? "Yes (VM likely)" : "No");
    
    return 0;
}''',
[{"type": "output_contains", "value": "Computer:"},
 {"type": "output_contains", "value": "Hypervisor:"}])

add(4, "threading", "remote", "CreateRemoteThread", "advanced",
"Write a C++ program demonstrating CreateRemoteThread in current process.",
r'''#include <windows.h>
#include <stdio.h>

DWORD WINAPI RemoteFunc(LPVOID arg) {
    DWORD* pResult = (DWORD*)arg;
    *pResult = GetCurrentThreadId();
    return 0xDEADBEEF;
}

int main() {
    printf("CreateRemoteThread Demo\n\n");
    
    HANDLE hProc = GetCurrentProcess();
    
    LPVOID pRemote = VirtualAllocEx(hProc, NULL, sizeof(DWORD),
        MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
    printf("Remote memory: 0x%p\n", pRemote);
    
    HANDLE hThread = CreateRemoteThread(hProc, NULL, 0,
        (LPTHREAD_START_ROUTINE)RemoteFunc, pRemote, 0, NULL);
    
    if (!hThread) {
        printf("CreateRemoteThread failed: %lu\n", GetLastError());
        return 1;
    }
    printf("Remote thread: 0x%p\n", hThread);
    
    WaitForSingleObject(hThread, INFINITE);
    
    DWORD exitCode;
    GetExitCodeThread(hThread, &exitCode);
    printf("Exit code: 0x%lX\n", exitCode);
    
    DWORD tid;
    ReadProcessMemory(hProc, pRemote, &tid, sizeof(tid), NULL);
    printf("TID written: %lu\n", tid);
    
    CloseHandle(hThread);
    VirtualFreeEx(hProc, pRemote, 0, MEM_RELEASE);
    
    return 0;
}''',
[{"type": "output_contains", "value": "Remote thread:"},
 {"type": "output_contains", "value": "Exit code: 0xDEADBEEF"}])

add(4, "native", "direct", "DirectSyscall", "advanced",
"Write a C++ program demonstrating syscall stub structure.",
r'''#include <windows.h>
#include <stdio.h>

DWORD GetSyscallNumber(const char* name) {
    BYTE* func = (BYTE*)GetProcAddress(GetModuleHandleA("ntdll.dll"), name);
    if (func && func[0] == 0x4C && func[1] == 0x8B && func[2] == 0xD1 && func[3] == 0xB8) {
        return *(DWORD*)(func + 4);
    }
    if (func && (func[0] == 0xE9 || func[0] == 0xFF)) return 0xFFFFFFFF;
    return 0;
}

int main() {
    printf("Direct Syscall Structure Demo\n\n");
    
    DWORD sysNtClose = GetSyscallNumber("NtClose");
    printf("NtClose syscall: 0x%04lX\n\n", sysNtClose);
    
    printf("Syscall stub structure (x64):\n");
    printf("  4C 8B D1          mov r10, rcx\n");
    printf("  B8 %02X %02X 00 00    mov eax, 0x%04lX\n",
        sysNtClose & 0xFF, (sysNtClose >> 8) & 0xFF, sysNtClose);
    printf("  0F 05             syscall\n");
    printf("  C3                ret\n\n");
    
    // Test via normal ntdll call
    HANDLE hEvent = CreateEvent(NULL, FALSE, FALSE, NULL);
    printf("Created handle: 0x%p\n", hEvent);
    
    typedef LONG (NTAPI *pNtClose)(HANDLE);
    pNtClose NtClose = (pNtClose)GetProcAddress(GetModuleHandleA("ntdll.dll"), "NtClose");
    LONG status = NtClose(hEvent);
    printf("NtClose returned: 0x%lX\n", status);
    
    return 0;
}''',
[{"type": "output_contains", "value": "mov r10, rcx"},
 {"type": "output_contains", "value": "NtClose returned: 0x0"}])

add(4, "exception", "seh", "SEH_demo", "advanced",
"Write a C++ program demonstrating structured exception handling.",
r'''#include <windows.h>
#include <stdio.h>

int FilterFunc(EXCEPTION_POINTERS* ep, const char* location) {
    printf("[Filter] Exception 0x%08lX at %s\n",
        ep->ExceptionRecord->ExceptionCode, location);
    return EXCEPTION_EXECUTE_HANDLER;
}

int main() {
    printf("Structured Exception Handling Demo\n\n");
    
    // Divide by zero
    __try {
        printf("Test 1: Divide by zero\n");
        volatile int x = 0;
        volatile int y = 1 / x;
        (void)y;
    } __except(FilterFunc(GetExceptionInformation(), "div0")) {
        printf("Caught divide by zero\n\n");
    }
    
    // Access violation
    __try {
        printf("Test 2: Access violation\n");
        volatile int* p = NULL;
        *p = 42;
    } __except(FilterFunc(GetExceptionInformation(), "AV")) {
        printf("Caught access violation\n\n");
    }
    
    // Custom exception
    __try {
        printf("Test 3: RaiseException\n");
        RaiseException(0xE0000001, 0, 0, NULL);
    } __except(FilterFunc(GetExceptionInformation(), "custom")) {
        printf("Caught custom exception\n\n");
    }
    
    printf("All exceptions handled successfully\n");
    return 0;
}''',
[{"type": "output_contains", "value": "Caught divide by zero"},
 {"type": "output_contains", "value": "All exceptions handled successfully"}])

# =============================================================================
# MORE TIER 1 PROBLEMS (to reach 40+)
# =============================================================================

add(1, "process", "handle", "DuplicateHandle", "beginner",
"Write a C++ program that duplicates a handle.",
r'''#include <windows.h>
#include <stdio.h>

int main() {
    printf("Handle Duplication Demo\n\n");
    
    HANDLE hEvent = CreateEventA(NULL, FALSE, FALSE, NULL);
    printf("Original handle: 0x%p\n", hEvent);
    
    HANDLE hDup;
    if (DuplicateHandle(GetCurrentProcess(), hEvent,
            GetCurrentProcess(), &hDup, 0, FALSE, DUPLICATE_SAME_ACCESS)) {
        printf("Duplicate handle: 0x%p\n", hDup);
        
        // Both handles work
        SetEvent(hEvent);
        DWORD result = WaitForSingleObject(hDup, 0);
        printf("Wait result: %s\n", result == WAIT_OBJECT_0 ? "SIGNALED" : "TIMEOUT");
        
        CloseHandle(hDup);
    }
    
    CloseHandle(hEvent);
    printf("Handles closed\n");
    return 0;
}''',
[{"type": "output_contains", "value": "SIGNALED"}])

add(1, "console", "input", "GetStdHandle", "beginner",
"Write a C++ program that gets console input mode.",
r'''#include <windows.h>
#include <stdio.h>

int main() {
    printf("Console Handle Demo\n\n");
    
    HANDLE hIn = GetStdHandle(STD_INPUT_HANDLE);
    HANDLE hOut = GetStdHandle(STD_OUTPUT_HANDLE);
    HANDLE hErr = GetStdHandle(STD_ERROR_HANDLE);
    
    printf("STD_INPUT_HANDLE:  0x%p\n", hIn);
    printf("STD_OUTPUT_HANDLE: 0x%p\n", hOut);
    printf("STD_ERROR_HANDLE:  0x%p\n", hErr);
    
    DWORD mode;
    if (GetConsoleMode(hIn, &mode)) {
        printf("\nInput mode: 0x%lX\n", mode);
        if (mode & ENABLE_LINE_INPUT) printf("  ENABLE_LINE_INPUT\n");
        if (mode & ENABLE_ECHO_INPUT) printf("  ENABLE_ECHO_INPUT\n");
        if (mode & ENABLE_PROCESSED_INPUT) printf("  ENABLE_PROCESSED_INPUT\n");
    }
    
    if (GetConsoleMode(hOut, &mode)) {
        printf("\nOutput mode: 0x%lX\n", mode);
        if (mode & ENABLE_PROCESSED_OUTPUT) printf("  ENABLE_PROCESSED_OUTPUT\n");
        if (mode & ENABLE_WRAP_AT_EOL_OUTPUT) printf("  ENABLE_WRAP_AT_EOL_OUTPUT\n");
    }
    
    return 0;
}''',
[{"type": "output_contains", "value": "STD_INPUT_HANDLE:"},
 {"type": "output_contains", "value": "Input mode:"}])

add(1, "sysinfo", "perf", "QueryPerformanceCounter", "beginner",
"Write a C++ program using high-resolution performance counters.",
r'''#include <windows.h>
#include <stdio.h>

int main() {
    printf("Performance Counter Demo\n\n");
    
    LARGE_INTEGER freq, start, end;
    
    QueryPerformanceFrequency(&freq);
    printf("Frequency: %lld Hz\n", freq.QuadPart);
    printf("Resolution: %.3f ns\n\n", 1e9 / freq.QuadPart);
    
    // Measure Sleep precision
    QueryPerformanceCounter(&start);
    Sleep(10);
    QueryPerformanceCounter(&end);
    
    double elapsed_ms = (double)(end.QuadPart - start.QuadPart) / freq.QuadPart * 1000;
    printf("Sleep(10) actual: %.3f ms\n", elapsed_ms);
    
    // Measure tight loop
    QueryPerformanceCounter(&start);
    volatile int sum = 0;
    for (int i = 0; i < 1000000; i++) sum += i;
    QueryPerformanceCounter(&end);
    
    double loop_us = (double)(end.QuadPart - start.QuadPart) / freq.QuadPart * 1e6;
    printf("1M iterations: %.2f us\n", loop_us);
    
    return 0;
}''',
[{"type": "output_contains", "value": "Frequency:"},
 {"type": "output_contains", "value": "Sleep(10) actual:"}])

add(1, "file", "copy", "CopyFile", "beginner",
"Write a C++ program that copies and moves files.",
r'''#include <windows.h>
#include <stdio.h>

int main() {
    printf("File Copy/Move Demo\n\n");
    
    // Create source file
    HANDLE h = CreateFileA("source.txt", GENERIC_WRITE, 0, NULL,
        CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, NULL);
    const char* data = "Test file content";
    DWORD written;
    WriteFile(h, data, strlen(data), &written, NULL);
    CloseHandle(h);
    printf("Created source.txt\n");
    
    // Copy file
    if (CopyFileA("source.txt", "copy.txt", FALSE)) {
        printf("Copied to copy.txt\n");
    }
    
    // Move file
    if (MoveFileA("copy.txt", "moved.txt")) {
        printf("Moved to moved.txt\n");
    }
    
    // Check files exist
    printf("\nFile status:\n");
    printf("  source.txt: %s\n", GetFileAttributesA("source.txt") != INVALID_FILE_ATTRIBUTES ? "EXISTS" : "MISSING");
    printf("  copy.txt:   %s\n", GetFileAttributesA("copy.txt") != INVALID_FILE_ATTRIBUTES ? "EXISTS" : "MISSING");
    printf("  moved.txt:  %s\n", GetFileAttributesA("moved.txt") != INVALID_FILE_ATTRIBUTES ? "EXISTS" : "MISSING");
    
    // Cleanup
    DeleteFileA("source.txt");
    DeleteFileA("moved.txt");
    printf("\nFiles cleaned up\n");
    
    return 0;
}''',
[{"type": "output_contains", "value": "Copied to copy.txt"},
 {"type": "output_contains", "value": "Moved to moved.txt"}])

add(1, "memory", "global", "GlobalAlloc", "beginner",
"Write a C++ program demonstrating GlobalAlloc/GlobalFree.",
r'''#include <windows.h>
#include <stdio.h>

int main() {
    printf("Global Memory Demo\n\n");
    
    // Allocate fixed memory
    HGLOBAL hMem = GlobalAlloc(GMEM_FIXED | GMEM_ZEROINIT, 1024);
    printf("GlobalAlloc (fixed): 0x%p\n", hMem);
    
    // Write data (fixed = pointer is usable directly)
    strcpy((char*)hMem, "Hello from GlobalAlloc");
    printf("Data: %s\n", (char*)hMem);
    
    // Get size
    SIZE_T size = GlobalSize(hMem);
    printf("Size: %zu bytes\n", size);
    
    GlobalFree(hMem);
    printf("Memory freed\n\n");
    
    // Allocate moveable memory
    hMem = GlobalAlloc(GMEM_MOVEABLE | GMEM_ZEROINIT, 2048);
    printf("GlobalAlloc (moveable): 0x%p\n", hMem);
    
    LPVOID ptr = GlobalLock(hMem);
    printf("Locked at: 0x%p\n", ptr);
    strcpy((char*)ptr, "Moveable memory");
    printf("Data: %s\n", (char*)ptr);
    GlobalUnlock(hMem);
    
    GlobalFree(hMem);
    printf("Memory freed\n");
    
    return 0;
}''',
[{"type": "output_contains", "value": "Hello from GlobalAlloc"},
 {"type": "output_contains", "value": "Moveable memory"}])

add(1, "memory", "local", "LocalAlloc", "beginner",
"Write a C++ program demonstrating LocalAlloc/LocalFree.",
r'''#include <windows.h>
#include <stdio.h>

int main() {
    printf("Local Memory Demo\n\n");
    
    HLOCAL hMem = LocalAlloc(LPTR, 512);  // LPTR = LMEM_FIXED | LMEM_ZEROINIT
    printf("LocalAlloc: 0x%p\n", hMem);
    
    // Use directly with LPTR
    strcpy((char*)hMem, "Test data in local memory");
    printf("Data: %s\n", (char*)hMem);
    
    // Reallocate
    hMem = LocalReAlloc(hMem, 1024, LMEM_MOVEABLE);
    printf("Reallocated: 0x%p\n", hMem);
    
    SIZE_T size = LocalSize(hMem);
    printf("New size: %zu bytes\n", size);
    
    LocalFree(hMem);
    printf("Memory freed\n");
    
    return 0;
}''',
[{"type": "output_contains", "value": "Test data in local memory"},
 {"type": "output_contains", "value": "Memory freed"}])

add(1, "file", "seek", "SetFilePointer", "beginner",
"Write a C++ program demonstrating file seeking.",
r'''#include <windows.h>
#include <stdio.h>

int main() {
    printf("File Seek Demo\n\n");
    
    HANDLE h = CreateFileA("seek_test.txt", GENERIC_READ | GENERIC_WRITE, 0, NULL,
        CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, NULL);
    
    // Write data
    const char* data = "0123456789ABCDEF";
    DWORD written;
    WriteFile(h, data, strlen(data), &written, NULL);
    printf("Wrote: %s\n", data);
    
    // Seek to beginning
    SetFilePointer(h, 0, NULL, FILE_BEGIN);
    printf("\nAfter FILE_BEGIN:\n");
    
    char buf[8] = {0};
    DWORD bytesRead;
    ReadFile(h, buf, 4, &bytesRead, NULL);
    printf("  Read 4: %s\n", buf);
    
    // Seek from current
    SetFilePointer(h, 4, NULL, FILE_CURRENT);
    ReadFile(h, buf, 4, &bytesRead, NULL);
    printf("  Skip 4, read 4: %s\n", buf);
    
    // Seek from end
    SetFilePointer(h, -4, NULL, FILE_END);
    ReadFile(h, buf, 4, &bytesRead, NULL);
    printf("  Last 4: %s\n", buf);
    
    CloseHandle(h);
    DeleteFileA("seek_test.txt");
    printf("\nFile cleaned up\n");
    
    return 0;
}''',
[{"type": "output_contains", "value": "Read 4: 0123"},
 {"type": "output_contains", "value": "Last 4: CDEF"}])

add(1, "process", "cmdline", "GetCommandLine", "beginner",
"Write a C++ program that parses command line arguments.",
r'''#include <windows.h>
#include <stdio.h>
#include <shellapi.h>
#pragma comment(lib, "shell32.lib")

int main(int argc, char* argv[]) {
    printf("Command Line Demo\n\n");
    
    // Raw command line
    printf("Raw: %s\n\n", GetCommandLineA());
    
    // C-style argc/argv
    printf("C-style (%d args):\n", argc);
    for (int i = 0; i < argc && i < 5; i++) {
        printf("  [%d] %s\n", i, argv[i]);
    }
    
    // Windows style (wide)
    int wargc;
    LPWSTR* wargv = CommandLineToArgvW(GetCommandLineW(), &wargc);
    
    printf("\nWindows style (%d args):\n", wargc);
    for (int i = 0; i < wargc && i < 5; i++) {
        wprintf(L"  [%d] %s\n", i, wargv[i]);
    }
    
    LocalFree(wargv);
    return 0;
}''',
[{"type": "output_contains", "value": "Raw:"},
 {"type": "output_contains", "value": "C-style"}])

add(1, "sysinfo", "locale", "GetLocaleInfo", "beginner",
"Write a C++ program that displays locale information.",
r'''#include <windows.h>
#include <stdio.h>

int main() {
    printf("Locale Information Demo\n\n");
    
    LCID lcid = GetUserDefaultLCID();
    printf("User LCID: 0x%08lX\n", lcid);
    
    char buf[256];
    
    GetLocaleInfoA(lcid, LOCALE_SENGLANGUAGE, buf, sizeof(buf));
    printf("Language: %s\n", buf);
    
    GetLocaleInfoA(lcid, LOCALE_SENGCOUNTRY, buf, sizeof(buf));
    printf("Country: %s\n", buf);
    
    GetLocaleInfoA(lcid, LOCALE_SISO639LANGNAME, buf, sizeof(buf));
    printf("ISO Language: %s\n", buf);
    
    GetLocaleInfoA(lcid, LOCALE_SISO3166CTRYNAME, buf, sizeof(buf));
    printf("ISO Country: %s\n", buf);
    
    GetLocaleInfoA(lcid, LOCALE_SCURRENCY, buf, sizeof(buf));
    printf("Currency: %s\n", buf);
    
    GetLocaleInfoA(lcid, LOCALE_SSHORTDATE, buf, sizeof(buf));
    printf("Short date: %s\n", buf);
    
    GetLocaleInfoA(lcid, LOCALE_STIMEFORMAT, buf, sizeof(buf));
    printf("Time format: %s\n", buf);
    
    return 0;
}''',
[{"type": "output_contains", "value": "Language:"},
 {"type": "output_contains", "value": "Currency:"}])

add(1, "error", "set", "SetLastError", "beginner",
"Write a C++ program demonstrating SetLastError/GetLastError.",
r'''#include <windows.h>
#include <stdio.h>

int main() {
    printf("Error Code Demo\n\n");
    
    // Clear error
    SetLastError(ERROR_SUCCESS);
    printf("After clear: %lu\n", GetLastError());
    
    // Set custom error
    SetLastError(ERROR_FILE_NOT_FOUND);
    printf("After set ERROR_FILE_NOT_FOUND: %lu\n", GetLastError());
    
    // APIs that fail set error
    HANDLE h = CreateFileA("nonexistent_12345.xyz", GENERIC_READ, 0, NULL,
        OPEN_EXISTING, 0, NULL);
    printf("\nAfter failed CreateFile: %lu (FILE_NOT_FOUND=%lu)\n",
        GetLastError(), ERROR_FILE_NOT_FOUND);
    
    // Success clears error on some APIs
    h = CreateFileA("test_err.txt", GENERIC_WRITE, 0, NULL,
        CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, NULL);
    printf("After success CreateFile: %lu\n", GetLastError());
    CloseHandle(h);
    DeleteFileA("test_err.txt");
    
    return 0;
}''',
[{"type": "output_contains", "value": "After set ERROR_FILE_NOT_FOUND: 2"},
 {"type": "output_contains", "value": "After failed CreateFile:"}])

# =============================================================================
# MORE TIER 2 PROBLEMS (to reach 50+)
# =============================================================================

add(2, "file", "async", "ReadFileEx", "intermediate",
"Write a C++ program demonstrating overlapped file I/O.",
r'''#include <windows.h>
#include <stdio.h>

volatile BOOL completed = FALSE;

VOID CALLBACK ReadComplete(DWORD dwErr, DWORD dwBytes, LPOVERLAPPED lpOv) {
    printf("[Callback] Completed: %lu bytes, error: %lu\n", dwBytes, dwErr);
    completed = TRUE;
}

int main() {
    printf("Overlapped I/O Demo\n\n");
    
    // Create test file
    HANDLE hWrite = CreateFileA("async_test.txt", GENERIC_WRITE, 0, NULL,
        CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, NULL);
    const char* data = "Async file I/O test data!";
    DWORD written;
    WriteFile(hWrite, data, strlen(data), &written, NULL);
    CloseHandle(hWrite);
    
    // Open for async read
    HANDLE hFile = CreateFileA("async_test.txt", GENERIC_READ, 0, NULL,
        OPEN_EXISTING, FILE_FLAG_OVERLAPPED, NULL);
    
    char buffer[256] = {0};
    OVERLAPPED ov = {0};
    
    printf("Starting async read...\n");
    if (ReadFileEx(hFile, buffer, sizeof(buffer)-1, &ov, ReadComplete)) {
        printf("ReadFileEx queued\n");
        
        // Wait alertably for completion
        SleepEx(1000, TRUE);
        
        printf("Buffer: %s\n", buffer);
    }
    
    CloseHandle(hFile);
    DeleteFileA("async_test.txt");
    
    return 0;
}''',
[{"type": "output_contains", "value": "Async file I/O test data!"}])

add(2, "threading", "tls", "TlsAlloc", "intermediate",
"Write a C++ program demonstrating thread-local storage.",
r'''#include <windows.h>
#include <stdio.h>

DWORD tlsIndex;

DWORD WINAPI Worker(LPVOID arg) {
    int id = (int)(INT_PTR)arg;
    
    int* pValue = (int*)LocalAlloc(LPTR, sizeof(int));
    *pValue = id * 100;
    TlsSetValue(tlsIndex, pValue);
    
    printf("[Thread %d] Set TLS: %d\n", id, *pValue);
    Sleep(100);
    
    int* pRead = (int*)TlsGetValue(tlsIndex);
    printf("[Thread %d] Got TLS: %d\n", id, *pRead);
    
    LocalFree(pValue);
    return 0;
}

int main() {
    printf("Thread-Local Storage Demo\n\n");
    
    tlsIndex = TlsAlloc();
    printf("TLS index: %lu\n\n", tlsIndex);
    
    HANDLE threads[3];
    for (int i = 0; i < 3; i++) {
        threads[i] = CreateThread(NULL, 0, Worker, (LPVOID)(INT_PTR)(i+1), 0, NULL);
    }
    
    WaitForMultipleObjects(3, threads, TRUE, INFINITE);
    
    for (int i = 0; i < 3; i++) CloseHandle(threads[i]);
    TlsFree(tlsIndex);
    
    printf("\nTLS freed\n");
    return 0;
}''',
[{"type": "output_contains", "value": "TLS index:"},
 {"type": "output_contains", "value": "TLS freed"}])

add(2, "sync", "interlocked", "InterlockedIncrement", "intermediate",
"Write a C++ program demonstrating interlocked operations.",
r'''#include <windows.h>
#include <stdio.h>

volatile LONG counter = 0;
volatile LONG64 counter64 = 0;

DWORD WINAPI Worker(LPVOID arg) {
    for (int i = 0; i < 100000; i++) {
        InterlockedIncrement(&counter);
        InterlockedIncrement64(&counter64);
    }
    return 0;
}

int main() {
    printf("Interlocked Operations Demo\n\n");
    
    printf("Initial: %ld\n", counter);
    
    // Basic operations
    LONG old = InterlockedExchange(&counter, 10);
    printf("Exchange(10): old=%ld, new=%ld\n", old, counter);
    
    old = InterlockedCompareExchange(&counter, 20, 10);
    printf("CAS(20,10): old=%ld, new=%ld\n", old, counter);
    
    old = InterlockedAdd(&counter, 5);
    printf("Add(5): old=%ld, new=%ld\n", old, counter);
    
    // Multi-threaded test
    counter = 0;
    counter64 = 0;
    
    HANDLE threads[4];
    for (int i = 0; i < 4; i++) {
        threads[i] = CreateThread(NULL, 0, Worker, NULL, 0, NULL);
    }
    
    WaitForMultipleObjects(4, threads, TRUE, INFINITE);
    
    printf("\n4 threads x 100K: LONG=%ld, LONG64=%lld\n", counter, counter64);
    printf("Expected: 400000\n");
    printf("Result: %s\n", counter == 400000 ? "CORRECT" : "RACE!");
    
    for (int i = 0; i < 4; i++) CloseHandle(threads[i]);
    return 0;
}''',
[{"type": "output_contains", "value": "Result: CORRECT"}])

add(2, "ipc", "sharedmem", "CreateFileMapping_anon", "intermediate",
"Write a C++ program demonstrating shared memory.",
r'''#include <windows.h>
#include <stdio.h>

int main() {
    printf("Shared Memory Demo\n\n");
    
    const char* mapName = "Local\\TestSharedMem";
    DWORD size = 4096;
    
    // Create named mapping
    HANDLE hMap = CreateFileMappingA(INVALID_HANDLE_VALUE, NULL,
        PAGE_READWRITE, 0, size, mapName);
    
    if (!hMap) {
        printf("CreateFileMapping failed: %lu\n", GetLastError());
        return 1;
    }
    printf("Mapping created: 0x%p\n", hMap);
    
    // Map view
    LPVOID pBuf = MapViewOfFile(hMap, FILE_MAP_ALL_ACCESS, 0, 0, size);
    printf("Mapped at: 0x%p\n", pBuf);
    
    // Write data
    strcpy((char*)pBuf, "Shared memory content!");
    printf("Wrote: %s\n", (char*)pBuf);
    
    // Open again (as if from another process)
    HANDLE hMap2 = OpenFileMappingA(FILE_MAP_READ, FALSE, mapName);
    if (hMap2) {
        LPVOID pBuf2 = MapViewOfFile(hMap2, FILE_MAP_READ, 0, 0, 0);
        printf("Second view at: 0x%p\n", pBuf2);
        printf("Read from second: %s\n", (char*)pBuf2);
        UnmapViewOfFile(pBuf2);
        CloseHandle(hMap2);
    }
    
    UnmapViewOfFile(pBuf);
    CloseHandle(hMap);
    printf("\nShared memory closed\n");
    
    return 0;
}''',
[{"type": "output_contains", "value": "Shared memory content!"},
 {"type": "output_contains", "value": "Read from second:"}])

add(2, "process", "priority", "SetPriorityClass", "intermediate",
"Write a C++ program that manipulates process priority.",
r'''#include <windows.h>
#include <stdio.h>

const char* PriorityStr(DWORD p) {
    switch (p) {
        case IDLE_PRIORITY_CLASS: return "IDLE";
        case BELOW_NORMAL_PRIORITY_CLASS: return "BELOW_NORMAL";
        case NORMAL_PRIORITY_CLASS: return "NORMAL";
        case ABOVE_NORMAL_PRIORITY_CLASS: return "ABOVE_NORMAL";
        case HIGH_PRIORITY_CLASS: return "HIGH";
        case REALTIME_PRIORITY_CLASS: return "REALTIME";
        default: return "UNKNOWN";
    }
}

int main() {
    printf("Process Priority Demo\n\n");
    
    HANDLE hProc = GetCurrentProcess();
    
    DWORD priority = GetPriorityClass(hProc);
    printf("Current priority: %s (0x%lX)\n", PriorityStr(priority), priority);
    
    // Change to BELOW_NORMAL
    SetPriorityClass(hProc, BELOW_NORMAL_PRIORITY_CLASS);
    priority = GetPriorityClass(hProc);
    printf("After BELOW_NORMAL: %s\n", PriorityStr(priority));
    
    // Change to ABOVE_NORMAL
    SetPriorityClass(hProc, ABOVE_NORMAL_PRIORITY_CLASS);
    priority = GetPriorityClass(hProc);
    printf("After ABOVE_NORMAL: %s\n", PriorityStr(priority));
    
    // Restore
    SetPriorityClass(hProc, NORMAL_PRIORITY_CLASS);
    priority = GetPriorityClass(hProc);
    printf("Restored: %s\n", PriorityStr(priority));
    
    // Thread priority
    int threadPri = GetThreadPriority(GetCurrentThread());
    printf("\nThread priority: %d\n", threadPri);
    
    return 0;
}''',
[{"type": "output_contains", "value": "NORMAL"},
 {"type": "output_contains", "value": "Thread priority:"}])

add(2, "process", "affinity", "SetProcessAffinityMask", "intermediate",
"Write a C++ program that queries CPU affinity.",
r'''#include <windows.h>
#include <stdio.h>

int main() {
    printf("CPU Affinity Demo\n\n");
    
    DWORD_PTR procMask, sysMask;
    GetProcessAffinityMask(GetCurrentProcess(), &procMask, &sysMask);
    
    printf("System mask: 0x%llX\n", (ULONGLONG)sysMask);
    printf("Process mask: 0x%llX\n", (ULONGLONG)procMask);
    
    // Count CPUs
    int count = 0;
    for (DWORD_PTR m = sysMask; m; m >>= 1) {
        if (m & 1) count++;
    }
    printf("Available CPUs: %d\n\n", count);
    
    // Show which CPUs
    printf("CPU availability:\n");
    for (int i = 0; i < 16 && ((DWORD_PTR)1 << i) <= sysMask; i++) {
        DWORD_PTR bit = (DWORD_PTR)1 << i;
        if (sysMask & bit) {
            printf("  CPU %d: %s\n", i, (procMask & bit) ? "allowed" : "restricted");
        }
    }
    
    return 0;
}''',
[{"type": "output_contains", "value": "System mask:"},
 {"type": "output_contains", "value": "Available CPUs:"}])

add(2, "security", "priv", "AdjustTokenPrivileges", "intermediate",
"Write a C++ program that enables privileges.",
r'''#include <windows.h>
#include <stdio.h>
#pragma comment(lib, "advapi32.lib")

BOOL EnablePrivilege(HANDLE hToken, LPCSTR priv) {
    LUID luid;
    if (!LookupPrivilegeValueA(NULL, priv, &luid)) return FALSE;
    
    TOKEN_PRIVILEGES tp;
    tp.PrivilegeCount = 1;
    tp.Privileges[0].Luid = luid;
    tp.Privileges[0].Attributes = SE_PRIVILEGE_ENABLED;
    
    if (!AdjustTokenPrivileges(hToken, FALSE, &tp, 0, NULL, NULL)) return FALSE;
    return GetLastError() != ERROR_NOT_ALL_ASSIGNED;
}

int main() {
    printf("Privilege Adjustment Demo\n\n");
    
    HANDLE hToken;
    OpenProcessToken(GetCurrentProcess(), TOKEN_ADJUST_PRIVILEGES | TOKEN_QUERY, &hToken);
    
    const char* privs[] = {
        SE_DEBUG_NAME, SE_BACKUP_NAME, SE_RESTORE_NAME,
        SE_SHUTDOWN_NAME, SE_SECURITY_NAME
    };
    
    for (int i = 0; i < 5; i++) {
        printf("%-30s ", privs[i]);
        if (EnablePrivilege(hToken, privs[i])) {
            printf("ENABLED\n");
        } else {
            printf("FAILED (%lu)\n", GetLastError());
        }
    }
    
    printf("\n(Run as Administrator for success)\n");
    CloseHandle(hToken);
    
    return 0;
}''',
[{"type": "output_contains", "value": "SeDebugPrivilege"}])

add(2, "console", "color", "SetConsoleTextAttribute", "intermediate",
"Write a C++ program that uses console colors.",
r'''#include <windows.h>
#include <stdio.h>

int main() {
    printf("Console Colors Demo\n\n");
    
    HANDLE hCon = GetStdHandle(STD_OUTPUT_HANDLE);
    
    CONSOLE_SCREEN_BUFFER_INFO csbi;
    GetConsoleScreenBufferInfo(hCon, &csbi);
    WORD origAttr = csbi.wAttributes;
    
    // Color combinations
    struct { WORD attr; const char* name; } colors[] = {
        {FOREGROUND_RED, "Red"},
        {FOREGROUND_GREEN, "Green"},
        {FOREGROUND_BLUE, "Blue"},
        {FOREGROUND_RED | FOREGROUND_GREEN, "Yellow"},
        {FOREGROUND_RED | FOREGROUND_BLUE, "Magenta"},
        {FOREGROUND_GREEN | FOREGROUND_BLUE, "Cyan"},
        {FOREGROUND_RED | FOREGROUND_INTENSITY, "Bright Red"},
        {FOREGROUND_GREEN | FOREGROUND_INTENSITY, "Bright Green"},
    };
    
    for (int i = 0; i < 8; i++) {
        SetConsoleTextAttribute(hCon, colors[i].attr);
        printf("%s text\n", colors[i].name);
    }
    
    // Background colors
    SetConsoleTextAttribute(hCon, BACKGROUND_RED | BACKGROUND_INTENSITY);
    printf("White on red background\n");
    
    SetConsoleTextAttribute(hCon, origAttr);
    printf("\nRestored to original\n");
    
    return 0;
}''',
[{"type": "output_contains", "value": "Red text"},
 {"type": "output_contains", "value": "Restored to original"}])

add(2, "file", "watch", "FindFirstChangeNotification", "intermediate",
"Write a C++ program that watches for directory changes.",
r'''#include <windows.h>
#include <stdio.h>

int main() {
    printf("Directory Watch Demo\n\n");
    
    const char* dir = "C:\\Windows\\Temp";
    
    HANDLE hChange = FindFirstChangeNotificationA(dir, FALSE,
        FILE_NOTIFY_CHANGE_FILE_NAME | FILE_NOTIFY_CHANGE_SIZE);
    
    if (hChange == INVALID_HANDLE_VALUE) {
        printf("FindFirstChangeNotification failed: %lu\n", GetLastError());
        return 1;
    }
    
    printf("Watching: %s\n", dir);
    printf("Waiting for changes (2 second timeout)...\n\n");
    
    DWORD result = WaitForSingleObject(hChange, 2000);
    
    switch (result) {
        case WAIT_OBJECT_0:
            printf("Change detected!\n");
            FindNextChangeNotification(hChange);
            break;
        case WAIT_TIMEOUT:
            printf("Timeout - no changes\n");
            break;
        default:
            printf("Wait failed: %lu\n", GetLastError());
    }
    
    FindCloseChangeNotification(hChange);
    printf("Watch closed\n");
    
    return 0;
}''',
[{"type": "output_contains", "value": "Watching:"}])

add(2, "registry", "notify", "RegNotifyChangeKeyValue", "intermediate",
"Write a C++ program that watches for registry changes.",
r'''#include <windows.h>
#include <stdio.h>
#pragma comment(lib, "advapi32.lib")

int main() {
    printf("Registry Watch Demo\n\n");
    
    HKEY hKey;
    LONG result = RegOpenKeyExA(HKEY_CURRENT_USER, "SOFTWARE", 0,
        KEY_NOTIFY, &hKey);
    
    if (result != ERROR_SUCCESS) {
        printf("RegOpenKeyEx failed: %ld\n", result);
        return 1;
    }
    
    HANDLE hEvent = CreateEventA(NULL, TRUE, FALSE, NULL);
    
    result = RegNotifyChangeKeyValue(hKey, TRUE,
        REG_NOTIFY_CHANGE_NAME | REG_NOTIFY_CHANGE_LAST_SET,
        hEvent, TRUE);
    
    if (result != ERROR_SUCCESS) {
        printf("RegNotifyChangeKeyValue failed: %ld\n", result);
        RegCloseKey(hKey);
        return 1;
    }
    
    printf("Watching HKCU\\SOFTWARE for changes...\n");
    printf("Waiting 2 seconds...\n\n");
    
    DWORD waitResult = WaitForSingleObject(hEvent, 2000);
    
    if (waitResult == WAIT_OBJECT_0) {
        printf("Registry change detected!\n");
    } else {
        printf("Timeout - no changes\n");
    }
    
    CloseHandle(hEvent);
    RegCloseKey(hKey);
    printf("Watch closed\n");
    
    return 0;
}''',
[{"type": "output_contains", "value": "Watching HKCU"}])

# =============================================================================
# MORE TIER 3 PROBLEMS (to reach 60+)
# =============================================================================

add(3, "security", "wellknown", "CreateWellKnownSid", "intermediate",
"Write a C++ program that displays well-known SIDs.",
r'''#include <windows.h>
#include <sddl.h>
#include <stdio.h>
#pragma comment(lib, "advapi32.lib")

void ShowSid(WELL_KNOWN_SID_TYPE type, const char* name) {
    BYTE sidBuf[SECURITY_MAX_SID_SIZE];
    DWORD sidSize = sizeof(sidBuf);
    
    if (CreateWellKnownSid(type, NULL, sidBuf, &sidSize)) {
        LPSTR sidStr;
        ConvertSidToStringSidA(sidBuf, &sidStr);
        printf("%-25s %s\n", name, sidStr);
        LocalFree(sidStr);
    }
}

int main() {
    printf("Well-Known SIDs\n\n");
    printf("%-25s %s\n", "Name", "SID");
    printf("------------------------- -----------\n");
    
    ShowSid(WinWorldSid, "Everyone");
    ShowSid(WinLocalSystemSid, "SYSTEM");
    ShowSid(WinLocalServiceSid, "LOCAL SERVICE");
    ShowSid(WinNetworkServiceSid, "NETWORK SERVICE");
    ShowSid(WinBuiltinAdministratorsSid, "Administrators");
    ShowSid(WinBuiltinUsersSid, "Users");
    ShowSid(WinAuthenticatedUserSid, "Authenticated Users");
    ShowSid(WinInteractiveSid, "Interactive");
    
    return 0;
}''',
[{"type": "output_contains", "value": "Everyone"},
 {"type": "output_contains", "value": "S-1-"}])

add(3, "threading", "pool2", "CreateThreadpoolWork", "intermediate",
"Write a C++ program using the Windows thread pool.",
r'''#include <windows.h>
#include <stdio.h>

volatile LONG completed = 0;

VOID CALLBACK WorkCallback(PTP_CALLBACK_INSTANCE inst, PVOID ctx, PTP_WORK work) {
    int id = (int)(INT_PTR)ctx;
    printf("[Work %d] Executing on TID %lu\n", id, GetCurrentThreadId());
    Sleep(50);
    InterlockedIncrement(&completed);
}

int main() {
    printf("Thread Pool (Vista+) Demo\n\n");
    
    PTP_WORK works[5];
    
    printf("Creating 5 work items...\n");
    for (int i = 0; i < 5; i++) {
        works[i] = CreateThreadpoolWork(WorkCallback, (PVOID)(INT_PTR)(i+1), NULL);
        SubmitThreadpoolWork(works[i]);
    }
    
    printf("Waiting for completion...\n\n");
    
    for (int i = 0; i < 5; i++) {
        WaitForThreadpoolWorkCallbacks(works[i], FALSE);
        CloseThreadpoolWork(works[i]);
    }
    
    printf("\nCompleted: %ld items\n", completed);
    return 0;
}''',
[{"type": "output_contains", "value": "Completed: 5 items"}])

add(3, "threading", "once", "InitOnceExecuteOnce", "intermediate",
"Write a C++ program demonstrating one-time initialization.",
r'''#include <windows.h>
#include <stdio.h>

INIT_ONCE initOnce = INIT_ONCE_STATIC_INIT;
int* sharedResource = NULL;
volatile LONG initCount = 0;

BOOL CALLBACK InitFunction(PINIT_ONCE initOnce, PVOID param, PVOID* ctx) {
    InterlockedIncrement(&initCount);
    printf("[Init] Running initialization (call %ld)\n", initCount);
    
    sharedResource = (int*)HeapAlloc(GetProcessHeap(), 0, sizeof(int));
    *sharedResource = 42;
    *ctx = sharedResource;
    
    return TRUE;
}

DWORD WINAPI Worker(LPVOID arg) {
    int id = (int)(INT_PTR)arg;
    PVOID ctx;
    
    InitOnceExecuteOnce(&initOnce, InitFunction, NULL, &ctx);
    printf("[Thread %d] Resource value: %d\n", id, *(int*)ctx);
    
    return 0;
}

int main() {
    printf("One-Time Initialization Demo\n\n");
    
    HANDLE threads[4];
    for (int i = 0; i < 4; i++) {
        threads[i] = CreateThread(NULL, 0, Worker, (LPVOID)(INT_PTR)(i+1), 0, NULL);
    }
    
    WaitForMultipleObjects(4, threads, TRUE, INFINITE);
    
    printf("\nInit function called %ld time(s)\n", initCount);
    
    for (int i = 0; i < 4; i++) CloseHandle(threads[i]);
    if (sharedResource) HeapFree(GetProcessHeap(), 0, sharedResource);
    
    return 0;
}''',
[{"type": "output_contains", "value": "Init function called 1 time(s)"}])

add(3, "memory", "section", "NtCreateSection", "intermediate",
"Write a C++ program that creates a memory section via native API.",
r'''#include <windows.h>
#include <stdio.h>

typedef LONG NTSTATUS;
typedef struct _OBJECT_ATTRIBUTES {
    ULONG Length;
    HANDLE RootDirectory;
    void* ObjectName;
    ULONG Attributes;
    void* SecurityDescriptor;
    void* SecurityQualityOfService;
} OBJECT_ATTRIBUTES, *POBJECT_ATTRIBUTES;

#define InitializeObjectAttributes(p, n, a, r, s) { \
    (p)->Length = sizeof(OBJECT_ATTRIBUTES); \
    (p)->RootDirectory = r; \
    (p)->Attributes = a; \
    (p)->ObjectName = n; \
    (p)->SecurityDescriptor = s; \
    (p)->SecurityQualityOfService = NULL; }

typedef NTSTATUS (NTAPI *pNtCreateSection)(PHANDLE, ULONG, POBJECT_ATTRIBUTES, 
    PLARGE_INTEGER, ULONG, ULONG, HANDLE);
typedef NTSTATUS (NTAPI *pNtMapViewOfSection)(HANDLE, HANDLE, PVOID*, ULONG_PTR,
    SIZE_T, PLARGE_INTEGER, PSIZE_T, DWORD, ULONG, ULONG);
typedef NTSTATUS (NTAPI *pNtUnmapViewOfSection)(HANDLE, PVOID);
typedef NTSTATUS (NTAPI *pNtClose)(HANDLE);

int main() {
    printf("NtCreateSection Demo\n\n");
    
    HMODULE ntdll = GetModuleHandleA("ntdll.dll");
    pNtCreateSection NtCreateSection = (pNtCreateSection)GetProcAddress(ntdll, "NtCreateSection");
    pNtMapViewOfSection NtMapViewOfSection = (pNtMapViewOfSection)GetProcAddress(ntdll, "NtMapViewOfSection");
    pNtUnmapViewOfSection NtUnmapViewOfSection = (pNtUnmapViewOfSection)GetProcAddress(ntdll, "NtUnmapViewOfSection");
    pNtClose NtClose = (pNtClose)GetProcAddress(ntdll, "NtClose");
    
    HANDLE hSection;
    LARGE_INTEGER maxSize = {0};
    maxSize.QuadPart = 4096;
    
    OBJECT_ATTRIBUTES oa;
    InitializeObjectAttributes(&oa, NULL, 0, NULL, NULL);
    
    NTSTATUS status = NtCreateSection(&hSection, SECTION_ALL_ACCESS, &oa,
        &maxSize, PAGE_READWRITE, SEC_COMMIT, NULL);
    
    printf("NtCreateSection: 0x%lX\n", status);
    printf("Section handle: 0x%p\n", hSection);
    
    PVOID baseAddr = NULL;
    SIZE_T viewSize = 0;
    status = NtMapViewOfSection(hSection, GetCurrentProcess(), &baseAddr, 0, 0,
        NULL, &viewSize, 1, 0, PAGE_READWRITE);
    
    printf("NtMapViewOfSection: 0x%lX\n", status);
    printf("Mapped at: 0x%p (size %zu)\n", baseAddr, viewSize);
    
    strcpy((char*)baseAddr, "Hello from section!");
    printf("Wrote: %s\n", (char*)baseAddr);
    
    NtUnmapViewOfSection(GetCurrentProcess(), baseAddr);
    NtClose(hSection);
    printf("\nSection closed\n");
    
    return 0;
}''',
[{"type": "output_contains", "value": "Hello from section!"}])

add(3, "services", "control", "ControlService", "intermediate",
"Write a C++ program that queries a service status.",
r'''#include <windows.h>
#include <stdio.h>
#pragma comment(lib, "advapi32.lib")

const char* StateStr(DWORD state) {
    switch (state) {
        case SERVICE_STOPPED: return "STOPPED";
        case SERVICE_START_PENDING: return "START_PENDING";
        case SERVICE_STOP_PENDING: return "STOP_PENDING";
        case SERVICE_RUNNING: return "RUNNING";
        case SERVICE_CONTINUE_PENDING: return "CONTINUE_PENDING";
        case SERVICE_PAUSE_PENDING: return "PAUSE_PENDING";
        case SERVICE_PAUSED: return "PAUSED";
        default: return "UNKNOWN";
    }
}

int main() {
    printf("Service Query Demo\n\n");
    
    SC_HANDLE hSCM = OpenSCManagerA(NULL, NULL, SC_MANAGER_CONNECT);
    if (!hSCM) {
        printf("OpenSCManager failed: %lu\n", GetLastError());
        return 1;
    }
    
    const char* services[] = {"wuauserv", "Spooler", "Winmgmt", "EventLog"};
    
    for (int i = 0; i < 4; i++) {
        SC_HANDLE hSvc = OpenServiceA(hSCM, services[i], SERVICE_QUERY_STATUS);
        if (hSvc) {
            SERVICE_STATUS_PROCESS ssp;
            DWORD needed;
            
            if (QueryServiceStatusEx(hSvc, SC_STATUS_PROCESS_INFO,
                    (LPBYTE)&ssp, sizeof(ssp), &needed)) {
                printf("%-15s %s (PID: %lu)\n", services[i],
                    StateStr(ssp.dwCurrentState), ssp.dwProcessId);
            }
            CloseServiceHandle(hSvc);
        } else {
            printf("%-15s ERROR (%lu)\n", services[i], GetLastError());
        }
    }
    
    CloseServiceHandle(hSCM);
    return 0;
}''',
[{"type": "output_contains", "value": "RUNNING"}])

add(3, "ipc", "completion", "CreateIoCompletionPort", "intermediate",
"Write a C++ program demonstrating I/O completion ports.",
r'''#include <windows.h>
#include <stdio.h>

int main() {
    printf("I/O Completion Port Demo\n\n");
    
    // Create completion port
    HANDLE hIocp = CreateIoCompletionPort(INVALID_HANDLE_VALUE, NULL, 0, 0);
    printf("IOCP created: 0x%p\n", hIocp);
    
    // Create test file for async I/O
    HANDLE hFile = CreateFileA("iocp_test.txt",
        GENERIC_READ | GENERIC_WRITE, 0, NULL,
        CREATE_ALWAYS, FILE_FLAG_OVERLAPPED, NULL);
    
    // Associate file with IOCP
    CreateIoCompletionPort(hFile, hIocp, 0x1234, 0);
    printf("File associated with IOCP\n");
    
    // Write asynchronously
    const char* data = "IOCP test data!";
    OVERLAPPED ov = {0};
    WriteFile(hFile, data, strlen(data), NULL, &ov);
    
    // Wait for completion
    DWORD transferred;
    ULONG_PTR key;
    LPOVERLAPPED lpOv;
    
    if (GetQueuedCompletionStatus(hIocp, &transferred, &key, &lpOv, 1000)) {
        printf("Completion received:\n");
        printf("  Bytes: %lu\n", transferred);
        printf("  Key: 0x%llX\n", (ULONGLONG)key);
    }
    
    CloseHandle(hFile);
    CloseHandle(hIocp);
    DeleteFileA("iocp_test.txt");
    
    printf("\nIOCP closed\n");
    return 0;
}''',
[{"type": "output_contains", "value": "Completion received:"}])

# =============================================================================
# MORE TIER 4 PROBLEMS (to reach 50+)
# =============================================================================

add(4, "native", "object", "NtQueryObject", "advanced",
"Write a C++ program that queries object information via native API.",
r'''#include <windows.h>
#include <stdio.h>

typedef LONG NTSTATUS;
typedef struct _UNICODE_STRING { USHORT Length, MaxLength; PWSTR Buffer; } UNICODE_STRING;
typedef struct _OBJECT_TYPE_INFORMATION {
    UNICODE_STRING TypeName;
    ULONG Reserved[22];
} OBJECT_TYPE_INFORMATION;

typedef NTSTATUS (NTAPI *pNtQueryObject)(HANDLE, ULONG, PVOID, ULONG, PULONG);

int main() {
    printf("NtQueryObject Demo\n\n");
    
    HMODULE ntdll = GetModuleHandleA("ntdll.dll");
    pNtQueryObject NtQueryObject = (pNtQueryObject)GetProcAddress(ntdll, "NtQueryObject");
    
    // Create various handles
    HANDLE hEvent = CreateEventA(NULL, FALSE, FALSE, NULL);
    HANDLE hMutex = CreateMutexA(NULL, FALSE, NULL);
    HANDLE hFile = CreateFileA("nul", GENERIC_READ, 0, NULL, OPEN_EXISTING, 0, NULL);
    
    HANDLE handles[] = {hEvent, hMutex, hFile, GetCurrentProcess()};
    const char* expected[] = {"Event", "Mutant", "File", "Process"};
    
    BYTE buf[256];
    ULONG len;
    
    for (int i = 0; i < 4; i++) {
        NTSTATUS status = NtQueryObject(handles[i], 2, buf, sizeof(buf), &len);
        if (status == 0) {
            OBJECT_TYPE_INFORMATION* oti = (OBJECT_TYPE_INFORMATION*)buf;
            wprintf(L"Handle 0x%p: %.*s (expected: %hs)\n",
                handles[i], oti->TypeName.Length/2, oti->TypeName.Buffer, expected[i]);
        }
    }
    
    CloseHandle(hEvent);
    CloseHandle(hMutex);
    CloseHandle(hFile);
    
    return 0;
}''',
[{"type": "output_contains", "value": "Event"},
 {"type": "output_contains", "value": "Process"}])

add(4, "internals", "handleinfo", "NtQueryInformationFile", "advanced",
"Write a C++ program that queries file information via native API.",
r'''#include <windows.h>
#include <stdio.h>

typedef LONG NTSTATUS;
typedef struct _IO_STATUS_BLOCK { NTSTATUS Status; ULONG_PTR Information; } IO_STATUS_BLOCK;
typedef struct _FILE_STANDARD_INFORMATION {
    LARGE_INTEGER AllocationSize;
    LARGE_INTEGER EndOfFile;
    ULONG NumberOfLinks;
    BOOLEAN DeletePending;
    BOOLEAN Directory;
} FILE_STANDARD_INFORMATION;

typedef NTSTATUS (NTAPI *pNtQueryInformationFile)(HANDLE, IO_STATUS_BLOCK*, PVOID, ULONG, ULONG);

int main() {
    printf("NtQueryInformationFile Demo\n\n");
    
    HMODULE ntdll = GetModuleHandleA("ntdll.dll");
    pNtQueryInformationFile NtQueryInformationFile = 
        (pNtQueryInformationFile)GetProcAddress(ntdll, "NtQueryInformationFile");
    
    HANDLE hFile = CreateFileA("C:\\Windows\\System32\\ntdll.dll",
        GENERIC_READ, FILE_SHARE_READ, NULL, OPEN_EXISTING, 0, NULL);
    
    if (hFile == INVALID_HANDLE_VALUE) {
        printf("CreateFile failed\n");
        return 1;
    }
    
    IO_STATUS_BLOCK iosb;
    FILE_STANDARD_INFORMATION fsi;
    
    NTSTATUS status = NtQueryInformationFile(hFile, &iosb, &fsi, sizeof(fsi), 5);
    
    printf("NtQueryInformationFile: 0x%lX\n\n", status);
    if (status == 0) {
        printf("File: ntdll.dll\n");
        printf("Allocation: %lld bytes\n", fsi.AllocationSize.QuadPart);
        printf("Size: %lld bytes\n", fsi.EndOfFile.QuadPart);
        printf("Links: %lu\n", fsi.NumberOfLinks);
        printf("Delete pending: %s\n", fsi.DeletePending ? "Yes" : "No");
        printf("Directory: %s\n", fsi.Directory ? "Yes" : "No");
    }
    
    CloseHandle(hFile);
    return 0;
}''',
[{"type": "output_contains", "value": "Size:"},
 {"type": "output_contains", "value": "Directory: No"}])

add(4, "native", "memory2", "NtAllocateVirtualMemory", "advanced",
"Write a C++ program that allocates memory via native API.",
r'''#include <windows.h>
#include <stdio.h>

typedef LONG NTSTATUS;
typedef NTSTATUS (NTAPI *pNtAllocateVirtualMemory)(HANDLE, PVOID*, ULONG_PTR, PSIZE_T, ULONG, ULONG);
typedef NTSTATUS (NTAPI *pNtFreeVirtualMemory)(HANDLE, PVOID*, PSIZE_T, ULONG);

int main() {
    printf("NtAllocateVirtualMemory Demo\n\n");
    
    HMODULE ntdll = GetModuleHandleA("ntdll.dll");
    pNtAllocateVirtualMemory NtAllocateVirtualMemory = 
        (pNtAllocateVirtualMemory)GetProcAddress(ntdll, "NtAllocateVirtualMemory");
    pNtFreeVirtualMemory NtFreeVirtualMemory = 
        (pNtFreeVirtualMemory)GetProcAddress(ntdll, "NtFreeVirtualMemory");
    
    PVOID baseAddr = NULL;
    SIZE_T regionSize = 0x10000;
    
    NTSTATUS status = NtAllocateVirtualMemory(GetCurrentProcess(), &baseAddr, 0,
        &regionSize, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
    
    printf("NtAllocateVirtualMemory: 0x%lX\n", status);
    printf("Base: 0x%p\n", baseAddr);
    printf("Size: 0x%zX\n", regionSize);
    
    // Use the memory
    strcpy((char*)baseAddr, "Hello from native allocation!");
    printf("Data: %s\n", (char*)baseAddr);
    
    // Compare with VirtualAlloc
    LPVOID pWin = VirtualAlloc(NULL, 0x10000, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
    printf("\nVirtualAlloc comparison: 0x%p\n", pWin);
    VirtualFree(pWin, 0, MEM_RELEASE);
    
    // Free native allocation
    regionSize = 0;
    NtFreeVirtualMemory(GetCurrentProcess(), &baseAddr, &regionSize, MEM_RELEASE);
    printf("Native memory freed\n");
    
    return 0;
}''',
[{"type": "output_contains", "value": "Hello from native allocation!"},
 {"type": "output_contains", "value": "Native memory freed"}])

add(4, "evasion", "timing", "TimingChecks", "advanced",
"Write a C++ program demonstrating timing-based analysis detection.",
r'''#include <windows.h>
#include <stdio.h>

int main() {
    printf("Timing Analysis Detection Demo\n\n");
    
    // Method 1: RDTSC
    LARGE_INTEGER freq, t1, t2;
    QueryPerformanceFrequency(&freq);
    
    QueryPerformanceCounter(&t1);
    Sleep(10);
    QueryPerformanceCounter(&t2);
    
    double sleep10 = (double)(t2.QuadPart - t1.QuadPart) / freq.QuadPart * 1000;
    printf("[1] Sleep(10) took: %.2f ms %s\n", sleep10,
        sleep10 > 50 ? "(Suspicious)" : "(Normal)");
    
    // Method 2: GetTickCount difference
    ULONGLONG tick1 = GetTickCount64();
    volatile int x = 0;
    for (int i = 0; i < 10000000; i++) x++;
    ULONGLONG tick2 = GetTickCount64();
    
    printf("[2] 10M loop took: %llu ms %s\n", tick2 - tick1,
        (tick2 - tick1) > 100 ? "(Suspicious)" : "(Normal)");
    
    // Method 3: Check for debugger via timing
    t1.QuadPart = 0;
    QueryPerformanceCounter(&t1);
    
    // Trigger exception (if debugger attached, this takes longer)
    __try {
        RaiseException(0, 0, 0, NULL);
    } __except(EXCEPTION_EXECUTE_HANDLER) {}
    
    QueryPerformanceCounter(&t2);
    double except_us = (double)(t2.QuadPart - t1.QuadPart) / freq.QuadPart * 1e6;
    printf("[3] Exception handling: %.2f us %s\n", except_us,
        except_us > 1000 ? "(Debugger likely)" : "(Normal)");
    
    return 0;
}''',
[{"type": "output_contains", "value": "Sleep(10) took:"},
 {"type": "output_contains", "value": "Exception handling:"}])

add(4, "pe", "rdata", "IMAGE_DATA_DIRECTORY", "advanced",
"Write a C++ program that parses PE data directories.",
r'''#include <windows.h>
#include <stdio.h>

const char* DirNames[] = {
    "EXPORT", "IMPORT", "RESOURCE", "EXCEPTION",
    "SECURITY", "BASERELOC", "DEBUG", "ARCHITECTURE",
    "GLOBALPTR", "TLS", "LOAD_CONFIG", "BOUND_IMPORT",
    "IAT", "DELAY_IMPORT", "CLR", "RESERVED"
};

int main() {
    printf("PE Data Directory Parser\n\n");
    
    BYTE* base = (BYTE*)GetModuleHandleA(NULL);
    PIMAGE_DOS_HEADER dos = (PIMAGE_DOS_HEADER)base;
    PIMAGE_NT_HEADERS nt = (PIMAGE_NT_HEADERS)(base + dos->e_lfanew);
    
    printf("%-15s %-10s %s\n", "Directory", "RVA", "Size");
    printf("--------------- ---------- --------\n");
    
    for (int i = 0; i < IMAGE_NUMBEROF_DIRECTORY_ENTRIES; i++) {
        IMAGE_DATA_DIRECTORY* dir = &nt->OptionalHeader.DataDirectory[i];
        if (dir->VirtualAddress != 0 || dir->Size != 0) {
            printf("%-15s 0x%08lX 0x%lX\n",
                i < 16 ? DirNames[i] : "?",
                dir->VirtualAddress, dir->Size);
        }
    }
    
    return 0;
}''',
[{"type": "output_contains", "value": "IMPORT"},
 {"type": "output_contains", "value": "RVA"}])

add(4, "threading", "setcontext", "SetThreadContext", "advanced",
"Write a C++ program that modifies thread context.",
r'''#include <windows.h>
#include <stdio.h>

volatile BOOL running = TRUE;
volatile int counter = 0;

DWORD WINAPI Worker(LPVOID arg) {
    while (running) {
        counter++;
        Sleep(10);
    }
    return counter;
}

int main() {
    printf("SetThreadContext Demo\n\n");
    
    HANDLE hThread = CreateThread(NULL, 0, Worker, NULL, 0, NULL);
    Sleep(50);
    
    SuspendThread(hThread);
    
    CONTEXT ctx = {0};
    ctx.ContextFlags = CONTEXT_INTEGER;
    GetThreadContext(hThread, &ctx);
    
    printf("Original RAX: 0x%llX\n", ctx.Rax);
    
    // Modify context
    ctx.Rax = 0xDEADBEEF;
    
    if (SetThreadContext(hThread, &ctx)) {
        printf("SetThreadContext: SUCCESS\n");
        
        // Verify
        CONTEXT ctx2 = {0};
        ctx2.ContextFlags = CONTEXT_INTEGER;
        GetThreadContext(hThread, &ctx2);
        printf("Modified RAX: 0x%llX\n", ctx2.Rax);
    }
    
    running = FALSE;
    ResumeThread(hThread);
    WaitForSingleObject(hThread, 1000);
    
    DWORD exitCode;
    GetExitCodeThread(hThread, &exitCode);
    printf("Thread counter: %lu\n", exitCode);
    
    CloseHandle(hThread);
    return 0;
}''',
[{"type": "output_contains", "value": "SetThreadContext: SUCCESS"},
 {"type": "output_contains", "value": "Modified RAX: 0xDEADBEEF"}])

add(4, "native", "handles", "NtDuplicateObject", "advanced",
"Write a C++ program that duplicates handles via native API.",
r'''#include <windows.h>
#include <stdio.h>

typedef LONG NTSTATUS;
typedef NTSTATUS (NTAPI *pNtDuplicateObject)(HANDLE, HANDLE, HANDLE, PHANDLE, ULONG, ULONG, ULONG);

int main() {
    printf("NtDuplicateObject Demo\n\n");
    
    HMODULE ntdll = GetModuleHandleA("ntdll.dll");
    pNtDuplicateObject NtDuplicateObject = 
        (pNtDuplicateObject)GetProcAddress(ntdll, "NtDuplicateObject");
    
    // Create source handle
    HANDLE hEvent = CreateEventA(NULL, FALSE, TRUE, NULL);
    printf("Original handle: 0x%p\n", hEvent);
    
    // Duplicate via native API
    HANDLE hDup;
    NTSTATUS status = NtDuplicateObject(
        GetCurrentProcess(), hEvent,
        GetCurrentProcess(), &hDup,
        0, 0, DUPLICATE_SAME_ACCESS);
    
    printf("NtDuplicateObject: 0x%lX\n", status);
    printf("Duplicate handle: 0x%p\n", hDup);
    
    // Verify both work
    if (WaitForSingleObject(hDup, 0) == WAIT_OBJECT_0) {
        printf("Duplicate works: signaled state preserved\n");
    }
    
    CloseHandle(hDup);
    CloseHandle(hEvent);
    printf("Handles closed\n");
    
    return 0;
}''',
[{"type": "output_contains", "value": "Duplicate handle:"},
 {"type": "output_contains", "value": "signaled state preserved"}])

# =============================================================================
# ADDITIONAL PROBLEMS - BATCH 2 (To reach 200+)
# =============================================================================

# --- More Tier 1 ---
add(1, "file", "volume", "GetVolumeInformation", "beginner",
"Write a C++ program that displays volume information.",
r'''#include <windows.h>
#include <stdio.h>

int main() {
    printf("Volume Information Demo\n\n");
    
    char volName[256], fsName[256];
    DWORD serial, maxLen, flags;
    
    if (GetVolumeInformationA("C:\\", volName, sizeof(volName),
            &serial, &maxLen, &flags, fsName, sizeof(fsName))) {
        printf("Volume: %s\n", volName[0] ? volName : "(no label)");
        printf("Serial: %08lX\n", serial);
        printf("File System: %s\n", fsName);
        printf("Max Component: %lu\n", maxLen);
        printf("Flags: 0x%08lX\n", flags);
        if (flags & FILE_CASE_SENSITIVE_SEARCH) printf("  Case sensitive search\n");
        if (flags & FILE_UNICODE_ON_DISK) printf("  Unicode on disk\n");
        if (flags & FILE_SUPPORTS_ENCRYPTION) printf("  Supports encryption\n");
    }
    return 0;
}''',
[{"type": "output_contains", "value": "File System:"}])

add(1, "sysinfo", "adapter", "GetAdaptersInfo", "beginner",
"Write a C++ program that displays network adapter information.",
r'''#include <windows.h>
#include <iphlpapi.h>
#include <stdio.h>
#pragma comment(lib, "iphlpapi.lib")

int main() {
    printf("Network Adapters\n\n");
    
    ULONG size = 0;
    GetAdaptersInfo(NULL, &size);
    
    PIP_ADAPTER_INFO info = (PIP_ADAPTER_INFO)malloc(size);
    if (GetAdaptersInfo(info, &size) != NO_ERROR) {
        printf("GetAdaptersInfo failed\n");
        free(info);
        return 1;
    }
    
    PIP_ADAPTER_INFO p = info;
    while (p) {
        printf("Adapter: %s\n", p->AdapterName);
        printf("  Description: %s\n", p->Description);
        printf("  IP: %s\n", p->IpAddressList.IpAddress.String);
        printf("  Gateway: %s\n", p->GatewayList.IpAddress.String);
        printf("  MAC: %02X-%02X-%02X-%02X-%02X-%02X\n",
            p->Address[0], p->Address[1], p->Address[2],
            p->Address[3], p->Address[4], p->Address[5]);
        printf("\n");
        p = p->Next;
    }
    
    free(info);
    return 0;
}''',
[{"type": "output_contains", "value": "Adapter:"}])

add(1, "time", "timer_callback", "SetTimer", "beginner",
"Write a C++ program demonstrating timer callbacks.",
r'''#include <windows.h>
#include <stdio.h>

volatile int timerCount = 0;

VOID CALLBACK TimerProc(HWND hwnd, UINT msg, UINT_PTR id, DWORD time) {
    timerCount++;
    printf("[Timer %llu] Tick #%d at %lu ms\n", (ULONGLONG)id, timerCount, time);
}

int main() {
    printf("Timer Callback Demo\n\n");
    
    // Create a timer (requires message pump in GUI apps)
    // For console, we'll simulate with SetWaitableTimer
    
    HANDLE hTimer = CreateWaitableTimerA(NULL, FALSE, NULL);
    
    LARGE_INTEGER due;
    due.QuadPart = -500000LL;  // 50ms
    
    printf("Setting periodic timer (50ms interval)...\n\n");
    SetWaitableTimer(hTimer, &due, 50, NULL, NULL, FALSE);
    
    for (int i = 0; i < 5; i++) {
        WaitForSingleObject(hTimer, INFINITE);
        timerCount++;
        printf("Timer tick #%d\n", timerCount);
    }
    
    CancelWaitableTimer(hTimer);
    CloseHandle(hTimer);
    
    printf("\nTimer cancelled\n");
    return 0;
}''',
[{"type": "output_contains", "value": "Timer tick #5"}])

add(1, "process", "session", "ProcessIdToSessionId", "beginner",
"Write a C++ program that gets session information.",
r'''#include <windows.h>
#include <stdio.h>

int main() {
    printf("Session Information Demo\n\n");
    
    DWORD pid = GetCurrentProcessId();
    DWORD sessionId;
    
    if (ProcessIdToSessionId(pid, &sessionId)) {
        printf("Process ID: %lu\n", pid);
        printf("Session ID: %lu\n", sessionId);
    }
    
    // Get active console session
    DWORD consoleSession = WTSGetActiveConsoleSessionId();
    printf("Active console session: %lu\n", consoleSession);
    
    // Check if we're in console session
    printf("In console session: %s\n", 
        sessionId == consoleSession ? "Yes" : "No");
    
    return 0;
}''',
[{"type": "output_contains", "value": "Session ID:"}])

add(1, "memory", "heap_default", "GetProcessHeap", "beginner",
"Write a C++ program using the process default heap.",
r'''#include <windows.h>
#include <stdio.h>

int main() {
    printf("Process Heap Demo\n\n");
    
    HANDLE hHeap = GetProcessHeap();
    printf("Process heap: 0x%p\n", hHeap);
    
    // Allocate from process heap
    LPVOID p1 = HeapAlloc(hHeap, HEAP_ZERO_MEMORY, 256);
    LPVOID p2 = HeapAlloc(hHeap, HEAP_ZERO_MEMORY, 512);
    
    printf("Allocated: 0x%p, 0x%p\n", p1, p2);
    
    strcpy((char*)p1, "Heap data");
    printf("Data: %s\n", (char*)p1);
    
    // Get heap info
    SIZE_T s1 = HeapSize(hHeap, 0, p1);
    SIZE_T s2 = HeapSize(hHeap, 0, p2);
    printf("Sizes: %zu, %zu\n", s1, s2);
    
    HeapFree(hHeap, 0, p1);
    HeapFree(hHeap, 0, p2);
    printf("Freed\n");
    
    // Count process heaps
    DWORD count = GetProcessHeaps(0, NULL);
    printf("\nTotal process heaps: %lu\n", count);
    
    return 0;
}''',
[{"type": "output_contains", "value": "Process heap:"},
 {"type": "output_contains", "value": "Total process heaps:"}])

# --- More Tier 2 ---
add(2, "threading", "barrier", "InitializeSynchronizationBarrier", "intermediate",
"Write a C++ program demonstrating synchronization barriers.",
r'''#include <windows.h>
#include <stdio.h>

SYNCHRONIZATION_BARRIER barrier;

DWORD WINAPI Worker(LPVOID arg) {
    int id = (int)(INT_PTR)arg;
    
    printf("[Thread %d] Phase 1\n", id);
    EnterSynchronizationBarrier(&barrier, 0);
    
    printf("[Thread %d] Phase 2\n", id);
    EnterSynchronizationBarrier(&barrier, 0);
    
    printf("[Thread %d] Done\n", id);
    return 0;
}

int main() {
    printf("Synchronization Barrier Demo\n\n");
    
    const int N = 3;
    InitializeSynchronizationBarrier(&barrier, N, -1);
    
    HANDLE threads[N];
    for (int i = 0; i < N; i++) {
        threads[i] = CreateThread(NULL, 0, Worker, (LPVOID)(INT_PTR)(i+1), 0, NULL);
    }
    
    WaitForMultipleObjects(N, threads, TRUE, INFINITE);
    
    for (int i = 0; i < N; i++) CloseHandle(threads[i]);
    DeleteSynchronizationBarrier(&barrier);
    
    printf("\nBarrier demo complete\n");
    return 0;
}''',
[{"type": "output_contains", "value": "Barrier demo complete"}])

add(2, "memory", "working", "SetProcessWorkingSetSize", "intermediate",
"Write a C++ program that queries working set information.",
r'''#include <windows.h>
#include <psapi.h>
#include <stdio.h>
#pragma comment(lib, "psapi.lib")

int main() {
    printf("Working Set Demo\n\n");
    
    HANDLE hProc = GetCurrentProcess();
    
    SIZE_T minWS, maxWS;
    GetProcessWorkingSetSize(hProc, &minWS, &maxWS);
    printf("Working set limits:\n");
    printf("  Min: %zu KB\n", minWS / 1024);
    printf("  Max: %zu KB\n", maxWS / 1024);
    
    PROCESS_MEMORY_COUNTERS_EX pmc;
    pmc.cb = sizeof(pmc);
    GetProcessMemoryInfo(hProc, (PROCESS_MEMORY_COUNTERS*)&pmc, sizeof(pmc));
    
    printf("\nMemory usage:\n");
    printf("  Working Set: %zu KB\n", pmc.WorkingSetSize / 1024);
    printf("  Peak WS: %zu KB\n", pmc.PeakWorkingSetSize / 1024);
    printf("  Private: %zu KB\n", pmc.PrivateUsage / 1024);
    printf("  Page Faults: %lu\n", pmc.PageFaultCount);
    
    // Empty working set
    printf("\nEmptying working set...\n");
    EmptyWorkingSet(hProc);
    
    GetProcessMemoryInfo(hProc, (PROCESS_MEMORY_COUNTERS*)&pmc, sizeof(pmc));
    printf("After empty: %zu KB\n", pmc.WorkingSetSize / 1024);
    
    return 0;
}''',
[{"type": "output_contains", "value": "Working set limits:"},
 {"type": "output_contains", "value": "After empty:"}])

add(2, "file", "compress", "GetCompressedFileSize", "intermediate",
"Write a C++ program that checks file compression status.",
r'''#include <windows.h>
#include <stdio.h>

int main() {
    printf("File Compression Demo\n\n");
    
    const char* files[] = {
        "C:\\Windows\\System32\\ntdll.dll",
        "C:\\Windows\\System32\\kernel32.dll",
        "C:\\Windows\\explorer.exe"
    };
    
    printf("%-40s %12s %12s\n", "File", "Size", "Compressed");
    printf("---------------------------------------- ------------ ------------\n");
    
    for (int i = 0; i < 3; i++) {
        DWORD high, highComp;
        DWORD size = GetFileSize(CreateFileA(files[i], 0, FILE_SHARE_READ,
            NULL, OPEN_EXISTING, 0, NULL), &high);
        DWORD compSize = GetCompressedFileSizeA(files[i], &highComp);
        
        if (compSize != INVALID_FILE_SIZE) {
            const char* fname = strrchr(files[i], '\\') + 1;
            printf("%-40s %12lu %12lu\n", fname, size, compSize);
        }
    }
    
    return 0;
}''',
[{"type": "output_contains", "value": "ntdll.dll"}])

add(2, "process", "times", "GetProcessTimes", "intermediate",
"Write a C++ program that gets process timing information.",
r'''#include <windows.h>
#include <stdio.h>

double FileTimeToSeconds(FILETIME ft) {
    ULARGE_INTEGER uli;
    uli.LowPart = ft.dwLowDateTime;
    uli.HighPart = ft.dwHighDateTime;
    return uli.QuadPart / 10000000.0;
}

int main() {
    printf("Process Times Demo\n\n");
    
    // Do some work
    volatile int sum = 0;
    for (int i = 0; i < 10000000; i++) sum += i;
    
    HANDLE hProc = GetCurrentProcess();
    FILETIME creation, exit, kernel, user;
    
    GetProcessTimes(hProc, &creation, &exit, &kernel, &user);
    
    printf("Process times:\n");
    printf("  Kernel: %.3f sec\n", FileTimeToSeconds(kernel));
    printf("  User:   %.3f sec\n", FileTimeToSeconds(user));
    printf("  Total:  %.3f sec\n", 
        FileTimeToSeconds(kernel) + FileTimeToSeconds(user));
    
    // Creation time
    SYSTEMTIME st;
    FileTimeToSystemTime(&creation, &st);
    printf("\nCreated: %04d-%02d-%02d %02d:%02d:%02d\n",
        st.wYear, st.wMonth, st.wDay, st.wHour, st.wMinute, st.wSecond);
    
    return 0;
}''',
[{"type": "output_contains", "value": "Kernel:"},
 {"type": "output_contains", "value": "Created:"}])

add(2, "sync", "slim", "AcquireSRWLockShared", "intermediate",
"Write a C++ program demonstrating reader/writer lock patterns.",
r'''#include <windows.h>
#include <stdio.h>

SRWLOCK lock = SRWLOCK_INIT;
int data = 0;
volatile LONG readers = 0;
volatile LONG writers = 0;

DWORD WINAPI Reader(LPVOID arg) {
    int id = (int)(INT_PTR)arg;
    for (int i = 0; i < 5; i++) {
        AcquireSRWLockShared(&lock);
        InterlockedIncrement(&readers);
        printf("[R%d] Read: %d (readers: %ld)\n", id, data, readers);
        InterlockedDecrement(&readers);
        ReleaseSRWLockShared(&lock);
        Sleep(20);
    }
    return 0;
}

DWORD WINAPI Writer(LPVOID arg) {
    int id = (int)(INT_PTR)arg;
    for (int i = 0; i < 3; i++) {
        AcquireSRWLockExclusive(&lock);
        InterlockedIncrement(&writers);
        data++;
        printf("[W%d] Wrote: %d (writers: %ld)\n", id, data, writers);
        InterlockedDecrement(&writers);
        ReleaseSRWLockExclusive(&lock);
        Sleep(50);
    }
    return 0;
}

int main() {
    printf("SRW Lock Demo\n\n");
    
    HANDLE threads[4];
    threads[0] = CreateThread(NULL, 0, Writer, (LPVOID)1, 0, NULL);
    threads[1] = CreateThread(NULL, 0, Reader, (LPVOID)1, 0, NULL);
    threads[2] = CreateThread(NULL, 0, Reader, (LPVOID)2, 0, NULL);
    threads[3] = CreateThread(NULL, 0, Writer, (LPVOID)2, 0, NULL);
    
    WaitForMultipleObjects(4, threads, TRUE, INFINITE);
    
    for (int i = 0; i < 4; i++) CloseHandle(threads[i]);
    printf("\nFinal data: %d\n", data);
    
    return 0;
}''',
[{"type": "output_contains", "value": "Final data: 6"}])

add(2, "ipc", "atom", "GlobalAddAtom", "intermediate",
"Write a C++ program demonstrating global atoms.",
r'''#include <windows.h>
#include <stdio.h>

int main() {
    printf("Global Atom Demo\n\n");
    
    // Add atoms
    ATOM a1 = GlobalAddAtomA("TestAtom1");
    ATOM a2 = GlobalAddAtomA("TestAtom2");
    ATOM a3 = GlobalAddAtomA("AnotherAtom");
    
    printf("Created atoms:\n");
    printf("  TestAtom1: 0x%04X\n", a1);
    printf("  TestAtom2: 0x%04X\n", a2);
    printf("  AnotherAtom: 0x%04X\n", a3);
    
    // Find atom
    ATOM found = GlobalFindAtomA("TestAtom1");
    printf("\nFindAtom(TestAtom1): 0x%04X (match: %s)\n", 
        found, found == a1 ? "yes" : "no");
    
    // Get atom name
    char name[256];
    GlobalGetAtomNameA(a2, name, sizeof(name));
    printf("GetAtomName(0x%04X): %s\n", a2, name);
    
    // Delete atoms
    GlobalDeleteAtom(a1);
    GlobalDeleteAtom(a2);
    GlobalDeleteAtom(a3);
    printf("\nAtoms deleted\n");
    
    return 0;
}''',
[{"type": "output_contains", "value": "Created atoms:"},
 {"type": "output_contains", "value": "Atoms deleted"}])

add(2, "security", "impersonate", "ImpersonateSelf", "intermediate",
"Write a C++ program demonstrating thread impersonation.",
r'''#include <windows.h>
#include <stdio.h>
#pragma comment(lib, "advapi32.lib")

int main() {
    printf("Thread Impersonation Demo\n\n");
    
    // Check current token
    HANDLE hToken;
    if (OpenThreadToken(GetCurrentThread(), TOKEN_QUERY, TRUE, &hToken)) {
        printf("Thread already has token\n");
        CloseHandle(hToken);
    } else {
        printf("No thread token (using process token)\n");
    }
    
    // Impersonate self
    if (ImpersonateSelf(SecurityImpersonation)) {
        printf("ImpersonateSelf: SUCCESS\n");
        
        // Now thread has its own token
        if (OpenThreadToken(GetCurrentThread(), TOKEN_QUERY, TRUE, &hToken)) {
            printf("Thread token acquired: 0x%p\n", hToken);
            
            // Get token info
            TOKEN_TYPE type;
            DWORD len;
            GetTokenInformation(hToken, TokenType, &type, sizeof(type), &len);
            printf("Token type: %s\n", 
                type == TokenPrimary ? "Primary" : "Impersonation");
            
            CloseHandle(hToken);
        }
        
        // Revert
        RevertToSelf();
        printf("Reverted to process token\n");
    }
    
    return 0;
}''',
[{"type": "output_contains", "value": "ImpersonateSelf: SUCCESS"},
 {"type": "output_contains", "value": "Reverted to process token"}])

add(2, "console", "buffer", "CreateConsoleScreenBuffer", "intermediate",
"Write a C++ program demonstrating console screen buffers.",
r'''#include <windows.h>
#include <stdio.h>

int main() {
    printf("Console Screen Buffer Demo\n\n");
    
    HANDLE hOrig = GetStdHandle(STD_OUTPUT_HANDLE);
    
    // Create new buffer
    HANDLE hNew = CreateConsoleScreenBuffer(
        GENERIC_READ | GENERIC_WRITE,
        FILE_SHARE_READ | FILE_SHARE_WRITE,
        NULL, CONSOLE_TEXTMODE_BUFFER, NULL);
    
    if (hNew == INVALID_HANDLE_VALUE) {
        printf("CreateConsoleScreenBuffer failed\n");
        return 1;
    }
    
    // Write to new buffer
    DWORD written;
    const char* msg = "Hello from alternate buffer!";
    WriteConsoleA(hNew, msg, strlen(msg), &written, NULL);
    
    printf("Created alternate buffer: 0x%p\n", hNew);
    printf("Written to alternate: %lu chars\n", written);
    
    // We won't switch buffers in this demo as it would disrupt output
    // In real use: SetConsoleActiveScreenBuffer(hNew);
    
    // Get buffer info
    CONSOLE_SCREEN_BUFFER_INFO csbi;
    GetConsoleScreenBufferInfo(hNew, &csbi);
    printf("Alternate buffer size: %dx%d\n", csbi.dwSize.X, csbi.dwSize.Y);
    
    CloseHandle(hNew);
    printf("Buffer closed\n");
    
    return 0;
}''',
[{"type": "output_contains", "value": "Created alternate buffer:"}])

# --- More Tier 3 ---
add(3, "pe", "reloc", "IMAGE_BASE_RELOCATION", "intermediate",
"Write a C++ program that parses PE base relocations.",
r'''#include <windows.h>
#include <stdio.h>

int main() {
    printf("PE Base Relocation Parser\n\n");
    
    BYTE* base = (BYTE*)GetModuleHandleA(NULL);
    PIMAGE_DOS_HEADER dos = (PIMAGE_DOS_HEADER)base;
    PIMAGE_NT_HEADERS nt = (PIMAGE_NT_HEADERS)(base + dos->e_lfanew);
    
    DWORD relocRVA = nt->OptionalHeader.DataDirectory[IMAGE_DIRECTORY_ENTRY_BASERELOC].VirtualAddress;
    DWORD relocSize = nt->OptionalHeader.DataDirectory[IMAGE_DIRECTORY_ENTRY_BASERELOC].Size;
    
    if (!relocRVA) {
        printf("No base relocations\n");
        return 0;
    }
    
    printf("Relocation directory RVA: 0x%lX, Size: 0x%lX\n\n", relocRVA, relocSize);
    
    PIMAGE_BASE_RELOCATION reloc = (PIMAGE_BASE_RELOCATION)(base + relocRVA);
    int blockCount = 0;
    int totalRelocs = 0;
    
    while (reloc->VirtualAddress && blockCount < 5) {
        int count = (reloc->SizeOfBlock - sizeof(IMAGE_BASE_RELOCATION)) / 2;
        printf("Block %d: RVA 0x%08lX, %d entries\n",
            blockCount++, reloc->VirtualAddress, count);
        totalRelocs += count;
        reloc = (PIMAGE_BASE_RELOCATION)((BYTE*)reloc + reloc->SizeOfBlock);
    }
    
    printf("\n... (showing first 5 blocks)\n");
    printf("Counted at least %d relocations\n", totalRelocs);
    
    return 0;
}''',
[{"type": "output_contains", "value": "Relocation directory RVA:"}])

add(3, "pe", "tls", "IMAGE_TLS_DIRECTORY", "intermediate",
"Write a C++ program that examines PE TLS directory.",
r'''#include <windows.h>
#include <stdio.h>

int main() {
    printf("PE TLS Directory Parser\n\n");
    
    BYTE* base = (BYTE*)GetModuleHandleA(NULL);
    PIMAGE_DOS_HEADER dos = (PIMAGE_DOS_HEADER)base;
    PIMAGE_NT_HEADERS nt = (PIMAGE_NT_HEADERS)(base + dos->e_lfanew);
    
    DWORD tlsRVA = nt->OptionalHeader.DataDirectory[IMAGE_DIRECTORY_ENTRY_TLS].VirtualAddress;
    DWORD tlsSize = nt->OptionalHeader.DataDirectory[IMAGE_DIRECTORY_ENTRY_TLS].Size;
    
    printf("TLS Directory: RVA=0x%lX, Size=0x%lX\n\n", tlsRVA, tlsSize);
    
    if (!tlsRVA) {
        printf("No TLS directory in this executable\n");
        printf("(TLS is commonly used in larger applications)\n");
        
        // Check ntdll for example
        BYTE* ntdll = (BYTE*)GetModuleHandleA("ntdll.dll");
        PIMAGE_DOS_HEADER ndos = (PIMAGE_DOS_HEADER)ntdll;
        PIMAGE_NT_HEADERS nnt = (PIMAGE_NT_HEADERS)(ntdll + ndos->e_lfanew);
        DWORD ntlsRVA = nnt->OptionalHeader.DataDirectory[IMAGE_DIRECTORY_ENTRY_TLS].VirtualAddress;
        printf("\nntdll.dll TLS: %s\n", ntlsRVA ? "Present" : "Not present");
        
        return 0;
    }
    
    PIMAGE_TLS_DIRECTORY64 tls = (PIMAGE_TLS_DIRECTORY64)(base + tlsRVA);
    
    printf("StartAddressOfRawData: 0x%llX\n", tls->StartAddressOfRawData);
    printf("EndAddressOfRawData:   0x%llX\n", tls->EndAddressOfRawData);
    printf("AddressOfIndex:        0x%llX\n", tls->AddressOfIndex);
    printf("AddressOfCallBacks:    0x%llX\n", tls->AddressOfCallBacks);
    printf("SizeOfZeroFill:        %lu\n", tls->SizeOfZeroFill);
    
    return 0;
}''',
[{"type": "output_contains", "value": "TLS Directory:"}])

add(3, "memory", "guard", "VirtualAlloc_guard", "intermediate",
"Write a C++ program demonstrating guard pages.",
r'''#include <windows.h>
#include <stdio.h>

int main() {
    printf("Guard Page Demo\n\n");
    
    // Allocate with guard page
    LPVOID pMem = VirtualAlloc(NULL, 0x10000,
        MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE | PAGE_GUARD);
    
    printf("Allocated with PAGE_GUARD: 0x%p\n", pMem);
    
    MEMORY_BASIC_INFORMATION mbi;
    VirtualQuery(pMem, &mbi, sizeof(mbi));
    printf("Protection: 0x%lX (has guard: %s)\n",
        mbi.Protect, (mbi.Protect & PAGE_GUARD) ? "yes" : "no");
    
    // Access will trigger guard exception
    __try {
        printf("\nAccessing memory...\n");
        *(volatile int*)pMem = 42;
        printf("Written: %d\n", *(int*)pMem);
    } __except(GetExceptionCode() == STATUS_GUARD_PAGE_VIOLATION ?
               EXCEPTION_EXECUTE_HANDLER : EXCEPTION_CONTINUE_SEARCH) {
        printf("Guard page exception caught!\n");
    }
    
    // Guard is now removed (one-shot)
    VirtualQuery(pMem, &mbi, sizeof(mbi));
    printf("\nAfter access, protection: 0x%lX (has guard: %s)\n",
        mbi.Protect, (mbi.Protect & PAGE_GUARD) ? "yes" : "no");
    
    // Can now access normally
    *(int*)pMem = 123;
    printf("Second access worked: %d\n", *(int*)pMem);
    
    VirtualFree(pMem, 0, MEM_RELEASE);
    return 0;
}''',
[{"type": "output_contains", "value": "Guard page exception caught!"}])

add(3, "threading", "waitchain", "WaitChainTraversal", "intermediate",
"Write a C++ program that opens a wait chain session.",
r'''#include <windows.h>
#include <stdio.h>

typedef HANDLE (WINAPI *pOpenThreadWaitChainSession)(DWORD, LPVOID);
typedef BOOL (WINAPI *pGetThreadWaitChain)(HANDLE, DWORD_PTR, DWORD, DWORD, 
    LPDWORD, LPVOID, LPBOOL);
typedef VOID (WINAPI *pCloseThreadWaitChainSession)(HANDLE);

int main() {
    printf("Wait Chain Demo\n\n");
    
    HMODULE advapi = LoadLibraryA("advapi32.dll");
    pOpenThreadWaitChainSession OpenWC = (pOpenThreadWaitChainSession)
        GetProcAddress(advapi, "OpenThreadWaitChainSession");
    pGetThreadWaitChain GetWC = (pGetThreadWaitChain)
        GetProcAddress(advapi, "GetThreadWaitChain");
    pCloseThreadWaitChainSession CloseWC = (pCloseThreadWaitChainSession)
        GetProcAddress(advapi, "CloseThreadWaitChainSession");
    
    if (!OpenWC || !GetWC || !CloseWC) {
        printf("Wait chain APIs not available\n");
        return 0;
    }
    
    HANDLE hSession = OpenWC(0, NULL);
    if (!hSession) {
        printf("OpenThreadWaitChainSession failed: %lu\n", GetLastError());
        return 1;
    }
    printf("Session opened: 0x%p\n", hSession);
    
    DWORD nodeCount = 16;
    BYTE nodes[16 * 48] = {0};  // WAITCHAIN_NODE_INFO array
    BOOL isDeadlock;
    
    DWORD tid = GetCurrentThreadId();
    BOOL result = GetWC(hSession, 0, 0, tid, &nodeCount, nodes, &isDeadlock);
    
    printf("GetThreadWaitChain: %s\n", result ? "OK" : "FAILED");
    printf("Node count: %lu\n", nodeCount);
    printf("Deadlock: %s\n", isDeadlock ? "Yes" : "No");
    
    CloseWC(hSession);
    printf("Session closed\n");
    
    return 0;
}''',
[{"type": "output_contains", "value": "Session opened:"}])

add(3, "dll", "delay", "DelayLoadDLL", "intermediate",
"Write a C++ program demonstrating delay load information.",
r'''#include <windows.h>
#include <stdio.h>

int main() {
    printf("Delay Load Info Demo\n\n");
    
    BYTE* base = (BYTE*)GetModuleHandleA(NULL);
    PIMAGE_DOS_HEADER dos = (PIMAGE_DOS_HEADER)base;
    PIMAGE_NT_HEADERS nt = (PIMAGE_NT_HEADERS)(base + dos->e_lfanew);
    
    DWORD delayRVA = nt->OptionalHeader.DataDirectory[IMAGE_DIRECTORY_ENTRY_DELAY_IMPORT].VirtualAddress;
    DWORD delaySize = nt->OptionalHeader.DataDirectory[IMAGE_DIRECTORY_ENTRY_DELAY_IMPORT].Size;
    
    printf("Delay Import Directory:\n");
    printf("  RVA: 0x%lX\n", delayRVA);
    printf("  Size: 0x%lX\n\n", delaySize);
    
    if (!delayRVA) {
        printf("No delay imports in this executable\n");
        printf("(Delay loads are used to defer DLL loading)\n");
        
        // Check regular imports instead
        printf("\nRegular imports present: ");
        DWORD impRVA = nt->OptionalHeader.DataDirectory[IMAGE_DIRECTORY_ENTRY_IMPORT].VirtualAddress;
        printf("%s\n", impRVA ? "Yes" : "No");
    }
    
    return 0;
}''',
[{"type": "output_contains", "value": "Delay Import Directory:"}])

add(3, "security", "dacl", "SetEntriesInAcl", "intermediate",
"Write a C++ program that creates an ACL programmatically.",
r'''#include <windows.h>
#include <aclapi.h>
#include <sddl.h>
#include <stdio.h>
#pragma comment(lib, "advapi32.lib")

int main() {
    printf("ACL Creation Demo\n\n");
    
    // Get current user SID
    HANDLE hToken;
    OpenProcessToken(GetCurrentProcess(), TOKEN_QUERY, &hToken);
    
    BYTE buf[256];
    DWORD len;
    GetTokenInformation(hToken, TokenUser, buf, sizeof(buf), &len);
    PSID pUserSid = ((TOKEN_USER*)buf)->User.Sid;
    
    LPSTR sidStr;
    ConvertSidToStringSidA(pUserSid, &sidStr);
    printf("User SID: %s\n\n", sidStr);
    
    // Create access entries
    EXPLICIT_ACCESS_A ea[2] = {0};
    
    // Entry 1: User gets full control
    ea[0].grfAccessPermissions = GENERIC_ALL;
    ea[0].grfAccessMode = SET_ACCESS;
    ea[0].grfInheritance = NO_INHERITANCE;
    ea[0].Trustee.TrusteeForm = TRUSTEE_IS_SID;
    ea[0].Trustee.TrusteeType = TRUSTEE_IS_USER;
    ea[0].Trustee.ptstrName = (LPSTR)pUserSid;
    
    // Entry 2: Everyone gets read
    ea[1].grfAccessPermissions = GENERIC_READ;
    ea[1].grfAccessMode = SET_ACCESS;
    ea[1].grfInheritance = NO_INHERITANCE;
    ea[1].Trustee.TrusteeForm = TRUSTEE_IS_NAME;
    ea[1].Trustee.TrusteeType = TRUSTEE_IS_WELL_KNOWN_GROUP;
    ea[1].Trustee.ptstrName = "Everyone";
    
    PACL pAcl;
    DWORD result = SetEntriesInAclA(2, ea, NULL, &pAcl);
    
    if (result == ERROR_SUCCESS) {
        printf("ACL created successfully\n");
        printf("ACE count: %d\n", pAcl->AceCount);
        LocalFree(pAcl);
    }
    
    LocalFree(sidStr);
    CloseHandle(hToken);
    
    return 0;
}''',
[{"type": "output_contains", "value": "ACL created successfully"}])

# --- More Tier 4 ---
add(4, "native", "process2", "NtOpenProcess", "advanced",
"Write a C++ program that opens a process via native API.",
r'''#include <windows.h>
#include <stdio.h>

typedef LONG NTSTATUS;
typedef struct _MY_OBJECT_ATTRIBUTES {
    ULONG Length;
    HANDLE RootDirectory;
    void* ObjectName;
    ULONG Attributes;
    void* SecurityDescriptor;
    void* SecurityQualityOfService;
} MY_OBJECT_ATTRIBUTES, *PMY_OBJECT_ATTRIBUTES;

typedef struct _MY_CLIENT_ID2 { HANDLE Process; HANDLE Thread; } MY_CLIENT_ID2;

typedef NTSTATUS (NTAPI *pNtOpenProcess)(PHANDLE, ULONG, PMY_OBJECT_ATTRIBUTES, MY_CLIENT_ID2*);

int main() {
    printf("NtOpenProcess Demo\n\n");
    
    HMODULE ntdll = GetModuleHandleA("ntdll.dll");
    pNtOpenProcess NtOpenProcess = (pNtOpenProcess)GetProcAddress(ntdll, "NtOpenProcess");
    
    MY_OBJECT_ATTRIBUTES oa = {sizeof(MY_OBJECT_ATTRIBUTES)};
    MY_CLIENT_ID2 cid = {0};
    cid.Process = (HANDLE)(ULONG_PTR)GetCurrentProcessId();
    
    HANDLE hProc;
    NTSTATUS status = NtOpenProcess(&hProc, PROCESS_QUERY_INFORMATION, &oa, &cid);
    
    printf("NtOpenProcess: 0x%lX\n", status);
    printf("Handle: 0x%p\n", hProc);
    
    if (status == 0) {
        // Compare with OpenProcess
        HANDLE hProc2 = OpenProcess(PROCESS_QUERY_INFORMATION, FALSE, GetCurrentProcessId());
        printf("OpenProcess: 0x%p\n", hProc2);
        
        CloseHandle(hProc2);
        CloseHandle(hProc);
    }
    
    return 0;
}''',
[{"type": "output_contains", "value": "NtOpenProcess: 0x0"},
 {"type": "output_contains", "value": "Handle:"}])

add(4, "internals", "environment", "RtlQueryEnvironmentVariable", "advanced",
"Write a C++ program using Rtl environment functions.",
r'''#include <windows.h>
#include <stdio.h>

typedef struct _MY_UNICODE_STRING { 
    USHORT Length; 
    USHORT MaxLength; 
    PWSTR Buffer; 
} MY_UNICODE_STRING, *PMY_UNICODE_STRING;

typedef LONG NTSTATUS;
typedef NTSTATUS (NTAPI *pRtlQueryEnvironmentVariable_U)(
    PWSTR, PMY_UNICODE_STRING, PMY_UNICODE_STRING);
typedef VOID (NTAPI *pRtlInitUnicodeString)(PMY_UNICODE_STRING, PCWSTR);

int main() {
    printf("Rtl Environment Demo\n\n");
    
    HMODULE ntdll = GetModuleHandleA("ntdll.dll");
    pRtlQueryEnvironmentVariable_U RtlQueryEnv = 
        (pRtlQueryEnvironmentVariable_U)GetProcAddress(ntdll, "RtlQueryEnvironmentVariable_U");
    pRtlInitUnicodeString RtlInitUs = 
        (pRtlInitUnicodeString)GetProcAddress(ntdll, "RtlInitUnicodeString");
    
    MY_UNICODE_STRING varName;
    RtlInitUs(&varName, L"COMPUTERNAME");
    
    wchar_t buffer[256];
    MY_UNICODE_STRING value = {0, sizeof(buffer), buffer};
    
    NTSTATUS status = RtlQueryEnv(NULL, &varName, &value);
    
    printf("RtlQueryEnvironmentVariable_U: 0x%lX\n\n", status);
    if (status == 0) {
        wprintf(L"COMPUTERNAME = %.*s\n", value.Length/2, value.Buffer);
    }
    
    // Try PATH
    RtlInitUs(&varName, L"PATH");
    value.Length = 0;
    status = RtlQueryEnv(NULL, &varName, &value);
    if (status == 0) {
        wprintf(L"PATH (first 50 chars) = %.50s...\n", value.Buffer);
    }
    
    return 0;
}''',
[{"type": "output_contains", "value": "COMPUTERNAME ="}])

add(4, "native", "thread2", "NtCreateThreadEx", "advanced",
"Write a C++ program demonstrating NtCreateThreadEx.",
r'''#include <windows.h>
#include <stdio.h>

typedef LONG NTSTATUS;
typedef NTSTATUS (NTAPI *pNtCreateThreadEx)(
    PHANDLE, ACCESS_MASK, PVOID, HANDLE, LPTHREAD_START_ROUTINE,
    PVOID, ULONG, SIZE_T, SIZE_T, SIZE_T, PVOID);

volatile LONG threadRan = 0;

DWORD WINAPI ThreadFunc(LPVOID arg) {
    InterlockedIncrement(&threadRan);
    printf("[Thread] Running with param: 0x%p\n", arg);
    return 0x12345678;
}

int main() {
    printf("NtCreateThreadEx Demo\n\n");
    
    HMODULE ntdll = GetModuleHandleA("ntdll.dll");
    pNtCreateThreadEx NtCreateThreadEx = 
        (pNtCreateThreadEx)GetProcAddress(ntdll, "NtCreateThreadEx");
    
    printf("NtCreateThreadEx @ 0x%p\n\n", NtCreateThreadEx);
    
    HANDLE hThread;
    NTSTATUS status = NtCreateThreadEx(&hThread, THREAD_ALL_ACCESS, NULL,
        GetCurrentProcess(), ThreadFunc, (PVOID)0xABCD, 0, 0, 0, 0, NULL);
    
    printf("NtCreateThreadEx: 0x%lX\n", status);
    printf("Thread handle: 0x%p\n", hThread);
    
    if (status == 0) {
        WaitForSingleObject(hThread, 1000);
        
        DWORD exitCode;
        GetExitCodeThread(hThread, &exitCode);
        printf("Exit code: 0x%lX\n", exitCode);
        printf("Thread ran: %s\n", threadRan ? "YES" : "NO");
        
        CloseHandle(hThread);
    }
    
    return 0;
}''',
[{"type": "output_contains", "value": "NtCreateThreadEx: 0x0"},
 {"type": "output_contains", "value": "Thread ran: YES"}])

add(4, "evasion", "hook", "HookDetection", "advanced",
"Write a C++ program that detects API hooks.",
r'''#include <windows.h>
#include <stdio.h>

BOOL IsHooked(PVOID addr) {
    BYTE* func = (BYTE*)addr;
    // Check for common hook patterns
    if (func[0] == 0xE9) return TRUE;  // jmp rel32
    if (func[0] == 0xFF && func[1] == 0x25) return TRUE;  // jmp [rip+disp]
    if (func[0] == 0x48 && func[1] == 0xB8) return TRUE;  // mov rax, imm64
    if (func[0] == 0x68) return TRUE;  // push imm32
    return FALSE;
}

BOOL IsNtdllHooked(const char* name) {
    BYTE* func = (BYTE*)GetProcAddress(GetModuleHandleA("ntdll.dll"), name);
    if (!func) return FALSE;
    
    // Check syscall stub pattern
    // Expected: 4C 8B D1 B8 XX XX 00 00
    if (func[0] != 0x4C || func[3] != 0xB8) {
        return TRUE;  // Not standard pattern
    }
    return FALSE;
}

int main() {
    printf("Hook Detection Demo\n\n");
    
    const char* funcs[] = {
        "NtAllocateVirtualMemory", "NtWriteVirtualMemory",
        "NtProtectVirtualMemory", "NtOpenProcess",
        "NtCreateThreadEx", "NtQuerySystemInformation"
    };
    
    printf("%-30s %s\n", "Function", "Status");
    printf("------------------------------ --------\n");
    
    for (int i = 0; i < 6; i++) {
        BOOL hooked = IsNtdllHooked(funcs[i]);
        printf("%-30s %s\n", funcs[i], hooked ? "HOOKED" : "Clean");
    }
    
    return 0;
}''',
[{"type": "output_contains", "value": "Status"},
 {"type": "output_contains", "value": "Clean"}])

add(4, "threading", "suspend2", "NtSuspendThread", "advanced",
"Write a C++ program using native thread suspend/resume.",
r'''#include <windows.h>
#include <stdio.h>

typedef LONG NTSTATUS;
typedef NTSTATUS (NTAPI *pNtSuspendThread)(HANDLE, PULONG);
typedef NTSTATUS (NTAPI *pNtResumeThread)(HANDLE, PULONG);

volatile BOOL running = TRUE;
volatile LONG counter = 0;

DWORD WINAPI Worker(LPVOID arg) {
    while (running) {
        InterlockedIncrement(&counter);
        Sleep(10);
    }
    return counter;
}

int main() {
    printf("NtSuspendThread/NtResumeThread Demo\n\n");
    
    HMODULE ntdll = GetModuleHandleA("ntdll.dll");
    pNtSuspendThread NtSuspendThread = 
        (pNtSuspendThread)GetProcAddress(ntdll, "NtSuspendThread");
    pNtResumeThread NtResumeThread = 
        (pNtResumeThread)GetProcAddress(ntdll, "NtResumeThread");
    
    HANDLE hThread = CreateThread(NULL, 0, Worker, NULL, 0, NULL);
    Sleep(50);
    
    printf("Counter before suspend: %ld\n", counter);
    
    ULONG suspendCount;
    NTSTATUS status = NtSuspendThread(hThread, &suspendCount);
    printf("NtSuspendThread: 0x%lX (count: %lu)\n", status, suspendCount);
    
    LONG c1 = counter;
    Sleep(100);
    LONG c2 = counter;
    printf("Counter unchanged: %ld -> %ld (%s)\n", c1, c2,
        c1 == c2 ? "SUSPENDED" : "still running");
    
    status = NtResumeThread(hThread, &suspendCount);
    printf("NtResumeThread: 0x%lX (count: %lu)\n", status, suspendCount);
    
    Sleep(50);
    printf("Counter after resume: %ld\n", counter);
    
    running = FALSE;
    WaitForSingleObject(hThread, 1000);
    CloseHandle(hThread);
    
    return 0;
}''',
[{"type": "output_contains", "value": "SUSPENDED"},
 {"type": "output_contains", "value": "Counter after resume:"}])

add(4, "internals", "lpc", "NtConnectPort", "advanced",
"Write a C++ program that demonstrates LPC port concepts.",
r'''#include <windows.h>
#include <stdio.h>

typedef struct _UNICODE_STRING { USHORT Length, MaxLength; PWSTR Buffer; } UNICODE_STRING;

typedef LONG NTSTATUS;
typedef VOID (NTAPI *pRtlInitUnicodeString)(UNICODE_STRING*, PCWSTR);
typedef NTSTATUS (NTAPI *pNtConnectPort)(PHANDLE, UNICODE_STRING*, void*, void*, 
    void*, PULONG, void*, void*);

int main() {
    printf("LPC Port Demo\n\n");
    
    HMODULE ntdll = GetModuleHandleA("ntdll.dll");
    pRtlInitUnicodeString RtlInitUs = 
        (pRtlInitUnicodeString)GetProcAddress(ntdll, "RtlInitUnicodeString");
    pNtConnectPort NtConnectPort = 
        (pNtConnectPort)GetProcAddress(ntdll, "NtConnectPort");
    
    printf("NtConnectPort @ 0x%p\n\n", NtConnectPort);
    
    // Try to connect to a well-known LPC port
    // Note: Most system ports require admin or aren't accessible
    UNICODE_STRING portName;
    RtlInitUs(&portName, L"\\RPC Control\\srvsvc");
    
    HANDLE hPort;
    ULONG maxMsgLen = 0;
    NTSTATUS status = NtConnectPort(&hPort, &portName, NULL, NULL, NULL, 
        &maxMsgLen, NULL, NULL);
    
    wprintf(L"Attempted to connect to: %s\n", portName.Buffer);
    printf("NtConnectPort: 0x%lX\n", status);
    
    if (status == 0) {
        printf("Connected! Max message: %lu\n", maxMsgLen);
        CloseHandle(hPort);
    } else {
        printf("(Connection refused - expected for system ports)\n");
    }
    
    // Show that ALPC is the modern replacement
    printf("\nNote: Windows Vista+ uses ALPC instead of LPC\n");
    
    return 0;
}''',
[{"type": "output_contains", "value": "NtConnectPort @"}])

# =============================================================================
# FINAL BATCH - Reaching 200+ problems
# =============================================================================

# More Tier 1 - to reach 50+
add(1, "string", "case", "CharUpper", "beginner",
"Write a C++ program demonstrating case conversion.",
r'''#include <windows.h>
#include <stdio.h>
#pragma comment(lib, "user32.lib")

int main() {
    printf("Case Conversion Demo\n\n");
    
    char str1[] = "Hello World";
    char str2[] = "HELLO WORLD";
    
    printf("Original: %s\n", str1);
    CharUpperA(str1);
    printf("Upper: %s\n", str1);
    
    CharLowerA(str2);
    printf("Lower: %s\n", str2);
    
    // In-place single char
    char c = 'a';
    printf("\nChar '%c' upper: '%c'\n", c, (char)(ULONG_PTR)CharUpperA((LPSTR)(ULONG_PTR)c));
    
    return 0;
}''',
[{"type": "output_contains", "value": "HELLO WORLD"}])

add(1, "file", "link", "CreateHardLink", "beginner",
"Write a C++ program demonstrating hard links.",
r'''#include <windows.h>
#include <stdio.h>

int main() {
    printf("Hard Link Demo\n\n");
    
    // Create test file
    HANDLE h = CreateFileA("original.txt", GENERIC_WRITE, 0, NULL,
        CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, NULL);
    const char* data = "Test content";
    DWORD written;
    WriteFile(h, data, strlen(data), &written, NULL);
    CloseHandle(h);
    printf("Created original.txt\n");
    
    // Create hard link
    if (CreateHardLinkA("link.txt", "original.txt", NULL)) {
        printf("Created link.txt -> original.txt\n");
        
        // Both point to same data
        h = CreateFileA("link.txt", GENERIC_READ, 0, NULL,
            OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
        char buf[256] = {0};
        DWORD bytesRead;
        ReadFile(h, buf, sizeof(buf)-1, &bytesRead, NULL);
        CloseHandle(h);
        printf("Read via link: %s\n", buf);
        
        DeleteFileA("link.txt");
    } else {
        printf("CreateHardLink failed: %lu\n", GetLastError());
    }
    
    DeleteFileA("original.txt");
    printf("Files cleaned\n");
    
    return 0;
}''',
[{"type": "output_contains", "value": "Test content"}])

add(1, "sysinfo", "power", "GetSystemPowerStatus", "beginner",
"Write a C++ program that displays power status.",
r'''#include <windows.h>
#include <stdio.h>

int main() {
    printf("Power Status Demo\n\n");
    
    SYSTEM_POWER_STATUS sps;
    if (!GetSystemPowerStatus(&sps)) {
        printf("GetSystemPowerStatus failed\n");
        return 1;
    }
    
    printf("AC Line Status: %s\n",
        sps.ACLineStatus == 0 ? "Offline (Battery)" :
        sps.ACLineStatus == 1 ? "Online (AC)" : "Unknown");
    
    printf("Battery Flag: 0x%02X\n", sps.BatteryFlag);
    if (sps.BatteryFlag & 1) printf("  High\n");
    if (sps.BatteryFlag & 2) printf("  Low\n");
    if (sps.BatteryFlag & 4) printf("  Critical\n");
    if (sps.BatteryFlag & 8) printf("  Charging\n");
    if (sps.BatteryFlag & 128) printf("  No battery\n");
    
    printf("Battery Life: %d%%\n", sps.BatteryLifePercent);
    
    if (sps.BatteryLifeTime != 0xFFFFFFFF) {
        printf("Time Remaining: %lu seconds\n", sps.BatteryLifeTime);
    }
    
    return 0;
}''',
[{"type": "output_contains", "value": "AC Line Status:"}])

add(1, "console", "cp", "GetConsoleCP", "beginner",
"Write a C++ program that displays console code pages.",
r'''#include <windows.h>
#include <stdio.h>

int main() {
    printf("Console Code Page Demo\n\n");
    
    UINT inputCP = GetConsoleCP();
    UINT outputCP = GetConsoleOutputCP();
    
    printf("Input Code Page: %u\n", inputCP);
    printf("Output Code Page: %u\n", outputCP);
    
    printf("\nCommon code pages:\n");
    printf("  437 - IBM PC\n");
    printf("  850 - Latin 1\n");
    printf("  1252 - Windows ANSI\n");
    printf("  65001 - UTF-8\n");
    
    // Get ACP and OEMCP
    printf("\nSystem code pages:\n");
    printf("  ACP (ANSI): %u\n", GetACP());
    printf("  OEM: %u\n", GetOEMCP());
    
    return 0;
}''',
[{"type": "output_contains", "value": "Input Code Page:"},
 {"type": "output_contains", "value": "ACP"}])

add(1, "process", "bits", "IsWow64Process", "beginner",
"Write a C++ program that checks process bitness.",
r'''#include <windows.h>
#include <stdio.h>

int main() {
    printf("Process Bitness Demo\n\n");
    
    BOOL isWow64 = FALSE;
    IsWow64Process(GetCurrentProcess(), &isWow64);
    
    printf("Current process:\n");
    printf("  sizeof(void*): %zu bytes\n", sizeof(void*));
    printf("  Compiled as: %s\n", sizeof(void*) == 8 ? "64-bit" : "32-bit");
    printf("  IsWow64Process: %s\n", isWow64 ? "Yes (32-on-64)" : "No");
    
    // System info
    SYSTEM_INFO si;
    GetNativeSystemInfo(&si);
    printf("\nSystem architecture: ");
    switch (si.wProcessorArchitecture) {
        case PROCESSOR_ARCHITECTURE_AMD64: printf("x64\n"); break;
        case PROCESSOR_ARCHITECTURE_INTEL: printf("x86\n"); break;
        case PROCESSOR_ARCHITECTURE_ARM64: printf("ARM64\n"); break;
        default: printf("Other\n");
    }
    
    return 0;
}''',
[{"type": "output_contains", "value": "sizeof(void*):"}])

# More Tier 2 - to reach 50+
add(2, "memory", "write", "WriteProcessMemory", "intermediate",
"Write a C++ program demonstrating cross-process memory access.",
r'''#include <windows.h>
#include <stdio.h>

int main() {
    printf("WriteProcessMemory Demo\n\n");
    
    HANDLE hProc = GetCurrentProcess();
    
    // Allocate remote memory
    LPVOID pRemote = VirtualAllocEx(hProc, NULL, 4096,
        MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
    printf("Allocated: 0x%p\n", pRemote);
    
    // Write
    const char* data = "Hello via WriteProcessMemory!";
    SIZE_T written;
    WriteProcessMemory(hProc, pRemote, data, strlen(data)+1, &written);
    printf("Wrote %zu bytes\n", written);
    
    // Read back
    char buffer[256] = {0};
    SIZE_T bytesRead;
    ReadProcessMemory(hProc, pRemote, buffer, sizeof(buffer)-1, &bytesRead);
    printf("Read %zu bytes: %s\n", bytesRead, buffer);
    
    VirtualFreeEx(hProc, pRemote, 0, MEM_RELEASE);
    printf("Memory freed\n");
    
    return 0;
}''',
[{"type": "output_contains", "value": "Hello via WriteProcessMemory!"}])

add(2, "threading", "affinity", "SetThreadAffinityMask", "intermediate",
"Write a C++ program that pins threads to CPUs.",
r'''#include <windows.h>
#include <stdio.h>

DWORD WINAPI Worker(LPVOID arg) {
    int cpu = (int)(INT_PTR)arg;
    
    DWORD_PTR mask = (DWORD_PTR)1 << cpu;
    DWORD_PTR prev = SetThreadAffinityMask(GetCurrentThread(), mask);
    
    printf("[Thread] Set affinity to CPU %d (prev: 0x%llX)\n", cpu, (ULONGLONG)prev);
    
    // Do work on specific CPU
    volatile int sum = 0;
    for (int i = 0; i < 10000000; i++) sum++;
    
    return cpu;
}

int main() {
    printf("Thread Affinity Demo\n\n");
    
    DWORD_PTR procMask, sysMask;
    GetProcessAffinityMask(GetCurrentProcess(), &procMask, &sysMask);
    printf("Process mask: 0x%llX\n\n", (ULONGLONG)procMask);
    
    // Create threads pinned to first 2 CPUs
    HANDLE threads[2];
    threads[0] = CreateThread(NULL, 0, Worker, (LPVOID)0, 0, NULL);
    threads[1] = CreateThread(NULL, 0, Worker, (LPVOID)1, 0, NULL);
    
    WaitForMultipleObjects(2, threads, TRUE, INFINITE);
    
    for (int i = 0; i < 2; i++) CloseHandle(threads[i]);
    printf("\nThreads completed\n");
    
    return 0;
}''',
[{"type": "output_contains", "value": "Set affinity to CPU"}])

add(2, "security", "owner", "SetSecurityInfo", "intermediate",
"Write a C++ program that queries file owner.",
r'''#include <windows.h>
#include <aclapi.h>
#include <sddl.h>
#include <stdio.h>
#pragma comment(lib, "advapi32.lib")

int main() {
    printf("File Owner Demo\n\n");
    
    const char* files[] = {
        "C:\\Windows\\System32\\ntdll.dll",
        "C:\\Windows\\explorer.exe",
        "C:\\Windows\\System32\\cmd.exe"
    };
    
    for (int i = 0; i < 3; i++) {
        PSECURITY_DESCRIPTOR pSD;
        PSID pOwner;
        
        if (GetNamedSecurityInfoA(files[i], SE_FILE_OBJECT,
                OWNER_SECURITY_INFORMATION, &pOwner, NULL, NULL, NULL, &pSD) == ERROR_SUCCESS) {
            
            char name[256], domain[256];
            DWORD nameLen = 256, domLen = 256;
            SID_NAME_USE use;
            
            if (LookupAccountSidA(NULL, pOwner, name, &nameLen, domain, &domLen, &use)) {
                const char* fname = strrchr(files[i], '\\') + 1;
                printf("%-15s Owner: %s\\%s\n", fname, domain, name);
            }
            
            LocalFree(pSD);
        }
    }
    
    return 0;
}''',
[{"type": "output_contains", "value": "ntdll.dll"}])

add(2, "file", "stream", "NtfsStreams", "intermediate",
"Write a C++ program demonstrating NTFS alternate data streams.",
r'''#include <windows.h>
#include <stdio.h>

int main() {
    printf("NTFS Alternate Data Stream Demo\n\n");
    
    // Create base file
    HANDLE h = CreateFileA("test_ads.txt", GENERIC_WRITE, 0, NULL,
        CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, NULL);
    const char* mainData = "Main file content";
    DWORD written;
    WriteFile(h, mainData, strlen(mainData), &written, NULL);
    CloseHandle(h);
    printf("Created main file: test_ads.txt\n");
    
    // Create alternate stream
    h = CreateFileA("test_ads.txt:hidden", GENERIC_WRITE, 0, NULL,
        CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, NULL);
    const char* hiddenData = "Hidden stream data!";
    WriteFile(h, hiddenData, strlen(hiddenData), &written, NULL);
    CloseHandle(h);
    printf("Created ADS: test_ads.txt:hidden\n");
    
    // Read main
    h = CreateFileA("test_ads.txt", GENERIC_READ, 0, NULL,
        OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
    char buf[256] = {0};
    DWORD bytesRead;
    ReadFile(h, buf, sizeof(buf)-1, &bytesRead, NULL);
    CloseHandle(h);
    printf("Main: %s\n", buf);
    
    // Read ADS
    h = CreateFileA("test_ads.txt:hidden", GENERIC_READ, 0, NULL,
        OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
    memset(buf, 0, sizeof(buf));
    ReadFile(h, buf, sizeof(buf)-1, &bytesRead, NULL);
    CloseHandle(h);
    printf("ADS: %s\n", buf);
    
    DeleteFileA("test_ads.txt");
    printf("\nFile deleted\n");
    
    return 0;
}''',
[{"type": "output_contains", "value": "Hidden stream data!"}])

add(2, "ipc", "anon_pipe", "CreatePipe", "intermediate",
"Write a C++ program demonstrating anonymous pipes.",
r'''#include <windows.h>
#include <stdio.h>

int main() {
    printf("Anonymous Pipe Demo\n\n");
    
    HANDLE hRead, hWrite;
    SECURITY_ATTRIBUTES sa = {sizeof(sa), NULL, TRUE};
    
    if (!CreatePipe(&hRead, &hWrite, &sa, 0)) {
        printf("CreatePipe failed\n");
        return 1;
    }
    
    printf("Pipe created:\n");
    printf("  Read handle: 0x%p\n", hRead);
    printf("  Write handle: 0x%p\n", hWrite);
    
    // Write
    const char* msg = "Hello through pipe!";
    DWORD written;
    WriteFile(hWrite, msg, strlen(msg)+1, &written, NULL);
    printf("\nWrote: %s (%lu bytes)\n", msg, written);
    
    // Read
    char buffer[256] = {0};
    DWORD bytesRead;
    ReadFile(hRead, buffer, sizeof(buffer)-1, &bytesRead, NULL);
    printf("Read: %s (%lu bytes)\n", buffer, bytesRead);
    
    CloseHandle(hRead);
    CloseHandle(hWrite);
    printf("\nPipes closed\n");
    
    return 0;
}''',
[{"type": "output_contains", "value": "Hello through pipe!"}])

add(2, "process", "inherit", "InheritHandles", "intermediate",
"Write a C++ program demonstrating handle inheritance setup.",
r'''#include <windows.h>
#include <stdio.h>

int main() {
    printf("Handle Inheritance Demo\n\n");
    
    // Create inheritable handle
    SECURITY_ATTRIBUTES sa = {sizeof(sa), NULL, TRUE};  // bInheritHandle = TRUE
    HANDLE hEvent = CreateEventA(&sa, FALSE, FALSE, NULL);
    
    printf("Created inheritable event: 0x%p\n", hEvent);
    
    // Check handle info
    DWORD flags;
    GetHandleInformation(hEvent, &flags);
    printf("HANDLE_FLAG_INHERIT: %s\n", 
        (flags & HANDLE_FLAG_INHERIT) ? "Yes" : "No");
    
    // Create non-inheritable
    HANDLE hEvent2 = CreateEventA(NULL, FALSE, FALSE, NULL);
    GetHandleInformation(hEvent2, &flags);
    printf("\nNon-inheritable event: 0x%p\n", hEvent2);
    printf("HANDLE_FLAG_INHERIT: %s\n", 
        (flags & HANDLE_FLAG_INHERIT) ? "Yes" : "No");
    
    // Modify inheritance
    SetHandleInformation(hEvent2, HANDLE_FLAG_INHERIT, HANDLE_FLAG_INHERIT);
    GetHandleInformation(hEvent2, &flags);
    printf("After SetHandleInformation: %s\n",
        (flags & HANDLE_FLAG_INHERIT) ? "Yes" : "No");
    
    CloseHandle(hEvent);
    CloseHandle(hEvent2);
    
    return 0;
}''',
[{"type": "output_contains", "value": "HANDLE_FLAG_INHERIT: Yes"}])

add(2, "registry", "transact", "RegOpenKeyTransacted", "intermediate",
"Write a C++ program demonstrating registry value types.",
r'''#include <windows.h>
#include <stdio.h>
#pragma comment(lib, "advapi32.lib")

int main() {
    printf("Registry Value Types Demo\n\n");
    
    HKEY hKey;
    RegCreateKeyExA(HKEY_CURRENT_USER, "SOFTWARE\\TestValueTypes", 0, NULL,
        REG_OPTION_NON_VOLATILE, KEY_ALL_ACCESS, NULL, &hKey, NULL);
    
    // REG_SZ
    const char* str = "String value";
    RegSetValueExA(hKey, "StringVal", 0, REG_SZ, (BYTE*)str, strlen(str)+1);
    printf("Set REG_SZ: %s\n", str);
    
    // REG_DWORD
    DWORD dw = 0x12345678;
    RegSetValueExA(hKey, "DwordVal", 0, REG_DWORD, (BYTE*)&dw, sizeof(dw));
    printf("Set REG_DWORD: 0x%lX\n", dw);
    
    // REG_BINARY
    BYTE bin[] = {0xDE, 0xAD, 0xBE, 0xEF};
    RegSetValueExA(hKey, "BinaryVal", 0, REG_BINARY, bin, sizeof(bin));
    printf("Set REG_BINARY: DEADBEEF\n");
    
    // REG_MULTI_SZ
    const char multi[] = "First\0Second\0Third\0";
    RegSetValueExA(hKey, "MultiVal", 0, REG_MULTI_SZ, (BYTE*)multi, sizeof(multi));
    printf("Set REG_MULTI_SZ: First, Second, Third\n");
    
    // Read back types
    printf("\nValue types:\n");
    DWORD type, size = 256;
    BYTE buf[256];
    
    RegQueryValueExA(hKey, "StringVal", NULL, &type, buf, &size);
    printf("  StringVal: %lu (REG_SZ=%d)\n", type, REG_SZ);
    
    size = sizeof(buf);
    RegQueryValueExA(hKey, "DwordVal", NULL, &type, buf, &size);
    printf("  DwordVal: %lu (REG_DWORD=%d)\n", type, REG_DWORD);
    
    RegCloseKey(hKey);
    RegDeleteKeyA(HKEY_CURRENT_USER, "SOFTWARE\\TestValueTypes");
    printf("\nKey deleted\n");
    
    return 0;
}''',
[{"type": "output_contains", "value": "Set REG_SZ:"},
 {"type": "output_contains", "value": "Key deleted"}])

# More Tier 3 - to reach 60+
add(3, "pe", "debug", "IMAGE_DEBUG_DIRECTORY", "intermediate",
"Write a C++ program that parses PE debug information.",
r'''#include <windows.h>
#include <stdio.h>

int main() {
    printf("PE Debug Directory Parser\n\n");
    
    BYTE* base = (BYTE*)GetModuleHandleA(NULL);
    PIMAGE_DOS_HEADER dos = (PIMAGE_DOS_HEADER)base;
    PIMAGE_NT_HEADERS nt = (PIMAGE_NT_HEADERS)(base + dos->e_lfanew);
    
    DWORD dbgRVA = nt->OptionalHeader.DataDirectory[IMAGE_DIRECTORY_ENTRY_DEBUG].VirtualAddress;
    DWORD dbgSize = nt->OptionalHeader.DataDirectory[IMAGE_DIRECTORY_ENTRY_DEBUG].Size;
    
    printf("Debug Directory: RVA=0x%lX, Size=0x%lX\n\n", dbgRVA, dbgSize);
    
    if (!dbgRVA) {
        printf("No debug directory\n");
        return 0;
    }
    
    int count = dbgSize / sizeof(IMAGE_DEBUG_DIRECTORY);
    PIMAGE_DEBUG_DIRECTORY dbg = (PIMAGE_DEBUG_DIRECTORY)(base + dbgRVA);
    
    printf("%-5s %-12s %-10s %s\n", "Entry", "Type", "RVA", "Size");
    printf("----- ------------ ---------- --------\n");
    
    for (int i = 0; i < count; i++) {
        const char* typeStr;
        switch (dbg[i].Type) {
            case IMAGE_DEBUG_TYPE_CODEVIEW: typeStr = "CodeView"; break;
            case IMAGE_DEBUG_TYPE_MISC: typeStr = "Misc"; break;
            case IMAGE_DEBUG_TYPE_POGO: typeStr = "POGO"; break;
            case IMAGE_DEBUG_TYPE_ILTCG: typeStr = "ILTCG"; break;
            default: typeStr = "Other";
        }
        printf("%-5d %-12s 0x%08lX 0x%lX\n",
            i, typeStr, dbg[i].AddressOfRawData, dbg[i].SizeOfData);
    }
    
    return 0;
}''',
[{"type": "output_contains", "value": "Debug Directory:"}])

add(3, "memory", "awe", "AllocateUserPhysicalPages", "intermediate",
"Write a C++ program that demonstrates AWE concepts.",
r'''#include <windows.h>
#include <stdio.h>

int main() {
    printf("Address Windowing Extensions (AWE) Demo\n\n");
    
    // AWE requires SeLockMemoryPrivilege, usually unavailable
    // This demonstrates the concept and API structure
    
    // Check page size
    SYSTEM_INFO si;
    GetSystemInfo(&si);
    printf("Page size: %lu bytes\n", si.dwPageSize);
    
    // Calculate large page size
    SIZE_T largePageSize = GetLargePageMinimum();
    printf("Large page minimum: %zu bytes (%zu MB)\n",
        largePageSize, largePageSize / (1024*1024));
    
    // AWE requires:
    printf("\nAWE requirements:\n");
    printf("  - SeLockMemoryPrivilege (Local Security Policy)\n");
    printf("  - Physical memory allocation\n");
    printf("  - Virtual address window creation\n");
    printf("  - MapUserPhysicalPages for mapping\n");
    
    // Demonstrate standard large page (if available)
    if (largePageSize > 0) {
        LPVOID p = VirtualAlloc(NULL, largePageSize,
            MEM_COMMIT | MEM_RESERVE | MEM_LARGE_PAGES, PAGE_READWRITE);
        if (p) {
            printf("\nLarge page allocated: 0x%p\n", p);
            VirtualFree(p, 0, MEM_RELEASE);
        } else {
            printf("\nLarge pages not available: %lu\n", GetLastError());
        }
    }
    
    return 0;
}''',
[{"type": "output_contains", "value": "Page size:"}])

add(3, "native", "ntfile", "NtCreateFile", "intermediate",
"Write a C++ program using NtCreateFile.",
r'''#include <windows.h>
#include <stdio.h>

typedef LONG NTSTATUS;
typedef struct _UNICODE_STRING { USHORT Length, MaxLength; PWSTR Buffer; } UNICODE_STRING;
typedef struct _OBJECT_ATTRIBUTES {
    ULONG Length;
    HANDLE RootDirectory;
    UNICODE_STRING* ObjectName;
    ULONG Attributes;
    void* SecurityDescriptor;
    void* SecurityQualityOfService;
} OBJECT_ATTRIBUTES;
typedef struct _IO_STATUS_BLOCK { NTSTATUS Status; ULONG_PTR Information; } IO_STATUS_BLOCK;

typedef NTSTATUS (NTAPI *pNtCreateFile)(PHANDLE, ACCESS_MASK, OBJECT_ATTRIBUTES*, 
    IO_STATUS_BLOCK*, PLARGE_INTEGER, ULONG, ULONG, ULONG, ULONG, PVOID, ULONG);
typedef VOID (NTAPI *pRtlInitUnicodeString)(UNICODE_STRING*, PCWSTR);
typedef NTSTATUS (NTAPI *pNtClose)(HANDLE);

int main() {
    printf("NtCreateFile Demo\n\n");
    
    HMODULE ntdll = GetModuleHandleA("ntdll.dll");
    pNtCreateFile NtCreateFile = (pNtCreateFile)GetProcAddress(ntdll, "NtCreateFile");
    pRtlInitUnicodeString RtlInitUs = (pRtlInitUnicodeString)GetProcAddress(ntdll, "RtlInitUnicodeString");
    pNtClose NtClose = (pNtClose)GetProcAddress(ntdll, "NtClose");
    
    // NT path format required
    UNICODE_STRING path;
    RtlInitUs(&path, L"\\??\\C:\\Windows\\System32\\ntdll.dll");
    
    OBJECT_ATTRIBUTES oa = {sizeof(OBJECT_ATTRIBUTES)};
    oa.ObjectName = &path;
    oa.Attributes = 0x40;  // OBJ_CASE_INSENSITIVE
    
    HANDLE hFile;
    IO_STATUS_BLOCK iosb;
    
    NTSTATUS status = NtCreateFile(&hFile, FILE_READ_ATTRIBUTES, &oa, &iosb,
        NULL, 0, FILE_SHARE_READ, 1, 0x20, NULL, 0);
    
    wprintf(L"Opening: %s\n", path.Buffer);
    printf("NtCreateFile: 0x%lX\n", status);
    printf("Handle: 0x%p\n", hFile);
    
    if (status == 0) {
        NtClose(hFile);
        printf("File closed\n");
    }
    
    return 0;
}''',
[{"type": "output_contains", "value": "NtCreateFile: 0x0"}])

add(3, "security", "token2", "DuplicateTokenEx", "intermediate",
"Write a C++ program demonstrating token duplication.",
r'''#include <windows.h>
#include <stdio.h>
#pragma comment(lib, "advapi32.lib")

int main() {
    printf("Token Duplication Demo\n\n");
    
    HANDLE hToken, hDupToken;
    OpenProcessToken(GetCurrentProcess(), TOKEN_ALL_ACCESS, &hToken);
    
    printf("Original token: 0x%p\n", hToken);
    
    // Duplicate as primary token
    if (DuplicateTokenEx(hToken, MAXIMUM_ALLOWED, NULL,
            SecurityImpersonation, TokenPrimary, &hDupToken)) {
        printf("Duplicated (Primary): 0x%p\n", hDupToken);
        
        // Query token type
        TOKEN_TYPE type;
        DWORD len;
        GetTokenInformation(hDupToken, TokenType, &type, sizeof(type), &len);
        printf("Type: %s\n", type == TokenPrimary ? "Primary" : "Impersonation");
        
        CloseHandle(hDupToken);
    }
    
    // Duplicate as impersonation token
    if (DuplicateTokenEx(hToken, MAXIMUM_ALLOWED, NULL,
            SecurityImpersonation, TokenImpersonation, &hDupToken)) {
        printf("Duplicated (Impersonation): 0x%p\n", hDupToken);
        
        TOKEN_TYPE type;
        DWORD len;
        GetTokenInformation(hDupToken, TokenType, &type, sizeof(type), &len);
        printf("Type: %s\n", type == TokenPrimary ? "Primary" : "Impersonation");
        
        CloseHandle(hDupToken);
    }
    
    CloseHandle(hToken);
    return 0;
}''',
[{"type": "output_contains", "value": "Duplicated (Primary):"},
 {"type": "output_contains", "value": "Duplicated (Impersonation):"}])

add(3, "threading", "completion2", "SetThreadpoolTimer", "intermediate",
"Write a C++ program using threadpool timers.",
r'''#include <windows.h>
#include <stdio.h>

volatile LONG timerCount = 0;

VOID CALLBACK TimerCallback(PTP_CALLBACK_INSTANCE inst, PVOID ctx, PTP_TIMER timer) {
    LONG count = InterlockedIncrement(&timerCount);
    printf("[Timer] Callback #%ld at %lu ms\n", count, GetTickCount());
}

int main() {
    printf("Threadpool Timer Demo\n\n");
    
    PTP_TIMER timer = CreateThreadpoolTimer(TimerCallback, NULL, NULL);
    if (!timer) {
        printf("CreateThreadpoolTimer failed\n");
        return 1;
    }
    
    printf("Timer created: 0x%p\n", timer);
    
    // Set timer: first callback in 100ms, then every 100ms
    ULARGE_INTEGER due;
    due.QuadPart = (ULONGLONG)(-1000000LL);  // 100ms relative
    FILETIME ft;
    ft.dwLowDateTime = due.LowPart;
    ft.dwHighDateTime = due.HighPart;
    
    SetThreadpoolTimer(timer, &ft, 100, 0);
    printf("Timer armed\n\n");
    
    // Wait for 5 callbacks
    while (timerCount < 5) {
        Sleep(50);
    }
    
    // Stop timer
    SetThreadpoolTimer(timer, NULL, 0, 0);
    WaitForThreadpoolTimerCallbacks(timer, TRUE);
    CloseThreadpoolTimer(timer);
    
    printf("\nTimer closed (total callbacks: %ld)\n", timerCount);
    return 0;
}''',
[{"type": "output_contains", "value": "Timer closed (total callbacks: 5)"}])

add(3, "ipc", "socket", "WSAStartup", "intermediate",
"Write a C++ program demonstrating Winsock initialization.",
r'''#include <winsock2.h>
#include <windows.h>
#include <stdio.h>
#pragma comment(lib, "ws2_32.lib")

int main() {
    printf("Winsock Demo\n\n");
    
    WSADATA wsa;
    int result = WSAStartup(MAKEWORD(2, 2), &wsa);
    
    if (result != 0) {
        printf("WSAStartup failed: %d\n", result);
        return 1;
    }
    
    printf("Winsock initialized:\n");
    printf("  Version: %d.%d\n", LOBYTE(wsa.wVersion), HIBYTE(wsa.wVersion));
    printf("  High version: %d.%d\n", LOBYTE(wsa.wHighVersion), HIBYTE(wsa.wHighVersion));
    printf("  Description: %s\n", wsa.szDescription);
    printf("  System status: %s\n", wsa.szSystemStatus);
    
    // Get hostname
    char hostname[256];
    gethostname(hostname, sizeof(hostname));
    printf("\nHostname: %s\n", hostname);
    
    // Create a socket (just to demonstrate)
    SOCKET s = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    if (s != INVALID_SOCKET) {
        printf("TCP socket created: %llu\n", (ULONGLONG)s);
        closesocket(s);
    }
    
    WSACleanup();
    printf("\nWinsock cleaned up\n");
    
    return 0;
}''',
[{"type": "output_contains", "value": "Winsock initialized:"},
 {"type": "output_contains", "value": "Hostname:"}])

# More Tier 4 - to reach 50+
add(4, "native", "registry", "NtOpenKey", "advanced",
"Write a C++ program accessing registry via native API.",
r'''#include <windows.h>
#include <stdio.h>

typedef LONG NTSTATUS;
typedef struct _UNICODE_STRING { USHORT Length, MaxLength; PWSTR Buffer; } UNICODE_STRING;
typedef struct _OBJECT_ATTRIBUTES {
    ULONG Length;
    HANDLE RootDirectory;
    UNICODE_STRING* ObjectName;
    ULONG Attributes;
    void* SecurityDescriptor;
    void* SecurityQualityOfService;
} OBJECT_ATTRIBUTES;

typedef NTSTATUS (NTAPI *pNtOpenKey)(PHANDLE, ACCESS_MASK, OBJECT_ATTRIBUTES*);
typedef NTSTATUS (NTAPI *pNtQueryValueKey)(HANDLE, UNICODE_STRING*, ULONG, PVOID, ULONG, PULONG);
typedef VOID (NTAPI *pRtlInitUnicodeString)(UNICODE_STRING*, PCWSTR);
typedef NTSTATUS (NTAPI *pNtClose)(HANDLE);

int main() {
    printf("NtOpenKey Demo\n\n");
    
    HMODULE ntdll = GetModuleHandleA("ntdll.dll");
    pNtOpenKey NtOpenKey = (pNtOpenKey)GetProcAddress(ntdll, "NtOpenKey");
    pNtQueryValueKey NtQueryValueKey = (pNtQueryValueKey)GetProcAddress(ntdll, "NtQueryValueKey");
    pRtlInitUnicodeString RtlInitUs = (pRtlInitUnicodeString)GetProcAddress(ntdll, "RtlInitUnicodeString");
    pNtClose NtClose = (pNtClose)GetProcAddress(ntdll, "NtClose");
    
    UNICODE_STRING keyPath;
    RtlInitUs(&keyPath, L"\\Registry\\Machine\\SOFTWARE\\Microsoft\\Windows NT\\CurrentVersion");
    
    OBJECT_ATTRIBUTES oa = {sizeof(OBJECT_ATTRIBUTES)};
    oa.ObjectName = &keyPath;
    oa.Attributes = 0x40;
    
    HANDLE hKey;
    NTSTATUS status = NtOpenKey(&hKey, KEY_READ, &oa);
    
    wprintf(L"Opening: %s\n", keyPath.Buffer);
    printf("NtOpenKey: 0x%lX\n", status);
    printf("Handle: 0x%p\n", hKey);
    
    if (status == 0) {
        NtClose(hKey);
        printf("Key closed\n");
    }
    
    return 0;
}''',
[{"type": "output_contains", "value": "NtOpenKey: 0x0"}])

add(4, "internals", "heap2", "RtlCreateHeap", "advanced",
"Write a C++ program using native heap functions.",
r'''#include <windows.h>
#include <stdio.h>

typedef PVOID (NTAPI *pRtlCreateHeap)(ULONG, PVOID, SIZE_T, SIZE_T, PVOID, PVOID);
typedef PVOID (NTAPI *pRtlAllocateHeap)(PVOID, ULONG, SIZE_T);
typedef BOOLEAN (NTAPI *pRtlFreeHeap)(PVOID, ULONG, PVOID);
typedef PVOID (NTAPI *pRtlDestroyHeap)(PVOID);

int main() {
    printf("RtlCreateHeap Demo\n\n");
    
    HMODULE ntdll = GetModuleHandleA("ntdll.dll");
    pRtlCreateHeap RtlCreateHeap = (pRtlCreateHeap)GetProcAddress(ntdll, "RtlCreateHeap");
    pRtlAllocateHeap RtlAllocateHeap = (pRtlAllocateHeap)GetProcAddress(ntdll, "RtlAllocateHeap");
    pRtlFreeHeap RtlFreeHeap = (pRtlFreeHeap)GetProcAddress(ntdll, "RtlFreeHeap");
    pRtlDestroyHeap RtlDestroyHeap = (pRtlDestroyHeap)GetProcAddress(ntdll, "RtlDestroyHeap");
    
    // Create heap (flags 0 = growable)
    PVOID hHeap = RtlCreateHeap(0, NULL, 0x10000, 0x1000, NULL, NULL);
    printf("RtlCreateHeap: 0x%p\n", hHeap);
    
    // Allocate
    PVOID p1 = RtlAllocateHeap(hHeap, 0x08, 256);  // HEAP_ZERO_MEMORY
    PVOID p2 = RtlAllocateHeap(hHeap, 0x08, 512);
    printf("Allocations: 0x%p, 0x%p\n", p1, p2);
    
    // Use memory
    strcpy((char*)p1, "Native heap allocation!");
    printf("Data: %s\n", (char*)p1);
    
    // Free
    RtlFreeHeap(hHeap, 0, p1);
    RtlFreeHeap(hHeap, 0, p2);
    printf("Memory freed\n");
    
    // Destroy
    RtlDestroyHeap(hHeap);
    printf("Heap destroyed\n");
    
    return 0;
}''',
[{"type": "output_contains", "value": "Native heap allocation!"},
 {"type": "output_contains", "value": "Heap destroyed"}])

add(4, "evasion", "parent", "ParentPidSpoof", "advanced",
"Write a C++ program demonstrating parent PID concept.",
r'''#include <windows.h>
#include <stdio.h>
#include <tlhelp32.h>

DWORD GetParentPid(DWORD pid) {
    HANDLE hSnap = CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0);
    if (hSnap == INVALID_HANDLE_VALUE) return 0;
    
    PROCESSENTRY32 pe = {sizeof(pe)};
    DWORD ppid = 0;
    
    if (Process32First(hSnap, &pe)) {
        do {
            if (pe.th32ProcessID == pid) {
                ppid = pe.th32ParentProcessID;
                break;
            }
        } while (Process32Next(hSnap, &pe));
    }
    
    CloseHandle(hSnap);
    return ppid;
}

int main() {
    printf("Parent PID Demo\n\n");
    
    DWORD pid = GetCurrentProcessId();
    DWORD ppid = GetParentPid(pid);
    
    printf("Current PID: %lu\n", pid);
    printf("Parent PID: %lu\n", ppid);
    
    // Get parent name
    HANDLE hSnap = CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0);
    PROCESSENTRY32 pe = {sizeof(pe)};
    
    if (Process32First(hSnap, &pe)) {
        do {
            if (pe.th32ProcessID == ppid) {
                printf("Parent name: %s\n", pe.szExeFile);
                break;
            }
        } while (Process32Next(hSnap, &pe));
    }
    CloseHandle(hSnap);
    
    // Note about PPID spoofing
    printf("\nNote: PPID can be spoofed via PROC_THREAD_ATTRIBUTE_PARENT_PROCESS\n");
    printf("in STARTUPINFOEX when calling CreateProcess.\n");
    
    return 0;
}''',
[{"type": "output_contains", "value": "Current PID:"},
 {"type": "output_contains", "value": "Parent name:"}])

add(4, "pe", "loadconfig", "IMAGE_LOAD_CONFIG", "advanced",
"Write a C++ program parsing PE load config directory.",
r'''#include <windows.h>
#include <stdio.h>

int main() {
    printf("PE Load Config Parser\n\n");
    
    BYTE* base = (BYTE*)GetModuleHandleA(NULL);
    PIMAGE_DOS_HEADER dos = (PIMAGE_DOS_HEADER)base;
    PIMAGE_NT_HEADERS nt = (PIMAGE_NT_HEADERS)(base + dos->e_lfanew);
    
    DWORD lcRVA = nt->OptionalHeader.DataDirectory[IMAGE_DIRECTORY_ENTRY_LOAD_CONFIG].VirtualAddress;
    DWORD lcSize = nt->OptionalHeader.DataDirectory[IMAGE_DIRECTORY_ENTRY_LOAD_CONFIG].Size;
    
    printf("Load Config: RVA=0x%lX, Size=0x%lX\n\n", lcRVA, lcSize);
    
    if (!lcRVA) {
        printf("No load config directory\n");
        return 0;
    }
    
    PIMAGE_LOAD_CONFIG_DIRECTORY64 lc = (PIMAGE_LOAD_CONFIG_DIRECTORY64)(base + lcRVA);
    
    printf("Size: %lu\n", lc->Size);
    printf("SecurityCookie: 0x%llX\n", lc->SecurityCookie);
    printf("SEHandlerTable: 0x%llX\n", lc->SEHandlerTable);
    printf("SEHandlerCount: %llu\n", lc->SEHandlerCount);
    printf("GuardCFCheckFunctionPointer: 0x%llX\n", lc->GuardCFCheckFunctionPointer);
    printf("GuardFlags: 0x%lX\n", lc->GuardFlags);
    
    if (lc->GuardFlags & 0x100) printf("  CF_INSTRUMENTED\n");
    if (lc->GuardFlags & 0x200) printf("  CFW_INSTRUMENTED\n");
    if (lc->GuardFlags & 0x400) printf("  CF_FUNCTION_TABLE_PRESENT\n");
    
    return 0;
}''',
[{"type": "output_contains", "value": "SecurityCookie:"}])

add(4, "native", "section2", "NtMapViewOfSection", "advanced",
"Write a C++ program demonstrating section mapping.",
r'''#include <windows.h>
#include <stdio.h>

typedef LONG NTSTATUS;
typedef struct _OBJECT_ATTRIBUTES {
    ULONG Length;
    HANDLE RootDirectory;
    void* ObjectName;
    ULONG Attributes;
    void* SecurityDescriptor;
    void* SecurityQualityOfService;
} OBJECT_ATTRIBUTES;

typedef NTSTATUS (NTAPI *pNtCreateSection)(PHANDLE, ULONG, void*, PLARGE_INTEGER, ULONG, ULONG, HANDLE);
typedef NTSTATUS (NTAPI *pNtMapViewOfSection)(HANDLE, HANDLE, PVOID*, ULONG_PTR, SIZE_T, 
    PLARGE_INTEGER, PSIZE_T, DWORD, ULONG, ULONG);
typedef NTSTATUS (NTAPI *pNtUnmapViewOfSection)(HANDLE, PVOID);
typedef NTSTATUS (NTAPI *pNtClose)(HANDLE);

int main() {
    printf("NtMapViewOfSection Demo\n\n");
    
    HMODULE ntdll = GetModuleHandleA("ntdll.dll");
    pNtCreateSection NtCreateSection = (pNtCreateSection)GetProcAddress(ntdll, "NtCreateSection");
    pNtMapViewOfSection NtMapViewOfSection = (pNtMapViewOfSection)GetProcAddress(ntdll, "NtMapViewOfSection");
    pNtUnmapViewOfSection NtUnmapViewOfSection = (pNtUnmapViewOfSection)GetProcAddress(ntdll, "NtUnmapViewOfSection");
    pNtClose NtClose = (pNtClose)GetProcAddress(ntdll, "NtClose");
    
    // Create section
    HANDLE hSection;
    LARGE_INTEGER maxSize = {{0x10000, 0}};
    NTSTATUS status = NtCreateSection(&hSection, SECTION_ALL_ACCESS, NULL, &maxSize,
        PAGE_READWRITE, SEC_COMMIT, NULL);
    printf("NtCreateSection: 0x%lX\n", status);
    
    // Map into current process
    PVOID baseAddr = NULL;
    SIZE_T viewSize = 0;
    status = NtMapViewOfSection(hSection, GetCurrentProcess(), &baseAddr, 0, 0,
        NULL, &viewSize, 1, 0, PAGE_READWRITE);
    printf("NtMapViewOfSection: 0x%lX\n", status);
    printf("Base: 0x%p, Size: 0x%zX\n", baseAddr, viewSize);
    
    // Use the memory
    strcpy((char*)baseAddr, "Mapped section memory!");
    printf("Data: %s\n", (char*)baseAddr);
    
    // Unmap and close
    NtUnmapViewOfSection(GetCurrentProcess(), baseAddr);
    NtClose(hSection);
    printf("Section unmapped and closed\n");
    
    return 0;
}''',
[{"type": "output_contains", "value": "Mapped section memory!"}])

add(4, "threading", "injection", "ThreadInjection", "advanced",
"Write a C++ program demonstrating thread injection concepts.",
r'''#include <windows.h>
#include <stdio.h>

DWORD WINAPI InjectedFunc(LPVOID arg) {
    // This would run in target process if actually injected
    printf("[Injected] Running with param: 0x%p\n", arg);
    return 0x42424242;
}

int main() {
    printf("Thread Injection Concept Demo\n\n");
    
    // Demonstrate the concept using current process
    HANDLE hProc = GetCurrentProcess();
    
    // 1. Allocate memory in target
    LPVOID pRemote = VirtualAllocEx(hProc, NULL, 4096,
        MEM_COMMIT | MEM_RESERVE, PAGE_EXECUTE_READWRITE);
    printf("1. Allocated RWX memory: 0x%p\n", pRemote);
    
    // 2. Write code to remote memory (shellcode would go here)
    // For demo, we use a function pointer
    printf("2. Would write code here (skipping for demo)\n");
    
    // 3. Create remote thread
    HANDLE hThread = CreateRemoteThread(hProc, NULL, 0,
        (LPTHREAD_START_ROUTINE)InjectedFunc, (LPVOID)0xDEADBEEF, 0, NULL);
    printf("3. CreateRemoteThread: 0x%p\n", hThread);
    
    WaitForSingleObject(hThread, 1000);
    
    DWORD exitCode;
    GetExitCodeThread(hThread, &exitCode);
    printf("4. Thread exit code: 0x%lX\n", exitCode);
    
    CloseHandle(hThread);
    VirtualFreeEx(hProc, pRemote, 0, MEM_RELEASE);
    
    printf("\nNote: Real injection targets other processes\n");
    printf("and requires proper privilege and handle access.\n");
    
    return 0;
}''',
[{"type": "output_contains", "value": "CreateRemoteThread:"},
 {"type": "output_contains", "value": "Thread exit code: 0x42424242"}])

add(4, "evasion", "env_checks", "EnvironmentChecks", "advanced",
"Write a C++ program demonstrating environment fingerprinting.",
r'''#include <windows.h>
#include <psapi.h>
#include <stdio.h>
#include <intrin.h>
#pragma comment(lib, "psapi.lib")
#pragma comment(lib, "user32.lib")

int main() {
    printf("Environment Fingerprinting Demo\n\n");
    
    // 1. Screen resolution
    int cx = GetSystemMetrics(SM_CXSCREEN);
    int cy = GetSystemMetrics(SM_CYSCREEN);
    printf("[1] Screen: %dx%d %s\n", cx, cy,
        (cx < 800 || cy < 600) ? "(Suspicious)" : "");
    
    // 2. Number of monitors
    int monitors = GetSystemMetrics(SM_CMONITORS);
    printf("[2] Monitors: %d %s\n", monitors,
        monitors == 0 ? "(Suspicious)" : "");
    
    // 3. Cursor position (unchanged = no user)
    POINT pt1, pt2;
    GetCursorPos(&pt1);
    Sleep(100);
    GetCursorPos(&pt2);
    printf("[3] Cursor moved: %s\n",
        (pt1.x != pt2.x || pt1.y != pt2.y) ? "Yes" : "No");
    
    // 4. Uptime
    ULONGLONG uptime = GetTickCount64();
    printf("[4] Uptime: %.2f hours %s\n", uptime / (1000.0*60*60),
        uptime < 600000 ? "(Recent boot)" : "");
    
    // 5. Process count
    DWORD procs[1024];
    DWORD needed;
    EnumProcesses(procs, sizeof(procs), &needed);
    int procCount = needed / sizeof(DWORD);
    printf("[5] Processes: %d %s\n", procCount,
        procCount < 20 ? "(Low - suspicious)" : "");
    
    // 6. CPUID brand
    int cpuInfo[4];
    char brand[49] = {0};
    __cpuid(cpuInfo, 0x80000002);
    memcpy(brand, cpuInfo, 16);
    __cpuid(cpuInfo, 0x80000003);
    memcpy(brand + 16, cpuInfo, 16);
    __cpuid(cpuInfo, 0x80000004);
    memcpy(brand + 32, cpuInfo, 16);
    printf("[6] CPU: %s\n", brand);
    
    return 0;
}''',
[{"type": "output_contains", "value": "Screen:"},
 {"type": "output_contains", "value": "CPU:"}])

# =============================================================================
# OUTPUT GENERATION
# =============================================================================

def format_sft(p: dict, idx: int) -> dict:
    """Format for SFT training (ChatML)"""
    text = f"""<|im_start|>system
You are an expert Windows systems programmer.
<|im_end|>
<|im_start|>user
{p['prompt']}
<|im_end|>
<|im_start|>assistant
{p['solution']}
<|im_end|>"""
    
    return {
        "text": text,
        "metadata": {
            "source": "windows_curriculum",
            "tier": p["tier"],
            "category": p["category"],
            "subcategory": p["subcategory"],
            "api": p["api"],
            "difficulty": p["difficulty"],
            "tags": p.get("tags", [])
        }
    }


def format_rlvr(p: dict, idx: int) -> dict:
    """Format for RLVR training (with test cases)"""
    return {
        "id": f"t{p['tier']}_{p['category']}_{p['subcategory']}_{idx:03d}",
        "tier": p["tier"],
        "prompt": p["prompt"],
        "solution": p["solution"],
        "test_cases": p["test_cases"],
        "category": p["category"],
        "subcategory": p["subcategory"],
        "api": p["api"],
        "difficulty": p["difficulty"],
        "tags": p.get("tags", []),
        "verification_strategy": p.get("verification_strategy", "stdout_contains"),
        "timeout_seconds": 30,
        "requires_admin": False
    }


def main():
    BASE_DIR.mkdir(exist_ok=True)
    
    # Sort by tier for curriculum ordering
    sorted_problems = sorted(PROBLEMS, key=lambda x: (x["tier"], x["category"], x["difficulty"]))
    
    # Generate SFT dataset
    sft_file = BASE_DIR / "windows_curriculum_sft.jsonl"
    with open(sft_file, 'w') as f:
        for i, p in enumerate(sorted_problems):
            f.write(json.dumps(format_sft(p, i)) + '\n')
    
    # Generate RLVR dataset
    rlvr_file = BASE_DIR / "windows_curriculum_rlvr.jsonl"
    with open(rlvr_file, 'w') as f:
        for i, p in enumerate(sorted_problems):
            f.write(json.dumps(format_rlvr(p, i)) + '\n')
    
    # Generate curriculum order file
    curriculum = {"tiers": {}, "total": len(PROBLEMS)}
    for p in PROBLEMS:
        tier = f"tier_{p['tier']}"
        if tier not in curriculum["tiers"]:
            curriculum["tiers"][tier] = {"count": 0, "categories": {}}
        curriculum["tiers"][tier]["count"] += 1
        cat = p["category"]
        if cat not in curriculum["tiers"][tier]["categories"]:
            curriculum["tiers"][tier]["categories"][cat] = 0
        curriculum["tiers"][tier]["categories"][cat] += 1
    
    with open(BASE_DIR / "curriculum_order.json", 'w') as f:
        json.dump(curriculum, f, indent=2)
    
    print(f"Generated {len(PROBLEMS)} curriculum problems")
    print(f"SFT output: {sft_file}")
    print(f"RLVR output: {rlvr_file}")
    
    # Stats
    print("\nBy Tier:")
    for tier in sorted(set(p["tier"] for p in PROBLEMS)):
        count = sum(1 for p in PROBLEMS if p["tier"] == tier)
        print(f"  Tier {tier}: {count}")
    
    print("\nBy Category:")
    cats = {}
    for p in PROBLEMS:
        cats[p["category"]] = cats.get(p["category"], 0) + 1
    for c, n in sorted(cats.items()):
        print(f"  {c}: {n}")
    
    print("\nBy Difficulty:")
    diffs = {}
    for p in PROBLEMS:
        diffs[p["difficulty"]] = diffs.get(p["difficulty"], 0) + 1
    for d, n in sorted(diffs.items()):
        print(f"  {d}: {n}")


if __name__ == "__main__":
    main()

