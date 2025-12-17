#include "InputMonitor.hpp"
#include <iostream>

HHOOK InputMonitor::hKeyboardHook = NULL;
HHOOK InputMonitor::hMouseHook = NULL;
std::atomic<int> InputMonitor::key_count(0);
std::atomic<int> InputMonitor::click_count(0);

InputMonitor::InputMonitor() : running(false), threadId(0) {}

InputMonitor::~InputMonitor() {
    stop();
}

void InputMonitor::start() {
    if (running) return;
    running = true;
    monitor_thread = std::thread(&InputMonitor::loop, this);
}

void InputMonitor::stop() {
    if (!running) return;
    running = false;
    if (threadId != 0) {
        PostThreadMessage(threadId, WM_QUIT, 0, 0);
    }
    if (monitor_thread.joinable()) {
        monitor_thread.join();
    }
}

void InputMonitor::get_and_reset_counts(int& keys, int& clicks) {
    keys = key_count.exchange(0);
    clicks = click_count.exchange(0);
}

LRESULT CALLBACK InputMonitor::KeyboardHookProc(int nCode, WPARAM wParam, LPARAM lParam) {
    if (nCode >= 0) {
        if (wParam == WM_KEYDOWN || wParam == WM_SYSKEYDOWN) {
            key_count++;
        }
    }
    return CallNextHookEx(hKeyboardHook, nCode, wParam, lParam);
}

LRESULT CALLBACK InputMonitor::MouseHookProc(int nCode, WPARAM wParam, LPARAM lParam) {
    if (nCode >= 0) {
        if (wParam == WM_LBUTTONDOWN || wParam == WM_RBUTTONDOWN || wParam == WM_MBUTTONDOWN) {
            click_count++;
        }
    }
    return CallNextHookEx(hMouseHook, nCode, wParam, lParam);
}

void InputMonitor::loop() {
    threadId = GetCurrentThreadId();
    hKeyboardHook = SetWindowsHookEx(WH_KEYBOARD_LL, KeyboardHookProc, GetModuleHandle(NULL), 0);
    hMouseHook = SetWindowsHookEx(WH_MOUSE_LL, MouseHookProc, GetModuleHandle(NULL), 0);

    MSG msg;
    // Message loop to keep hooks alive
    while (GetMessage(&msg, NULL, 0, 0)) {
        TranslateMessage(&msg);
        DispatchMessage(&msg);
    }

    UnhookWindowsHookEx(hKeyboardHook);
    UnhookWindowsHookEx(hMouseHook);
    threadId = 0;
}
